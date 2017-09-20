#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Worker that receives input from Piped RDD.
"""
from __future__ import print_function
import os
import sys
import time
import socket
import traceback

if sys.version < '3':
    from itertools import imap as map

from pyspark.accumulators import _accumulatorRegistry
from pyspark.broadcast import Broadcast, _broadcastRegistry
from pyspark.taskcontext import TaskContext
from pyspark.files import SparkFiles
from pyspark.serializers import write_with_length, write_int, read_long, \
    write_long, read_int, SpecialLengths, UTF8Deserializer, PickleSerializer, \
    BatchedSerializer, VectorizedSerializer
from pyspark import shuffle
from pyspark.sql.types import StructType, toArrowSchema

pickleSer = PickleSerializer()
utf8_deserializer = UTF8Deserializer()


class PythonEvalType(object):
    NON_UDF = 0
    SQL_BATCHED_UDF = 1
    SQL_VECTORIZED_UDF = 2
    SQL_VECTORIZED_UDAF = 3


def report_times(outfile, boot, init, finish):
    write_int(SpecialLengths.TIMING_DATA, outfile)
    write_long(int(1000 * boot), outfile)
    write_long(int(1000 * init), outfile)
    write_long(int(1000 * finish), outfile)


def add_path(path):
    # worker can be used, so donot add path multiple times
    if path not in sys.path:
        # overwrite system packages
        sys.path.insert(1, path)


def read_command(serializer, file):
    command = serializer._read_with_length(file)
    if isinstance(command, Broadcast):
        command = serializer.loads(command.value)
    return command


def chain(f, g):
    """chain two function together """
    return lambda *a: g(f(*a))


def wrap_udf(f, return_type):
    if return_type.needConversion():
        toInternal = return_type.toInternal
        return lambda *a: toInternal(f(*a))
    else:
        return lambda *a: f(*a)


def read_single_udf(pickleSer, infile, vectorized=False):
    num_arg = read_int(infile)
    arg_offsets = [read_int(infile) for i in range(num_arg)]
    func = None
    for i in range(read_int(infile)):
        f, return_type = read_command(pickleSer, infile)
        if func is None:
            func = f
        else:
            func = chain(func, f)
    if (vectorized):
        # todo: shall we do conversion depends on data type?
        return arg_offsets, func, return_type
    else:
        # the last returnType will be the return type of UDF
        return arg_offsets, wrap_udf(func, return_type)


def read_udfs(pickleSer, infile):
    num_udfs = read_int(infile)
    udfs = {}
    call_udf = []
    for i in range(num_udfs):
        arg_offsets, udf = read_single_udf(pickleSer, infile)
        udfs['f%d' % i] = udf
        args = ["a[%d]" % o for o in arg_offsets]
        call_udf.append("f%d(%s)" % (i, ", ".join(args)))
    # Create function like this:
    #   lambda a: (f0(a0), f1(a1, a2), f2(a3))
    # In the special case of a single UDF this will return a single result rather
    # than a tuple of results; this is the format that the JVM side expects.
    mapper_str = "lambda a: (%s)" % (", ".join(call_udf))
    mapper = eval(mapper_str, udfs)

    func = lambda _, it: map(mapper, it)
    ser = BatchedSerializer(PickleSerializer(), 100)
    # profiling is not supported for UDF
    return func, None, ser, ser


def read_vectorized_udfs(pickleSer, infile):
    num_udfs = read_int(infile)
    udfs = {}
    call_udfs = []
    types = []
    for i in range(num_udfs):
        arg_offsets, udf, return_type = read_single_udf(pickleSer, infile, True)
        types.append(return_type)
        udfs['f%d' % i] = udf
        # the first value of the inputs is the number of elements in this batch, and we only
        # need it for 0-parameter UDF.
        if arg_offsets:
            args = ["a[%d]" % (o + 1) for o in arg_offsets]
        else:
            args = ["a[0]"]
        call_udfs.append("f%d(%s)" % (i, ", ".join(args)))

    def chk_len(v, size):
        if len(v) == size:
            return v
        else:
            raise Exception("The length of returned value should be the same as input value")

    call_and_chk_len = ['chk_len(%s, a[0])' % call_udf for call_udf in call_udfs]
    udfs['chk_len'] = chk_len
    # Create function like this:
    #   lambda a: [chk_len(f0(a[0]), a[0]), chk_len(f1(a[1], a[2]), a[0]), ...]
    mapper_str = "lambda a: [%s]" % (", ".join(call_and_chk_len))
    mapper = eval(mapper_str, udfs)

    func = lambda _, it: map(mapper, it)
    ser = VectorizedSerializer()
    ser.schema = toArrowSchema(types)
    # profiling is not supported for UDF
    return func, None, ser, ser


def read_vectorized_udafs(pickleSer, infile):
    import pandas as pd
    num_udfs = read_int(infile)
    udfs = {}
    call_udf = []
    types = []
    for i in range(num_udfs):
        arg_offsets, udf, return_type = read_single_udf(pickleSer, infile, True)
        if type(return_type) == StructType:
            for field in return_type.fields:
                types.append(field.dataType)
        else:
            types.append(return_type)
        udfs['f%d' % i] = udf
        args = ["a[%d]" % o for o in arg_offsets]
        call_udf.append("untuplize(f%d(%s))" % (i, ", ".join(args)))

    def untuplize(v):
        if not isinstance(v, tuple):
            return (v,)
        else:
            return v

    udfs['untuplize'] = untuplize

    # Create function like this:
    #   lambda a: [f0(a[0]), f1(a[1], a[2]), f2(a[3])]
    apply_str = "lambda a: [%s]" % (", ".join(call_udf))
    applyer = eval(apply_str, udfs)

    def mapper(vectors):
        num_rows = vectors[0]
        vs = vectors[1:]
        results = [list(sum(applyer([v[r] for v in vs]), ())) for r in range(num_rows)]
        return [pd.Series(x) for x in zip(*results)]

    func = lambda _, it: map(mapper, it)
    ser = VectorizedSerializer()
    ser.schema = toArrowSchema(types)
    # profiling is not supported for UDF
    return func, None, ser, ser


def main(infile, outfile):
    try:
        boot_time = time.time()
        split_index = read_int(infile)
        if split_index == -1:  # for unit tests
            exit(-1)

        version = utf8_deserializer.loads(infile)
        if version != "%d.%d" % sys.version_info[:2]:
            raise Exception(("Python in worker has different version %s than that in " +
                             "driver %s, PySpark cannot run with different minor versions." +
                             "Please check environment variables PYSPARK_PYTHON and " +
                             "PYSPARK_DRIVER_PYTHON are correctly set.") %
                            ("%d.%d" % sys.version_info[:2], version))

        # initialize global state
        taskContext = TaskContext._getOrCreate()
        taskContext._stageId = read_int(infile)
        taskContext._partitionId = read_int(infile)
        taskContext._attemptNumber = read_int(infile)
        taskContext._taskAttemptId = read_long(infile)
        shuffle.MemoryBytesSpilled = 0
        shuffle.DiskBytesSpilled = 0
        _accumulatorRegistry.clear()

        # fetch name of workdir
        spark_files_dir = utf8_deserializer.loads(infile)
        SparkFiles._root_directory = spark_files_dir
        SparkFiles._is_running_on_worker = True

        # fetch names of includes (*.zip and *.egg files) and construct PYTHONPATH
        add_path(spark_files_dir)  # *.py files that were added will be copied here
        num_python_includes = read_int(infile)
        for _ in range(num_python_includes):
            filename = utf8_deserializer.loads(infile)
            add_path(os.path.join(spark_files_dir, filename))
        if sys.version > '3':
            import importlib
            importlib.invalidate_caches()

        # fetch names and values of broadcast variables
        num_broadcast_variables = read_int(infile)
        for _ in range(num_broadcast_variables):
            bid = read_long(infile)
            if bid >= 0:
                path = utf8_deserializer.loads(infile)
                _broadcastRegistry[bid] = Broadcast(path=path)
            else:
                bid = - bid - 1
                _broadcastRegistry.pop(bid)

        _accumulatorRegistry.clear()
        mode = read_int(infile)
        if mode == PythonEvalType.SQL_VECTORIZED_UDAF:
            func, profiler, deserializer, serializer = read_vectorized_udafs(pickleSer, infile)
        elif mode == PythonEvalType.SQL_VECTORIZED_UDF:
            func, profiler, deserializer, serializer = read_vectorized_udfs(pickleSer, infile)
        elif mode == PythonEvalType.SQL_BATCHED_UDF:
            func, profiler, deserializer, serializer = read_udfs(pickleSer, infile)
        else:
            func, profiler, deserializer, serializer = read_command(pickleSer, infile)

        init_time = time.time()

        def process():
            iterator = deserializer.load_stream(infile)
            serializer.dump_stream(func(split_index, iterator), outfile)

        if profiler:
            profiler.profile(process)
        else:
            process()
    except Exception:
        try:
            write_int(SpecialLengths.PYTHON_EXCEPTION_THROWN, outfile)
            write_with_length(traceback.format_exc().encode("utf-8"), outfile)
        except IOError:
            # JVM close the socket
            pass
        except Exception:
            # Write the error to stderr if it happened while serializing
            print("PySpark worker failed with exception:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
        exit(-1)
    finish_time = time.time()
    report_times(outfile, boot_time, init_time, finish_time)
    write_long(shuffle.MemoryBytesSpilled, outfile)
    write_long(shuffle.DiskBytesSpilled, outfile)

    # Mark the beginning of the accumulators section of the output
    write_int(SpecialLengths.END_OF_DATA_SECTION, outfile)
    write_int(len(_accumulatorRegistry), outfile)
    for (aid, accum) in _accumulatorRegistry.items():
        pickleSer._write_with_length((aid, accum._value), outfile)

    # check end of stream
    if read_int(infile) == SpecialLengths.END_OF_STREAM:
        write_int(SpecialLengths.END_OF_STREAM, outfile)
    else:
        # write a different value to tell JVM to not reuse this worker
        write_int(SpecialLengths.END_OF_DATA_SECTION, outfile)
        exit(-1)


if __name__ == '__main__':
    # Read a local port to connect to from stdin
    java_port = int(sys.stdin.readline())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", java_port))
    sock_file = sock.makefile("rwb", 65536)
    main(sock_file, sock_file)

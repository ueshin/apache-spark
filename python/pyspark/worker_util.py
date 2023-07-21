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
Util functions for workers.
"""
import importlib
import os
import sys
from typing import Any, IO

from pyspark.broadcast import Broadcast
from pyspark.errors import PySparkRuntimeError
from pyspark.files import SparkFiles
from pyspark.serializers import (
    read_int,
    FramedSerializer,
    UTF8Deserializer,
    CPickleSerializer,
)

pickleSer = CPickleSerializer()
utf8_deserializer = UTF8Deserializer()


def add_path(path: str) -> None:
    # worker can be used, so do not add path multiple times
    if path not in sys.path:
        # overwrite system packages
        sys.path.insert(1, path)


def read_command(serializer: FramedSerializer, file: IO) -> Any:
    command = serializer._read_with_length(file)
    if isinstance(command, Broadcast):
        command = serializer.loads(command.value)
    return command


def check_python_version(infile: IO) -> None:
    """
    Check the Python version between the running process and the one used to serialize the command.
    """
    version = utf8_deserializer.loads(infile)
    if version != "%d.%d" % sys.version_info[:2]:
        raise PySparkRuntimeError(
            error_class="PYTHON_VERSION_MISMATCH",
            message_parameters={
                "worker_version": str(sys.version_info[:2]),
                "driver_version": str(version),
            },
        )


def setup_spark_files(infile: IO) -> None:
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

    importlib.invalidate_caches()

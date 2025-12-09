/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.sql.execution.python

import java.io.DataOutputStream
import java.net.ServerSocket
import java.nio.channels.Channels

import scala.collection.concurrent
import scala.util.Random

import org.apache.arrow.compression.{Lz4CompressionCodec, ZstdCompressionCodec}
import org.apache.arrow.flight._
import org.apache.arrow.vector.{VectorSchemaRoot, VectorUnloader}
import org.apache.arrow.vector.compression.{CompressionCodec, NoCompressionCodec}
import org.apache.arrow.vector.ipc.ArrowStreamWriter
import org.apache.arrow.vector.ipc.WriteChannel
import org.apache.arrow.vector.ipc.message.MessageSerializer

import org.apache.spark.{SparkEnv, SparkException, TaskContext}
import org.apache.spark.api.python.{BasePythonRunner, PythonWorker}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.arrow
import org.apache.spark.sql.execution.arrow.{ArrowWriter, ArrowWriterWrapper}
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.execution.python.PythonArrowFlightInput.releasePort
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.ArrowUtils
import org.apache.spark.util.Utils

/**
 * A trait that can be mixed-in with [[BasePythonRunner]]. It implements the logic from
 * JVM (an iterator of internal rows + additional data if required) to Python (Arrow).
 */
private[python] trait PythonArrowInput[IN] {
  self: BasePythonRunner[IN, _] =>
  protected val schema: StructType

  protected val timeZoneId: String

  protected val errorOnDuplicatedFieldNames: Boolean

  protected val largeVarTypes: Boolean

  protected def pythonMetrics: Map[String, SQLMetric]

  protected def writeUDF(dataOut: DataOutputStream): Unit

  protected def handleMetadataBeforeExec(stream: DataOutputStream): Unit = {}

  private val arrowSchema = ArrowUtils.toArrowSchema(
    schema, timeZoneId, errorOnDuplicatedFieldNames, largeVarTypes)
  protected val allocator =
    ArrowUtils.rootAllocator.newChildAllocator(s"stdout writer for $pythonExec", 0, Long.MaxValue)
  protected val root = VectorSchemaRoot.create(arrowSchema, allocator)

  // Create compression codec based on config
  protected val compressionCodecName = SQLConf.get.arrowCompressionCodec
  protected val codec = compressionCodecName match {
    case "none" => NoCompressionCodec.INSTANCE
    case "zstd" =>
      val compressionLevel = SQLConf.get.arrowZstdCompressionLevel
      val factory = CompressionCodec.Factory.INSTANCE
      val codecType = new ZstdCompressionCodec(compressionLevel).getCodecType()
      factory.createCodec(codecType)
    case "lz4" =>
      val factory = CompressionCodec.Factory.INSTANCE
      val codecType = new Lz4CompressionCodec().getCodecType()
      factory.createCodec(codecType)
    case other =>
      throw SparkException.internalError(
        s"Unsupported Arrow compression codec: $other. Supported values: none, zstd, lz4")
  }
}

private[python] trait PythonArrowStreamInput[IN] {
  self: BasePythonRunner[IN, _] with PythonArrowInput[IN] =>
  protected val unloader = new VectorUnloader(root, true, codec, true)

  protected var writer: ArrowStreamWriter = _

  protected def writeNextBatchToArrowStream(
      root: VectorSchemaRoot,
      writer: ArrowStreamWriter,
      dataOut: DataOutputStream,
      inputIterator: Iterator[IN]): Boolean

  protected def close(): Unit = {
    Utils.tryWithSafeFinally {
      // end writes footer to the output stream and doesn't clean any resources.
      // It could throw exception if the output stream is closed, so it should be
      // in the try block.
      writer.end()
    } {
      root.close()
      allocator.close()
    }
  }

  protected override def newWriter(
      env: SparkEnv,
      worker: PythonWorker,
      inputIterator: Iterator[IN],
      partitionIndex: Int,
      context: TaskContext): Writer = {
    new Writer(env, worker, inputIterator, partitionIndex, context) {

      protected override def writeCommand(dataOut: DataOutputStream): Unit = {
        handleMetadataBeforeExec(dataOut)
        writeUDF(dataOut)
      }

      override def writeNextInputToStream(dataOut: DataOutputStream): Boolean = {

        if (writer == null) {
          writer = new ArrowStreamWriter(root, null, dataOut)
          writer.start()
        }

        assert(writer != null)
        writeNextBatchToArrowStream(root, writer, dataOut, inputIterator)
      }
    }
  }
}

private[python] trait BasicPythonArrowInput
  extends PythonArrowStreamInput[Iterator[InternalRow]] {
  self: BasePythonRunner[Iterator[InternalRow], _] with PythonArrowInput[Iterator[InternalRow]] =>
  protected val arrowWriter: arrow.ArrowWriter = ArrowWriter.create(root)

  protected val maxRecordsPerBatch: Int = {
    val v = SQLConf.get.arrowMaxRecordsPerBatch
    if (v > 0) v else Int.MaxValue
  }

  protected val maxBytesPerBatch: Long = SQLConf.get.arrowMaxBytesPerBatch

  protected def writeNextBatchToArrowStream(
      root: VectorSchemaRoot,
      writer: ArrowStreamWriter,
      dataOut: DataOutputStream,
      inputIterator: Iterator[Iterator[InternalRow]]): Boolean = {

    if (inputIterator.hasNext) {
      val startData = dataOut.size()
      val nextBatch = inputIterator.next()

      while (nextBatch.hasNext) {
        arrowWriter.write(nextBatch.next())
      }

      arrowWriter.finish()
      // Use unloader to get compressed batch and write it manually
      val batch = unloader.getRecordBatch()
      try {
        val writeChannel = new WriteChannel(Channels.newChannel(dataOut))
        MessageSerializer.serialize(writeChannel, batch)
      } finally {
        batch.close()
      }
      arrowWriter.reset()
      val deltaData = dataOut.size() - startData
      pythonMetrics("pythonDataSent") += deltaData
      true
    } else {
      super[PythonArrowStreamInput].close()
      false
    }
  }
}


private[python] trait BatchedPythonArrowInput extends BasicPythonArrowInput {
  self: BasePythonRunner[Iterator[InternalRow], _] with PythonArrowInput[Iterator[InternalRow]] =>
  // Marker inside the input iterator to indicate the start of the next batch.
  private var nextBatchStart: Iterator[InternalRow] = Iterator.empty

  override protected def writeNextBatchToArrowStream(
      root: VectorSchemaRoot,
      writer: ArrowStreamWriter,
      dataOut: DataOutputStream,
      inputIterator: Iterator[Iterator[InternalRow]]): Boolean = {
    if (!nextBatchStart.hasNext) {
      if (inputIterator.hasNext) {
        nextBatchStart = inputIterator.next()
      }
    }
    if (nextBatchStart.hasNext) {
      val startData = dataOut.size()

      val numRowsInBatch = BatchedPythonArrowInput.writeSizedBatch(
        arrowWriter, writer, nextBatchStart, maxBytesPerBatch, maxRecordsPerBatch, unloader,
        dataOut)
      assert(0 < numRowsInBatch && numRowsInBatch <= maxRecordsPerBatch, numRowsInBatch)

      val deltaData = dataOut.size() - startData
      pythonMetrics("pythonDataSent") += deltaData
      true
    } else {
      super[BasicPythonArrowInput].close()
      false
    }
  }
}

private[python] object BatchedPythonArrowInput {
  /**
   * Split a group into smaller Arrow batches within
   * a separate and complete Arrow streaming format in order
   * to work around Arrow 2G limit, see ARROW-4890.
   *
   * The return value is the number of rows in the batch.
   * Each split Arrow batch also does not have mixed grouped. For example:
   *
   *        +------------------------+      +------------------------+      +--------------------
   *        |Group (by k1) v1, v2, v3|      |Group (by k2) v1, v2, v3|      |                 ...
   *        +------------------------+      +------------------------+      +--------------------
   *
   * +------+-----------------+------+------+-----------------+------+------+--------------------
   * |Schema|            Batch| Batch|Schema|            Batch| Batch|Schema|           Batch ...
   * +------+-----------------+------+------+-----------------+------+------+--------------------
   * |    Arrow Streaming Format     |    Arrow Streaming Format     |    Arrow Streaming Form...
   *
   * Here, each (Arrow) batch does not span multiple groups.
   * These (Arrow) batches within each complete Arrow IPC Format are
   * reconstructed into the group back as pandas instances later on the Python worker side.
   */
  def writeSizedBatch(
      arrowWriter: ArrowWriter,
      writer: ArrowStreamWriter,
      rowIter: Iterator[InternalRow],
      maxBytesPerBatch: Long,
      maxRecordsPerBatch: Int,
      unloader: VectorUnloader,
      dataOut: DataOutputStream): Int = {
    var numRowsInBatch: Int = 0

    def underBatchSizeLimit: Boolean =
      (maxBytesPerBatch == Int.MaxValue) || (arrowWriter.sizeInBytes() < maxBytesPerBatch)

    while (rowIter.hasNext && numRowsInBatch < maxRecordsPerBatch &&
      underBatchSizeLimit) {
      arrowWriter.write(rowIter.next())
      numRowsInBatch += 1
    }
    arrowWriter.finish()
    // Use unloader to get compressed batch and write it manually
    val batch = unloader.getRecordBatch()
    try {
      val writeChannel = new WriteChannel(Channels.newChannel(dataOut))
      MessageSerializer.serialize(writeChannel, batch)
    } finally {
      batch.close()
    }
    arrowWriter.reset()
    numRowsInBatch
  }
}

/**
 * Enables an optimization that splits each group into the sized batches.
 */
private[python] trait GroupedPythonArrowInput { self: RowInputArrowPythonRunner =>

  // Helper method to create VectorUnloader with compression for grouped operations
  private def createUnloaderForGroup(root: VectorSchemaRoot): VectorUnloader = {
    val codec = SQLConf.get.arrowCompressionCodec match {
      case "none" => NoCompressionCodec.INSTANCE
      case "zstd" =>
        val compressionLevel = SQLConf.get.arrowZstdCompressionLevel
        val factory = CompressionCodec.Factory.INSTANCE
        val codecType = new ZstdCompressionCodec(compressionLevel).getCodecType()
        factory.createCodec(codecType)
      case "lz4" =>
        val factory = CompressionCodec.Factory.INSTANCE
        val codecType = new Lz4CompressionCodec().getCodecType()
        factory.createCodec(codecType)
      case other =>
        throw SparkException.internalError(
          s"Unsupported Arrow compression codec: $other. Supported values: none, zstd, lz4")
    }
    new VectorUnloader(root, true, codec, true)
  }
  protected override def newWriter(
      env: SparkEnv,
      worker: PythonWorker,
      inputIterator: Iterator[Iterator[InternalRow]],
      partitionIndex: Int,
      context: TaskContext): Writer = {
    new Writer(env, worker, inputIterator, partitionIndex, context) {
      protected override def writeCommand(dataOut: DataOutputStream): Unit = {
        writeUDF(dataOut)
      }

      var writer: ArrowWriterWrapper = null
      // Marker inside the input iterator to indicate the start of the next batch.
      private var nextBatchStart: Iterator[InternalRow] = Iterator.empty

      override def writeNextInputToStream(dataOut: DataOutputStream): Boolean = {
        if (!nextBatchStart.hasNext) {
          if (inputIterator.hasNext) {
            dataOut.writeInt(1) // Notify that there is a group to read.
            assert(writer == null || writer.isClosed)
            writer = ArrowWriterWrapper.createAndStartArrowWriter(
              schema, timeZoneId, pythonExec,
              errorOnDuplicatedFieldNames, largeVarTypes, dataOut, context)
            // Set the unloader with compression after creating the writer
            writer.unloader = createUnloaderForGroup(writer.root)
            nextBatchStart = inputIterator.next()
          }
        }
        if (nextBatchStart.hasNext) {
          val startData = dataOut.size()
          val numRowsInBatch: Int = BatchedPythonArrowInput.writeSizedBatch(writer.arrowWriter,
            writer.streamWriter, nextBatchStart, maxBytesPerBatch, maxRecordsPerBatch,
            writer.unloader, dataOut)
          if (!nextBatchStart.hasNext) {
            writer.streamWriter.end()
            // We don't need a try catch block here as the close() method is registered with
            // the TaskCompletionListener.
            writer.close()
          }
          assert(0 < numRowsInBatch && numRowsInBatch <= maxRecordsPerBatch, numRowsInBatch)
          val deltaData = dataOut.size() - startData
          pythonMetrics("pythonDataSent") += deltaData
          true
        } else {
          dataOut.writeInt(0) // End of data is marked by sending 0.
          false
        }
      }
    }
  }
}

object PythonArrowFlightInput {
  private val runningPorts: concurrent.Map[Int, Int] = concurrent.TrieMap.empty

  private def available(port: Int): Boolean = {
    var s: ServerSocket = null
    try {
      s = new ServerSocket(port)
      true
    } catch {
      case _: Exception =>
        false
    } finally {
      if (s != null) s.close()
    }
  }

  @scala.annotation.tailrec
  def getAvailablePort(): Int = {
    val port = Random.between(32768, 65535)
    synchronized {
      if (available(port) && !runningPorts.contains(port)) {
        runningPorts.put(port, port)
        return port
      }
    }
    getAvailablePort()
  }

  def releasePort(port: Int): Unit = {
    runningPorts.remove(port)
  }
}

private[python] trait PythonArrowFlightInput {
  self: BasePythonRunner[Iterator[InternalRow], _] with PythonArrowInput[Iterator[InternalRow]] =>
  protected val arrowWriter: arrow.ArrowWriter

  protected override def newWriter(
      env: SparkEnv,
      worker: PythonWorker,
      inputIterator: Iterator[Iterator[InternalRow]],
      partitionIndex: Int,
      context: TaskContext): Writer = {
    new Writer(env, worker, inputIterator, partitionIndex, context) {

      private val port = PythonArrowFlightInput.getAvailablePort()
      private val location = Location.forGrpcInsecure("localhost", port)
      private var flightClient: FlightClient = _
      private var writer: FlightClient.ClientStreamListener = _

      protected override def writeCommand(dataOut: DataOutputStream): Unit = {
        dataOut.writeInt(port)
        handleMetadataBeforeExec(dataOut)
        writeUDF(dataOut)
      }

      override def writeNextInputToStream(dataOut: DataOutputStream): Boolean = {
        if (flightClient == null) {
          if (inputIterator.hasNext) {
            Thread.sleep(100)
            flightClient = FlightClient.builder(allocator, location).build()
            val descriptor = FlightDescriptor.path(s"batch_${System.nanoTime()}")
            writer = flightClient.startPut(descriptor, root, new SyncPutListener())
            dataOut.writeBoolean(true)
          } else {
            dataOut.writeBoolean(false)
          }
        }

        if (inputIterator.hasNext) {
          val nextBatch = inputIterator.next()

          while (nextBatch.hasNext) {
            arrowWriter.write(nextBatch.next())
          }
          arrowWriter.finish()
          writer.putNext()

          arrowWriter.reset()
          true
        } else {
          writer.completed()
          writer.getResult()
          if (flightClient != null) {
            flightClient.close()
            flightClient = null
          }
          releasePort(port)
          false
        }
      }
    }
  }
}

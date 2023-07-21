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

package org.apache.spark.api.python

import java.io.{DataOutputStream, File}
import java.nio.charset.StandardCharsets

import org.apache.spark.SparkFiles

private[spark] object PythonWorkerUtils {

  /**
   * Write a string in UTF-8 charset.
   *
   * It will be read by `UTF8Deserializer.loads` in Python.
   */
  def writeUTF(str: String, dataOut: DataOutputStream): Unit = {
    val bytes = str.getBytes(StandardCharsets.UTF_8)
    dataOut.writeInt(bytes.length)
    dataOut.write(bytes)
  }

  /**
   * Write a Python version to check if the Python version is expected.
   *
   * It will be read and checked by `worker_util.check_python_version`.
   */
  def writePythonVersion(pythonVer: String, dataOut: DataOutputStream): Unit = {
    writeUTF(pythonVer, dataOut)
  }

  /**
   * Write Spark files to set up them in the worker.
   *
   * It will be read and used by `worker_util.setup_spark_files`.
   */
  def writeSparkFiles(
      jobArtifactUUID: Option[String],
      pythonIncludes: Set[String],
      dataOut: DataOutputStream): Unit = {
    // sparkFilesDir
    val root = jobArtifactUUID.map { uuid =>
      new File(SparkFiles.getRootDirectory(), uuid).getAbsolutePath
    }.getOrElse(SparkFiles.getRootDirectory())
    PythonRDD.writeUTF(root, dataOut)

    // Python includes (*.zip and *.egg files)
    dataOut.writeInt(pythonIncludes.size)
    for (include <- pythonIncludes) {
      PythonRDD.writeUTF(include, dataOut)
    }
  }
}

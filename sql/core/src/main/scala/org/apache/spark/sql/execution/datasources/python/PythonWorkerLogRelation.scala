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

package org.apache.spark.sql.execution.datasources.python

import org.apache.spark.SparkEnv
import org.apache.spark.rdd.{BlockRDD, RDD}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.classic.SparkSession
import org.apache.spark.sql.functions.{col, from_json}
import org.apache.spark.sql.sources.{BaseRelation, TableScan}
import org.apache.spark.sql.types.StructType
import org.apache.spark.storage.{BlockId, PythonWorkerLogBlockId, PythonWorkerLogLine}
import org.apache.spark.util.LogUtils

class PythonWorkerLogRelation(
    @transient val sparkSession: SparkSession)
  extends BaseRelation with TableScan {

  override def sqlContext: SQLContext = sparkSession.sqlContext

  override def schema: StructType = StructType.fromDDL(LogUtils.SPARK_LOG_SCHEMA)

  override def buildScan(): RDD[Row] = {
    import sparkSession.implicits._

    new BlockRDD[PythonWorkerLogLine](
      sparkSession.sparkContext, findBlocks(sparkSession.sessionUUID))
      .toDF()
      .withColumn("json", from_json(col("message"), schema))
      .orderBy("eventTime", "sequenceId")
      .select("json.*")
      .rdd
  }

  private def findBlocks(sessionId: String): Array[BlockId] = {
    SparkEnv.get.blockManager.master.getMatchingBlockIds(
      id => id.isInstanceOf[PythonWorkerLogBlockId] && {
        val blockId = id.asInstanceOf[PythonWorkerLogBlockId]
        blockId.sessionId == sessionId
      }, askStorageEndpoints = true
    ).distinct.toArray
  }

  override def toString: String = "PythonWorkerLog"
}

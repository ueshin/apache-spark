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

import org.apache.spark.api.python.PythonFunction
import org.apache.spark.sql.Column
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.types._

/**
 * A user-defined aggregate Python function. This is used by the Python API.
 */
case class UserDefinedAggregatePythonFunction(
    name: String,
    evaluate: PythonFunction,
    returnType: DataType,
    partial: PythonFunction,
    bufferType: DataType) {

  /**
   * Creates a `Column` for this UDAF using given `Column`s as input arguments.
   *
   * @since 2.3.0
   */
  @scala.annotation.varargs
  def apply(exprs: Column*): Column = {
    val aggregateExpression =
      AggregateExpression(
        PythonUDAF(exprs.map(_.expr), this),
        Complete,
        isDistinct = false)
    Column(aggregateExpression)
  }

  /**
   * Creates a `Column` for this UDAF using the distinct values of the given
   * `Column`s as input arguments.
   *
   * @since 2.3.0
   */
  @scala.annotation.varargs
  def distinct(exprs: Column*): Column = {
    val aggregateExpression =
      AggregateExpression(
        PythonUDAF(exprs.map(_.expr), this),
        Complete,
        isDistinct = true)
    Column(aggregateExpression)
  }
}

/**
 * The internal wrapper used to hook a [[UserDefinedAggregatePythonFunction]] `udaf` in the
 * internal aggregation code path.
 */
case class PythonUDAF(
    children: Seq[Expression],
    udaf: UserDefinedAggregatePythonFunction)
  extends AggregateFunction
  with Unevaluable
  with NonSQLExpression
  with UserDefinedExpression {

  override def nullable: Boolean = true

  override def dataType: DataType = udaf.returnType

  override val aggBufferSchema: StructType = udaf.bufferType match {
    case StructType(fields) => StructType(fields.map { field =>
      StructField(s"${udaf.name}.${field.name}",
        ArrayType(field.dataType, containsNull = field.nullable), nullable = false)
    })
    case dt => new StructType().add(udaf.name, ArrayType(dt, containsNull = true), nullable = false)
  }

  override val aggBufferAttributes: Seq[AttributeReference] = aggBufferSchema.toAttributes

  // Note: although this simply copies aggBufferAttributes, this common code can not be placed
  // in the superclass because that will lead to initialization ordering issues.
  override val inputAggBufferAttributes: Seq[AttributeReference] =
    aggBufferAttributes.map(_.newInstance())

  override def toString: String = {
    s"${udaf.name}(${children.mkString(",")})"
  }

  override def nodeName: String = udaf.name
}

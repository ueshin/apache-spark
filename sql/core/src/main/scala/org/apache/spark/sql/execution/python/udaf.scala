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
import org.apache.spark.internal.Logging
import org.apache.spark.sql.Column
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.catalyst.expressions.codegen.GenerateMutableProjection
import org.apache.spark.sql.catalyst.util.{ArrayData, ArrayDataBuffer}
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

case class BufferInputs(
    children: Seq[Expression],
    mutableAggBufferOffset: Int = 0,
    inputAggBufferOffset: Int = 0)
  extends ImperativeAggregate
  with NonSQLExpression
  with Logging {

  override def withNewMutableAggBufferOffset(newMutableAggBufferOffset: Int): ImperativeAggregate =
    copy(mutableAggBufferOffset = newMutableAggBufferOffset)

  override def withNewInputAggBufferOffset(newInputAggBufferOffset: Int): ImperativeAggregate =
    copy(inputAggBufferOffset = newInputAggBufferOffset)

  override def nullable: Boolean = true

  override def dataType: DataType = aggBufferSchema

  override val aggBufferSchema: StructType =
    StructType(children.zipWithIndex.map {
      case (child, i) =>
        StructField(s"_$i", ArrayType(child.dataType, child.nullable), nullable = false)
    })

  override val aggBufferAttributes: Seq[AttributeReference] = aggBufferSchema.toAttributes

  // Note: although this simply copies aggBufferAttributes, this common code can not be placed
  // in the superclass because that will lead to initialization ordering issues.
  override val inputAggBufferAttributes: Seq[AttributeReference] =
    aggBufferAttributes.map(_.newInstance())

  private[this] lazy val childrenSchema: StructType = {
    val inputFields = children.zipWithIndex.map {
      case (child, index) =>
        StructField(s"input$index", child.dataType, child.nullable, Metadata.empty)
    }
    StructType(inputFields)
  }

  private lazy val inputProjection = {
    val inputAttributes = childrenSchema.toAttributes
    log.debug(
      s"Creating MutableProj: $children, inputSchema: $inputAttributes.")
    GenerateMutableProjection.generate(children, inputAttributes)
  }

  override def initialize(buffer: InternalRow): Unit = {
    aggBufferSchema.zipWithIndex.foreach { case (_, i) =>
      buffer.update(i + mutableAggBufferOffset, new ArrayDataBuffer())
    }
  }

  override def update(buffer: InternalRow, input: InternalRow): Unit = {
    val projected = inputProjection(input)
    aggBufferSchema.zip(childrenSchema).zipWithIndex.foreach {
      case ((StructField(_, dt @ ArrayType(_, _), _, _), childSchema), i) =>
        val bufferOffset = i + mutableAggBufferOffset
        val arrayDataBuffer =
          buffer.get(bufferOffset, dt).asInstanceOf[ArrayDataBuffer]
        if (projected.isNullAt(i)) {
          arrayDataBuffer += null
        } else {
          arrayDataBuffer += InternalRow.copyValue(projected.get(i, childSchema.dataType))
        }
    }
  }

  override def merge(buffer1: InternalRow, buffer2: InternalRow): Unit = {
    aggBufferSchema.zipWithIndex.foreach {
      case (StructField(_, dt @ ArrayType(elementType, _), _, _), i) =>
        val bufferOffset = i + mutableAggBufferOffset
        val inputOffset = i + inputAggBufferOffset
        val arrayDataBuffer1 = buffer1.get(bufferOffset, dt).asInstanceOf[ArrayDataBuffer]
        buffer2.get(inputOffset, dt) match {
          case arrayDataBuffer2: UnsafeArrayData =>
            elementType match {
              case BooleanType => arrayDataBuffer1 ++= arrayDataBuffer2.toBooleanArray()
              case ByteType => arrayDataBuffer1 ++= arrayDataBuffer2.toByteArray()
              case ShortType => arrayDataBuffer1 ++= arrayDataBuffer2.toShortArray()
              case IntegerType => arrayDataBuffer1 ++= arrayDataBuffer2.toIntArray()
              case LongType => arrayDataBuffer1 ++= arrayDataBuffer2.toLongArray()
              case FloatType => arrayDataBuffer1 ++= arrayDataBuffer2.toFloatArray()
              case DoubleType => arrayDataBuffer1 ++= arrayDataBuffer2.toDoubleArray()
            }
          case arrayDataBuffer2: ArrayData =>
            arrayDataBuffer1 ++= arrayDataBuffer2
        }
    }
  }

  private val row = new GenericInternalRow(aggBufferSchema.size)

  override def eval(buffer: InternalRow): Any = {
    aggBufferSchema.zipWithIndex.foreach { case (buffSchema, i) =>
      val bufferOffset = i + mutableAggBufferOffset
      row.update(i, buffer.get(bufferOffset, buffSchema.dataType))
    }
    row
  }
}

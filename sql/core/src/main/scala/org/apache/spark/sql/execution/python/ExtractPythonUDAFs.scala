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

import scala.collection.mutable.{ArrayBuffer, Map}

import org.apache.spark.api.python.PythonEvalType
import org.apache.spark.sql.AnalysisException
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ProjectExec, SparkPlan}
import org.apache.spark.sql.execution.aggregate.{AggregateExec, AggUtils}
import org.apache.spark.sql.types.{ArrayType, StructType}

object ExtractPythonUDAFs extends Rule[SparkPlan] {

  private def isPythonUDAF(aggregateExpression: AggregateExpression): Boolean = {
    aggregateExpression.aggregateFunction.isInstanceOf[PythonUDAF]
  }

  private def hasPythonUDAF(aggregateExpressions: Seq[AggregateExpression]): Boolean = {
    aggregateExpressions.exists(isPythonUDAF)
  }

  private def hasDistinct(aggregateExpressions: Seq[AggregateExpression]): Boolean = {
    aggregateExpressions.exists(_.isDistinct)
  }

  override def apply(plan: SparkPlan): SparkPlan = plan transformUp {
    case agg: AggregateExec if !hasPythonUDAF(agg.aggregateExpressions) => agg
    case agg: AggregateExec if hasDistinct(agg.aggregateExpressions) =>
      throw new AnalysisException("Vectorized UDAF with distinct is not supported.")
    case agg: AggregateExec =>

      val newAggExprs = ArrayBuffer.empty[AggregateExpression] ++ agg.aggregateExpressions
      val newAggAttrs = ArrayBuffer.empty[Attribute] ++ agg.aggregateAttributes

      val buffers = ArrayBuffer.empty[BufferInputs]
      val udafs = ArrayBuffer.empty[PythonUDF]
      val udafResultAttrs = ArrayBuffer.empty[AttributeReference]

      val replacingReslutExprs = Map.empty[Expression, NamedExpression] ++
        agg.groupingExpressions.map(expr => expr -> expr.toAttribute)

      agg.aggregateExpressions.foreach {
        case aggExpr if isPythonUDAF(aggExpr) =>
          val pythonUDAF = aggExpr.aggregateFunction.asInstanceOf[PythonUDAF]

          aggExpr.mode match {
            case Partial =>
              val buffer = buffers.find { buf =>
                buf.children.length == pythonUDAF.children.length &&
                  buf.children.zip(pythonUDAF.children).forall { case (c, child) =>
                    c.semanticEquals(child)
                  }
              } match {
                case Some(buf) =>
                  newAggExprs -= aggExpr
                  newAggAttrs --= pythonUDAF.aggBufferAttributes

                  buf
                case None =>
                  val buf = BufferInputs(pythonUDAF.children)
                  buffers += buf

                  newAggExprs.update(
                    newAggExprs.indexOf(aggExpr), aggExpr.copy(aggregateFunction = buf))

                  val index = newAggAttrs.indexOfSlice(pythonUDAF.aggBufferAttributes)
                  newAggAttrs --= pythonUDAF.aggBufferAttributes
                  newAggAttrs.insertAll(index, buf.aggBufferAttributes)

                  buf
              }

              if (pythonUDAF.udaf.supportsPartial) {
                val udaf = PythonUDF(pythonUDAF.udaf.name, pythonUDAF.udaf.func,
                  pythonUDAF.udaf.returnType, buffer.aggBufferAttributes,
                  PythonEvalType.SQL_PANDAS_GROUP_AGGREGATE_UDF)
                udafs += udaf

                val (resultAttrs, replacingExprs) = pythonUDAF.inputAggBufferAttributes.map {
                  attr =>
                    val arrayType = attr.dataType.asInstanceOf[ArrayType]
                    val resultAttr = AttributeReference(
                      attr.name, arrayType.elementType, arrayType.containsNull)()
                    (resultAttr, attr -> Alias(CreateArray(Seq(resultAttr)), attr.name)())
                }.unzip
                udafResultAttrs ++= resultAttrs
                replacingReslutExprs ++= replacingExprs
              } else {
                replacingReslutExprs ++=
                  pythonUDAF.inputAggBufferAttributes.zip(
                    buffer.inputAggBufferAttributes.zip(buffer.aggBufferAttributes)).map {
                    case (attr, (newAttr, buffer)) =>
                      attr -> Alias(buffer, newAttr.name)(
                        newAttr.exprId, newAttr.qualifier, Option(newAttr.metadata))
                  }
              }

            case Final =>
              val buffer = BufferInputs(pythonUDAF.inputAggBufferAttributes.map { attr =>
                val arrayType = attr.dataType.asInstanceOf[ArrayType]
                AttributeReference(attr.name, arrayType.elementType, arrayType.containsNull)()
              })

              newAggExprs.update(
                newAggExprs.indexOf(aggExpr), aggExpr.copy(aggregateFunction = buffer))

              val bufferOut = AttributeReference("buffer", buffer.dataType, buffer.nullable)()
              newAggAttrs.update(newAggAttrs.indexOf(aggExpr.resultAttribute), bufferOut)

              val udafInputs = buffer.dataType.asInstanceOf[StructType].zipWithIndex.map {
                case (field, idx) =>
                  GetStructField(bufferOut, idx, Option(field.name))
              }
              val udaf = PythonUDF(pythonUDAF.udaf.name, pythonUDAF.udaf.func,
                pythonUDAF.udaf.returnType, udafInputs,
                PythonEvalType.SQL_PANDAS_GROUP_AGGREGATE_UDF)
              udafs += udaf

              val resultAttr = AttributeReference(udaf.name, udaf.dataType, udaf.nullable)()
              udafResultAttrs += resultAttr
              replacingReslutExprs += aggExpr.resultAttribute -> resultAttr

            case _ =>
              throw new AnalysisException(s"Unsupported aggregate mode: ${aggExpr.mode}.")
          }
        case aggExpr =>
          aggExpr.mode match {
            case Partial =>
              val af = aggExpr.aggregateFunction
              replacingReslutExprs ++=
                af.inputAggBufferAttributes.zip(af.aggBufferAttributes).map {
                  case (attr, buffer) =>
                    attr -> Alias(buffer, attr.name)(
                      attr.exprId, attr.qualifier, Option(attr.metadata))
                }
            case _ =>
          }
      }

      val newAgg = AggUtils.createAggregate(
        requiredChildDistributionExpressions = agg.requiredChildDistributionExpressions,
        groupingExpressions = agg.groupingExpressions,
        aggregateExpressions = newAggExprs,
        aggregateAttributes = newAggAttrs,
        initialInputBufferOffset = agg.initialInputBufferOffset,
        resultExpressions = agg.groupingExpressions ++ newAggAttrs,
        child = agg.child)

      val exec = if (udafs.size > 0) {
        ArrowEvalPythonExec(
          udafs,
          newAgg.output ++ udafResultAttrs,
          newAgg,
          PythonEvalType.SQL_PANDAS_GROUP_AGGREGATE_UDF,
          Some(1))
      } else {
        newAgg
      }

      val project = agg.resultExpressions.map { expr =>
        expr.transformUp {
          case expr if replacingReslutExprs.contains(expr) => replacingReslutExprs(expr)
        }.asInstanceOf[NamedExpression]
      }

      ProjectExec(project, exec)
  }
}

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

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ProjectExec, SparkPlan}
import org.apache.spark.sql.execution.aggregate.{HashAggregateExec, ObjectHashAggregateExec, SortAggregateExec}
import org.apache.spark.sql.types.ArrayType

object ExtractPythonUDAFs extends Rule[SparkPlan] with PredicateHelper {

  def isPythonUDAF(aggregateExpression: AggregateExpression): Boolean = {
    aggregateExpression.aggregateFunction.isInstanceOf[PythonUDAF]
  }

  def hasPythonUDAF(aggregateExpressions: Seq[AggregateExpression]): Boolean = {
    aggregateExpressions.exists(isPythonUDAF)
  }

  def hasPythonUDAF(plan: SparkPlan): Boolean = plan match {
    case agg: HashAggregateExec if hasPythonUDAF(agg.aggregateExpressions) => true
    case agg: ObjectHashAggregateExec if hasPythonUDAF(agg.aggregateExpressions) => true
    case agg: SortAggregateExec if hasPythonUDAF(agg.aggregateExpressions) => true
    case _ => false
  }

  def apply(plan: SparkPlan): SparkPlan = plan transformUp {
    case agg if hasPythonUDAF(agg) =>
      val groupingExpressions = getGroupingExpressions(agg)
      val groupingAttributes = groupingExpressions.map(_.toAttribute)

      val aggregateExpressions = getAggregateExpressions(agg)
      val aggregateAttributes = getAggregateAttributes(agg)
      val resultExpressions = getResultExpressions(agg)

      val newAggregateExpressions = ArrayBuffer.empty[AggregateExpression] ++ aggregateExpressions
      val newAggregateAttributes = ArrayBuffer.empty[Attribute] ++ aggregateAttributes

      val buffers = ArrayBuffer.empty[BufferInputs]
      val udafs = ArrayBuffer.empty[PythonUDF]
      val resultAttrs = ArrayBuffer.empty[AttributeReference]

      val replacingExprs = mutable.Map.empty[Expression, NamedExpression] ++
        groupingExpressions.zip(groupingAttributes)

      aggregateExpressions.foreach {
        case aggExpr if isPythonUDAF(aggExpr) =>
          val pythonUDAF = aggExpr.aggregateFunction.asInstanceOf[PythonUDAF]

          aggExpr.mode match {
            case Partial =>
              newAggregateExpressions -= aggExpr
              newAggregateAttributes --= pythonUDAF.aggBufferAttributes

              val buffer = buffers.find { buf =>
                buf.children.length == pythonUDAF.children.length &&
                  buf.children.zip(pythonUDAF.children).forall { case (c, child) =>
                    c.semanticEquals(child)
                  }
              }.getOrElse {
                val buf = BufferInputs(pythonUDAF.children)
                buffers += buf

                newAggregateExpressions += aggExpr.copy(aggregateFunction = buf)
                newAggregateAttributes ++= buf.aggBufferAttributes

                buf
              }

              val udaf = PythonUDF(pythonUDAF.udaf.name, pythonUDAF.udaf.partial,
                pythonUDAF.udaf.bufferType, buffer.aggBufferAttributes, vectorized = true)
              udafs += udaf
              val newAttrs = pythonUDAF.inputAggBufferAttributes.map { attr =>
                val elementType = attr.dataType.asInstanceOf[ArrayType].elementType
                val newAttr = AttributeReference(attr.name, elementType, attr.nullable)()
                newAttr -> Alias(CreateArray(Seq(newAttr)), attr.name)()
              }
              resultAttrs ++= newAttrs.map(_._1)
              replacingExprs ++= pythonUDAF.inputAggBufferAttributes.zip(newAttrs.map(_._2))

            case Final =>
              val buffer = BufferInputs(pythonUDAF.inputAggBufferAttributes.map { attr =>
                val elementType = attr.dataType.asInstanceOf[ArrayType].elementType
                AttributeReference(attr.name, elementType, attr.nullable)()
              })

              newAggregateExpressions.update(
                newAggregateExpressions.indexOf(aggExpr), aggExpr.copy(aggregateFunction = buffer))

              val bufferOut = AttributeReference("buffer", buffer.dataType, buffer.nullable)()
              newAggregateAttributes.update(
                 newAggregateAttributes.indexOf(aggExpr.resultAttribute), bufferOut)

              val udafInputs = buffer.inputAggBufferAttributes.zipWithIndex.map {
                case (attr, idx) =>
                  GetStructField(bufferOut, idx, Option(attr.name))
              }
              val udaf = PythonUDF(pythonUDAF.udaf.name, pythonUDAF.udaf.evaluate,
                pythonUDAF.udaf.returnType, udafInputs, vectorized = true)
              udafs += udaf
              val newAttr = AttributeReference(udaf.name, udaf.dataType, nullable = true)()
              resultAttrs += newAttr
              replacingExprs += aggExpr.resultAttribute -> newAttr

            case _ =>
          }
        case aggExpr =>
          aggExpr.mode match {
            case Partial =>
              val af = aggExpr.aggregateFunction
              replacingExprs ++= af.inputAggBufferAttributes.zip(af.aggBufferAttributes).map {
                case (attr, buffer) =>
                  attr -> Alias(buffer, attr.name)(
                    attr.exprId, attr.qualifier, Option(attr.metadata))
              }
            case _ =>
          }
      }

      val newAgg = copy(
        plan = agg,
        aggregateExpressions = newAggregateExpressions,
        aggregateAttributes = newAggregateAttributes,
        resultExpressions = groupingExpressions ++ newAggregateAttributes)

      val exec = BatchAggregateEvalPythonExec(udafs, newAgg.output ++ resultAttrs, newAgg)

      val project = resultExpressions.map {
        expr => expr.transformUp {
          case expr if replacingExprs.contains(expr) => replacingExprs(expr)
        }.asInstanceOf[NamedExpression]
      }

      ProjectExec(project, exec)
  }

  def getGroupingExpressions(plan: SparkPlan): Seq[NamedExpression] = plan match {
    case agg: HashAggregateExec => agg.groupingExpressions
    case agg: ObjectHashAggregateExec => agg.groupingExpressions
    case agg: SortAggregateExec => agg.groupingExpressions
  }

  def getAggregateExpressions(plan: SparkPlan): Seq[AggregateExpression] = plan match {
    case agg: HashAggregateExec => agg.aggregateExpressions
    case agg: ObjectHashAggregateExec => agg.aggregateExpressions
    case agg: SortAggregateExec => agg.aggregateExpressions
  }

  def getAggregateAttributes(plan: SparkPlan): Seq[Attribute] = plan match {
    case agg: HashAggregateExec => agg.aggregateAttributes
    case agg: ObjectHashAggregateExec => agg.aggregateAttributes
    case agg: SortAggregateExec => agg.aggregateAttributes
  }

  def getResultExpressions(plan: SparkPlan): Seq[NamedExpression] = plan match {
    case agg: HashAggregateExec => agg.resultExpressions
    case agg: ObjectHashAggregateExec => agg.resultExpressions
    case agg: SortAggregateExec => agg.resultExpressions
  }

  def copy(
      plan: SparkPlan,
      aggregateExpressions: Seq[AggregateExpression],
      aggregateAttributes: Seq[Attribute],
      resultExpressions: Seq[NamedExpression]): SparkPlan =
    plan match {
      case agg: HashAggregateExec =>
        agg.copy(
          aggregateExpressions = aggregateExpressions,
          aggregateAttributes = aggregateAttributes,
          resultExpressions = resultExpressions)
      case agg: ObjectHashAggregateExec =>
        agg.copy(
          aggregateExpressions = aggregateExpressions,
          aggregateAttributes = aggregateAttributes,
          resultExpressions = resultExpressions)
      case agg: SortAggregateExec =>
        agg.copy(
          aggregateExpressions = aggregateExpressions,
          aggregateAttributes = aggregateAttributes,
          resultExpressions = resultExpressions)
    }
}

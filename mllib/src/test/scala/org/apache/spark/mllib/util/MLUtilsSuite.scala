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

package org.apache.spark.mllib.util

import scala.util.Random

import org.scalatest.BeforeAndAfterAll
import org.scalatest.FunSuite
import org.scalatest.matchers.ShouldMatchers

import org.apache.spark.mllib.regression._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._

class MLUtilsSuite extends FunSuite with BeforeAndAfterAll with ShouldMatchers {
  @transient private var sc: SparkContext = _

  override def beforeAll() {
    sc = new SparkContext("local", "test")
  }

  override def afterAll() {
    sc.stop()
    System.clearProperty("spark.driver.port")
  }

  // This learner always says everything is 0
  def terribleLearner(trainingData: RDD[LabeledPoint]): RegressionModel = {
    object AlwaysZero extends RegressionModel {
      override def predict(testData: RDD[Array[Double]]): RDD[Double] = {
        testData.map(_ => 0)
      }
      override def predict(testData: Array[Double]): Double = {
        0
      }
    }
    AlwaysZero
  }

  // Always returns its input
  def exactLearner(trainingData: RDD[LabeledPoint]): RegressionModel = {
    new LinearRegressionModel(Array(1.0), 0)
  }

  test("Test cross validation with a terrible learner") {
    val data = sc.parallelize(1.to(100).zip(1.to(100))).map(
      x => LabeledPoint(x._1, Array(x._2)))
    val expectedError = 1.to(100).map(x => x*x).sum / 100.0
    for (seed <- 1 to 5) {
      for (folds <- 2 to 5) {
        val avgError = MLUtils.crossValidate(data, folds, seed, terribleLearner)
        avgError should equal (expectedError)
      }
    }
  }
  test("Test cross validation with a reasonable learner") {
    val data = sc.parallelize(1.to(100).zip(1.to(100))).map(
      x => LabeledPoint(x._1, Array(x._2)))
    for (seed <- 1 to 5) {
      for (folds <- 2 to 5) {
        val avgError = MLUtils.crossValidate(data, folds, seed, exactLearner)
        avgError should equal (0)
      }
    }
  }

  test("Cross validation requires more than one fold") {
    val data = sc.parallelize(1.to(100).zip(1.to(100))).map(
      x => LabeledPoint(x._1, Array(x._2)))
    val thrown = intercept[java.lang.IllegalArgumentException] {
      val avgError = MLUtils.crossValidate(data, 1, 1, terribleLearner)
    }
    assert(thrown.getClass === classOf[IllegalArgumentException])
  }
}

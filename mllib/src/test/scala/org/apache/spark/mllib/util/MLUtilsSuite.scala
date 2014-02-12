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

  // Always returns its input
  def exactLearner(trainingData: RDD[LabeledPoint]): RegressionModel = {
    new LinearRegressionModel(Array(1.0), 0)
  }

  test("kfoldRdd") {
    val data = sc.parallelize(1 to 100, 2)
    val collectedData = data.collect().sorted
    val twoFoldedRdd = MLUtils.kFoldRdds(data, 2, 1)
    assert(twoFoldedRdd(0)._1.collect().sorted === twoFoldedRdd(1)._2.collect().sorted)
    assert(twoFoldedRdd(0)._2.collect().sorted === twoFoldedRdd(1)._1.collect().sorted)
    for (folds <- 2 to 10) {
      for (seed <- 1 to 5) {
        val foldedRdds = MLUtils.kFoldRdds(data, folds, seed)
        assert(foldedRdds.size === folds)
        foldedRdds.map{case (test, train) =>
          val result = test.union(train).collect().sorted
          assert(test.collect().size > 0, "Non empty test data")
          assert(train.collect().size > 0, "Non empty training data")
          assert(result ===  collectedData,
            "Each training+test set combined contains all of the data")
        }
        // K fold cross validation should only have each element in the test set exactly once
        assert(foldedRdds.map(_._1).reduce((x,y) => x.union(y)).collect().sorted ===
          data.collect().sorted)
      }
    }
  }

}

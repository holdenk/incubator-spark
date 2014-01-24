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

package org.apache.spark.examples

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import scala.math.exp
import org.apache.spark.util.Vector
import org.apache.spark._


object SparkSVM {
  def main(args: Array[String]) {
    if (args.length == 0) {
      System.err.println("Usage: SparkSVM <master>")
      System.exit(1)
    }
    val sc = new SparkContext(args(0), "SparkSVM",
      System.getenv("SPARK_HOME"), SparkContext.jarOfClass(this.getClass))
    val data = sc.textFile("/home/holden/a")

    val trainingData = data.map { line =>
      val parts = line.split(' ')
      LabeledPoint(parts(0).toDouble, parts.tail.map(x => x.toDouble).toArray)
    }

    // Run training algorithm
    val numIterations = 20
    val model = SVMWithSGD.train(trainingData, numIterations)

    
    // Evaluate model on training examples and compute training error
    val labelAndPreds = trainingData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / trainingData.count
    println("trainError = " + trainErr)
  }
}

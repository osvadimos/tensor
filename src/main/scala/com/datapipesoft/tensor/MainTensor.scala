package com.datapipesoft.tensor

import java.nio.file.Paths

import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api.tf
import org.slf4j.LoggerFactory


object MainTensor {

  private[this] val logger = Logger(LoggerFactory.getLogger("Examples / MNIST"))

  def main(args: Array[String]): Unit = {
    println("Test")
    val dataSet = MNISTLoader.load(Paths.get("datasets/MNIST"))
    val trainImages = tf.data.TensorSlicesDataset(dataSet.trainImages)
    val trainLabels = tf.data.TensorSlicesDataset(dataSet.trainLabels)
    val testImages = tf.data.TensorSlicesDataset(dataSet.testImages)
    val testLabels = tf.data.TensorSlicesDataset(dataSet.testLabels)
    val trainData =
      trainImages.zip(trainLabels)
        .repeat()
        .shuffle(10000)
        .batch(256)
        .prefetch(10)
    val evalTrainData = trainImages.zip(trainLabels).batch(1000).prefetch(10)
    val evalTestData = testImages.zip(testLabels).batch(1000).prefetch(10)
    logger.info("Building the logistic regression model.")
  }

}

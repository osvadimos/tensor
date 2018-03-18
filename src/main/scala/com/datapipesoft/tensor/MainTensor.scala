package com.datapipesoft.tensor

import java.nio.file.Paths


object MainTensor {

  def main(args: Array[String]): Unit = {
    println("Test")
    val dataSet = MNISTLoader.load(Paths.get("datasets/MNIST"))
    println(dataSet)
  }

}

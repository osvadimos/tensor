package com.datapipesoft.tensor

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.INT32

object MainTensor {

  def main(args: Array[String]): Unit = {
    println("Test")
    val tensor = Tensor.zeros(INT32, Shape(2, 5))
    println(tensor.toString())
  }

}

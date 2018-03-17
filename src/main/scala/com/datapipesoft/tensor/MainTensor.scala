package com.datapipesoft.tensor

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.FLOAT32
import org.platanios.tensorflow.api.types.INT32

object MainTensor {

  def main(args: Array[String]): Unit = {
    println("Test")
    val t0 = Tensor.ones(INT32, Shape())     // Creates a scalar equal to the value 1
    val t1 = Tensor.ones(INT32, Shape(10))   // Creates a vector with 10 elements, all of which are equal to 1
    val t2 = Tensor.ones(INT32, Shape(5, 2)) // Creates a matrix with 5 rows with 2 columns

    // You can also create tensors in the following way:
    val t3 = Tensor(2.0, 5.6)                                 // Creates a vector that contains the numbers 2.0 and 5.6
    val t4 = Tensor(Tensor(1.2f, -8.4f), Tensor(-2.3f, 0.4f)) // Creates a matrix with 2 rows and 2 columns
    println("end test")
  }

}

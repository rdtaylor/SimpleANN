package com.korpisystems.SimpleANN.Exceptions


case class UnevenTrainingInstanceLists(inputs: Int, expected_outputs: Int)
  extends Exception {

    val msg: String = 
      if (inputs > expected_outputs) {
        "There are more inputs (" + inputs + ") than expected outputs (" +
        expected_outputs + ")."
      }
      else {
        "There are more expected outputs (" + expected_outputs +
        ") than inputs (" + inputs + ")."
      }

    new Exception(msg)
}

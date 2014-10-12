package com.korpisystems.SimpleANN

import org.scalatest.FunSuite


class ANNTest extends FunSuite {
  test("ANN learns non-linear XOR properly") {
    val inputs: ExpectedValues = List(
      List(1, 1)
    , List(0, 1)
    , List(1, 0)
    , List(0, 0)
    )

    val expected_out: ExpectedValues = List(
      List(0)
    , List(1)
    , List(1)
    , List(0)
    )

    val ann = new ANN(List(2, 4, 1), 1.0)
    ann.train(inputs, expected_out, iter=5000)

    val xor_1_1 = ann.getOutput(inputs(0))(0)
    assert(xor_1_1 < 0.04)

    val xor_0_1 = ann.getOutput(inputs(1))(0)
    assert(xor_0_1 > 0.96)

    val xor_1_0 = ann.getOutput(inputs(2))(0)
    assert(xor_1_0 > 0.96)

    val xor_0_0 = ann.getOutput(inputs(3))(0)
    assert(xor_0_0 < 0.04)
  }

  test("ANN learns XOR with multiple hidden layers") {
    val inputs: ExpectedValues = List(
      List(1, 1)
    , List(0, 1)
    , List(1, 0)
    , List(0, 0)
    )

    val expected_out: ExpectedValues = List(
      List(0)
    , List(1)
    , List(1)
    , List(0)
    )

    val ann = new ANN(List(2, 4, 3, 1), 1.0)
    ann.train(inputs, expected_out, iter=5000)

    val xor_1_1 = ann.getOutput(inputs(0))(0)
    assert(xor_1_1 < 0.04)

    val xor_0_1 = ann.getOutput(inputs(1))(0)
    assert(xor_0_1 > 0.96)

    val xor_1_0 = ann.getOutput(inputs(2))(0)
    assert(xor_1_0 > 0.96)

    val xor_0_0 = ann.getOutput(inputs(3))(0)
    assert(xor_0_0 < 0.04)
  }

  test("ANN learns first input is output") {
    val inputs: ExpectedValues = List(
      List(1, 1)
    , List(0, 1)
    , List(1, 0)
    , List(0, 0)
    )

    val expected_out: ExpectedValues = List(
      List(1)
    , List(0)
    , List(1)
    , List(0)
    )

    val ann = new ANN(List(2, 4, 1), 1.0)
    ann.train(inputs, expected_out, iter=5000)

    val first_1_1 = ann.getOutput(inputs(0))(0)
    assert(first_1_1 > 0.96)

    val first_0_1 = ann.getOutput(inputs(1))(0)
    assert(first_0_1 < 0.04)

    val first_1_0 = ann.getOutput(inputs(2))(0)
    assert(first_1_0 > 0.96)

    val first_0_0 = ann.getOutput(inputs(3))(0)
    assert(first_0_0 < 0.04)
  }

  test("ANN learns second input is output") {
    val inputs: ExpectedValues = List(
      List(1, 1)
    , List(0, 1)
    , List(1, 0)
    , List(0, 0)
    )

    val expected_out: ExpectedValues = List(
      List(1)
    , List(1)
    , List(0)
    , List(0)
    )

    val ann = new ANN(List(2, 4, 1), 1.0)
    ann.train(inputs, expected_out, iter=5000)

    val first_1_1 = ann.getOutput(inputs(0))(0)
    assert(first_1_1 > 0.96)

    val first_0_1 = ann.getOutput(inputs(1))(0)
    assert(first_0_1 > 0.96)

    val first_1_0 = ann.getOutput(inputs(2))(0)
    assert(first_1_0 < 0.04)

    val first_0_0 = ann.getOutput(inputs(3))(0)
    assert(first_0_0 < 0.04)
  }

  test("UnevenTrainingInstanceLists thrown properly") {
    val inputs: ExpectedValues = List(
      List(1, 1)
    , List(0, 1)
    , List(1, 0)
    , List(0, 0)
    )

    val expected_out: ExpectedValues = List(
      List(1)
    , List(0)
    , List(1)
    )

    val ann = new ANN(List(2, 4, 1), 1.0)
    intercept[Exceptions.UnevenTrainingInstanceLists] {
      ann.train(inputs, expected_out, iter=5000)
    }
  }
}

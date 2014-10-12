package com.korpisystems.SimpleANN

import Math._


abstract class Activator {
  def activation(x: NodeSignal): NodeSignal
  def derivative(x: NodeSignal): NodeSignal
}

object sigmoid extends Activator {
  def activation(x: NodeSignal): NodeSignal = {
    1 / (1 + pow(E, -x))
  }

  def derivative(x: NodeSignal): NodeSignal = {
    x * (1 - x)
  }
}

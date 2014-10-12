package com.korpisystems.SimpleANN


class ANN(layout: List[Int], eta: Double, activator: Activator) {

  def this(layout: List[Int]) = this(layout, 0.25, sigmoid)
  def this(layout: List[Int], eta: Double) = this(layout, eta, sigmoid)

  private type NodeValues[N] = List[List[N]]
  private type Weights = NodeValues[List[Double]]
  private type Nodes = NodeValues[NodeSignal]
  private type LayerOutput = List[NodeSignal]

  private val n_layers: Int = layout.length - 1
  private var weights = getInitWeights()
  private var biases = getInitBiases()

  def train(inputs: ExpectedValues
           , expected_out: ExpectedValues
           , iter: Int = 5000
           ): Unit = {

    if(inputs.length != expected_out.length) {
      throw new Exceptions.UnevenTrainingInstanceLists(inputs.length
                                                      , expected_out.length)
    }

    def trainInstance(input: LayerOutput
                     , expected_out: LayerOutput
                     ): Unit = {

      val output = calcOutput(input, weights, biases)
      val errors = getErrors(expected_out, output, weights)

      weights = updateWeights(errors, weights, output)
      biases = updateBiases(errors, biases)
    }

    (1 to iter).foreach(iteration => {
      for (index <- 0 until inputs.length) {
        trainInstance(inputs(index), expected_out(index)) 
      }
    })
  }

  def getOutput(input: LayerOutput): LayerOutput = {
    val full_output: Nodes = calcOutput(input, weights, biases)
    full_output.last
  }

  private def calcOutput(input: LayerOutput
                        , weights: Weights
                        , biases: Nodes
                        ): Nodes = {

    def getLayerOutput(layer: Int
                      , prev_layer_output: LayerOutput
                      ): LayerOutput =
      (for (layer_node <- 0 until layout(layer)) yield {

        val node_inputs: List[NodeSignal] =
          (for (connection <- 0 until layout(layer - 1))
            yield weights(layer - 1)(layer_node)(connection) *
            prev_layer_output(connection)).toList

        activator.activation(node_inputs.fold(0.0)(_ + _) +
          biases(layer - 1)(layer_node))
      }).toList

    def iterateLayers(fn: (Int, LayerOutput) => LayerOutput
                     , prev_layer_output: LayerOutput
                     , layers: List[Int]
                     ): Nodes =
      layers match {
        case (layer :: Nil) => List(fn(layer, prev_layer_output))
        case (layer :: next_layers) => {
          val output: LayerOutput = fn(layer, prev_layer_output)
          List(output) ++ iterateLayers(fn, output, next_layers)
        }
      }

    val layers: List[Int] = (1 to n_layers).toList
    List(input) ++ iterateLayers(getLayerOutput, input, layers)
  }

  private def getErrors(expected_out: LayerOutput
                       , output: Nodes
                       , weights: Weights
                       ): Nodes = {

    val last_layer_error = (for (layer_node <- 0 until layout(n_layers))
      yield {
        activator.derivative(output(n_layers)(layer_node)) *
        (expected_out(layer_node) - output(n_layers)(layer_node))
      }).toList

    def getLayerError(layer: Int, prev_output: LayerOutput): LayerOutput = {
      (for (node <- 0 until layout(layer)) yield {

        val err_weight_sum =
          (for (connection <- 0 until layout(layer + 1)) yield {
            prev_output(connection) * weights(layer)(connection)(node)
        }).fold(0.0)(_ + _)
        
        err_weight_sum * activator.derivative(output(layer)(node))

      }).toList
    }

    def iterate(fn: ((Int, LayerOutput) => LayerOutput),
      prev_output: LayerOutput, layers: List[Int]): Nodes = layers match {
        case Nil => List(prev_output)
        case (x :: xs) =>  {
          val new_output = fn(x, prev_output)
          List(prev_output) ++ iterate(getLayerError, new_output, xs)
        }
    }

    val layers = ((n_layers - 1) to 1 by -1).toList
    val reversed_errors: Nodes = iterate(getLayerError
                                        , last_layer_error
                                        , layers)
    reversed_errors.reverse
  }

  private def updateWeights(errors: Nodes
                           , weights: Weights
                           , input: Nodes
                           ) = {

    val reversed_weights = (for (layer <- (n_layers to 1 by -1)) yield {
      (for (node <- 0 until layout(layer)) yield {
        (for (connection <- 0 until layout(layer - 1)) yield {

          val old_weight =  weights(layer - 1)(node)(connection)
          val delta_weight = eta * errors(layer - 1)(node) *
            input(layer - 1)(connection)

          old_weight + delta_weight

        }).toList
      }).toList
    }).toList

    reversed_weights.reverse
  }
  
  private def updateBiases(errors: Nodes, biases: Nodes): Nodes = {

    (for (layer <- 1 to n_layers) yield {
      (for (node <- 0 until layout(layer)) yield {

        val old_bias =  biases(layer - 1)(node)
        val delta_bias = eta * 1 * errors(layer - 1)(node)

        old_bias + delta_bias

      }).toList
    }).toList
  }

  private def getInitBiases(random: Boolean = true): Nodes = {
    (for (layers <- 1 to n_layers; layer_nodes <- 0 until layout(layers))
      yield {
        (for (layer_nodes <- 0 until layout(layers)) yield
          getInitValue(random)).toList
      }).toList
  }

  private def getInitWeights(random: Boolean = true): Weights = {
    (for (layers <- 1 to n_layers) yield {
      (for (layer_nodes <- 0 until layout(layers)) yield {
        (for (connections <- 0 until layout(layers - 1)) yield
          getInitValue(random, 1.0)).toList
      }).toList
    }).toList
  }

  private def getInitValue(random: Boolean = true
                          , const: NodeSignal = 0.0
                          ): NodeSignal = {
    if(random) {
      2 * (util.Random.nextDouble - 0.5)
    }
    else {
      const
    }
  }

}

Simple Artificial Neural Network
================================

This package is an implementation of a multilayer perceptron trained through backpropagation.


Usage
=====

## Instantiate an ANN object:

```
val ann = new ANN(layout, eta, activator)
```

e.g.:

```
val ann = new ANN(List(2, 4, 1), 1.0, sigmoid)
```

The first argument, layout, provides the number of nodes and layers in the network.  For example, the above list of integers -- List(2, 4, 1) -- means a network consisting of three layers with two nodes in the input layer, four nodes in the hidden layer, and one node in the output layer.  The number of nodes in the input layer must match the number of intended inputs while the same is true of the output layer.

The second argument, eta, is the rate of learning (with a default of 0.25).  The higher the eta, the faster the network will learn, but the less precise it will be.  The lower the eta, the more precise its responses, but the slower the learning rate.  This setting may need to be adjusted for different tasks.

The third argument, activator, is the activation function to be used by the network.  The default is the included sigmoid function included in Activation.scala.

## Train the network:

```
ann.train(inputs, expected_out, iter=5000)
```

e.g. to train binary "and" on the first and second inputs as well as a binary "and" on the third and fourth inputs:

```
  val inputs: ExpectedValues = List(
    List(1, 1, 1, 1)
    , List(1, 1, 0, 0)
    , List(0, 0, 1, 1)
    , List(1, 0, 1, 0)
    , List(0, 1, 0, 1)
    , List(0, 0, 0, 0)
    )
  val expected_out = List(
    List(1, 1)
    , List(1, 0)
    , List(0, 1)
    , List(0, 0)
    , List(0, 0)
    , List(0, 0)
    )
  ann.train(inputs, expected_out)
```

The first argument, inputs, contains a list of lists with desired inputs for the network to learn.  The number of nodes in the input layer must equal the number of inputs in each nested list.

The second argument, expected_out, corresponds to the expected output from the network for each of the previous inputs.  The first set of inputs (i.e. inputs(0)) should give the first expected output (i.e. expected_out(0)).  The number of outputs in each nested list must equal the number of output nodes in the network.

The last argument, iter, is the number of iterations of training to run.  The default of 5000 ought to be altered if the network reaches its equilibrium more quickly or slowly.

## Predict a value:

```
ann.getOutput(input)
```

e.g.:

```
  val outputs = ann.getOutput(List(1, 0))
```

(See tests (src/test) for further examples).


Compilation:
============

In sbt, type "assembly" to create the .jar file.  It will be placed in target/scala-${SCALA_VERSION}/SimpleANN.jar.  Then, include the .jar in your project for use.

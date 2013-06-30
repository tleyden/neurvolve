package neurvolve

import (
	ng "github.com/tleyden/neurgo"
)

func AddNeuron(neuralNet *ng.NeuralNetwork) *ng.NeuralNetwork {

	// find out how many layers if hidden nodes the network has
	numLayers := neuralNet.NumLayers()

	// choose a random layer L
	targetLayerIndex := randomLayer(numLayers)

	// calculate previous and next layer indexes
	prevLayerIndex := targetLayerIndex - 1
	nextLayerIndex := targetLayerIndex + 1

	// choose a random node (sensor, neuron) in the previous
	// layer (L-1), call this N0
	prevLayerNodes := neuralNet.NodesInLayer(prevLayerIndex)
	prevLayerNode := randomNodeFrom(prevLayerNodes)

	// choose a random node (neuron, actutor) in the next
	// layer (L+1), call this N1

	// create new Neuron N,

	// create random Bias and weights for neuron

	// make connection from N0 -> N and from N -> N1  (if N1 == actuator, unweighted cxn)

	return neuralNet

}

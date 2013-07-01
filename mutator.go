package neurvolve

import (
	ng "github.com/tleyden/neurgo"
	"log"
	"math"
)

// Add a new neuron in a whole new layer, which must be to the _right_ of
// an existing layer and therefore cannot form a new input layer.  (this
// circumvents the minor issue that we don't have a way of knowing the
// size of the input vectors coming from the sensors)
func AddNeuronNewRightLayer(neuralNet *ng.NeuralNetwork) *ng.NeuralNetwork {

	neuralNet = neuralNet.Copy()

	// find out how many layers this network has (includes sensor/actuator layers)
	numLayers := neuralNet.NumLayers()

	leftLayerIndex := randomNeuronLayerIndex(numLayers)

	rightLayerIndex := leftLayerIndex + 1
	rightLayerNodes := neuralNet.NodesInLayer(rightLayerIndex)

	// create random Bias
	randBias := ng.RandomInRange(-1*math.Pi, math.Pi)

	// create new Neuron N,
	processor := &ng.Neuron{Bias: randBias, ActivationFunction: ng.Sigmoid}
	neuron := &ng.Node{Name: "added_neuron"}
	neuron.SetProcessor(processor)

	// disconnect all neurons in the left layer from nodes they are
	// currently connected to, and connect them to the new neuron
	leftLayerNodes := neuralNet.NodesInLayer(leftLayerIndex)
	for _, leftLayerNode := range leftLayerNodes {
		leftLayerNode.DisconnectAllBidirectional()
	}

	for _, leftLayerNode := range leftLayerNodes {
		// this weight vector has only one weight, which is true for all
		// neurons except input layer neurons which may get larger input
		// vectors from sensors.
		weights := randomWeights(1)
		leftLayerNode.ConnectBidirectionalWeighted(neuron, weights)
	}

	// connect newNeuron to all nodes in rightLayer
	for _, rightLayerNode := range rightLayerNodes {
		if rightLayerNode.IsNeuron() {
			weights := randomWeights(1)
			neuron.ConnectBidirectionalWeighted(rightLayerNode, weights)
		} else {
			neuron.ConnectBidirectional(rightLayerNode)
		}
	}

	return neuralNet
}

// Add a new neuron to an existing layer.  Connect all nodes in previous
// layer to this node.  Connect this node to all nodes in next layer.
func AddNeuronExistingLayer(neuralNet *ng.NeuralNetwork) *ng.NeuralNetwork {

	neuralNet = neuralNet.Copy()

	// find out how many layers this network has (includes sensor/actuator layers)
	numLayers := neuralNet.NumLayers()

	targetLayerIndex := randomNeuronLayerIndex(numLayers)

	// calculate previous and next layer indexes
	prevLayerIndex := targetLayerIndex - 1
	nextLayerIndex := targetLayerIndex + 1

	prevLayerNodes := neuralNet.NodesInLayer(prevLayerIndex)

	nextLayerNodes := neuralNet.NodesInLayer(nextLayerIndex)

	// create random Bias
	randBias := ng.RandomInRange(-1*math.Pi, math.Pi)

	// create new Neuron N,
	processor := &ng.Neuron{Bias: randBias, ActivationFunction: ng.Sigmoid}
	neuron := &ng.Node{Name: "added_neuron"}
	neuron.SetProcessor(processor)

	// find out how big the weight vector should be by looking at
	// another node in the same layer
	targetLayerNodes := neuralNet.NodesInLayer(targetLayerIndex)
	targetLayerNode := randomNodeFrom(targetLayerNodes)
	inboundWeightVectorLen := discoverWeightVectorLen(targetLayerNode)

	// create random weights for neuron
	// TODO: how many??
	inboundWeights := randomWeights(inboundWeightVectorLen)

	// make connection from prev layer nodes to this node
	for _, prevLayerNode := range prevLayerNodes {
		prevLayerNode.ConnectBidirectionalWeighted(neuron, inboundWeights)
	}

	// make connection from this node to next layer nodes  (if N1 == actuator, unweighted cxn)
	randomNextLayerNode := randomNodeFrom(nextLayerNodes)
	if randomNextLayerNode.IsNeuron() {
		for _, nextLayerNode := range nextLayerNodes {
			weightVectorLen := discoverWeightVectorLen(nextLayerNode)
			weights := randomWeights(weightVectorLen)
			neuron.ConnectBidirectionalWeighted(nextLayerNode, weights)
		}

	} else {
		for _, nextLayerNode := range nextLayerNodes {
			neuron.ConnectBidirectional(nextLayerNode)
		}
	}

	return neuralNet

}

// choose a random neural layer based on num total layers.  note that the first and last
// layer should be ignored, since they are not neural layers (have sensors and actuators)
func randomNeuronLayerIndex(numLayers int) int {
	if numLayers < 3 {
		log.Panicf("Expecting at least 3 layers, got: %d", numLayers)
	}
	startIndex := 1           // ignore sensor layer
	endIndex := numLayers - 1 // ignore actuator layer
	return ng.RandomIntInRange(startIndex, endIndex)
}

func randomNodeFrom(nodes []*ng.Node) *ng.Node {
	if len(nodes) == 0 {
		log.Panicf("Expecting non-empty list of nodes")
	}
	randomIndex := ng.RandomIntInRange(0, len(nodes))
	return nodes[randomIndex]
}

func discoverWeightVectorLen(node *ng.Node) int {
	if len(node.Inbound()) == 0 {
		log.Panicf("Expecting node with non-empty inbound connections")
	}
	connection := node.Inbound()[0]
	return len(connection.Weights())
}

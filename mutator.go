package neurvolve

import (
	ng "github.com/tleyden/neurgo"
	"log"
)

func inboundConnectionCandidates(neuron *ng.Neuron) []*ng.NodeId {

	cortex := neuron.Cortex
	neuronNodeIds := cortex.NeuronNodeIds()
	sensorNodeIds := cortex.SensorNodeIds()
	availableNodeIds := append(neuronNodeIds, sensorNodeIds...)

	// hackish way to delete a vew elements from this slice.
	// put in a map and delete from map, then back to slice. TODO: fixme
	availableNodeIdMap := make(map[string]*ng.NodeId)
	for _, nodeId := range availableNodeIds {
		availableNodeIdMap[nodeId.UUID] = nodeId
	}
	for _, inboundConnection := range neuron.Inbound {
		nodeId := inboundConnection.NodeId
		delete(availableNodeIdMap, nodeId.UUID)
	}
	availableNodeIds = make([]*ng.NodeId, 0)
	for _, nodeId := range availableNodeIdMap {
		availableNodeIds = append(availableNodeIds, nodeId)
	}
	return availableNodeIds

}

func NeuronAddInlinkNonRecurrent(neuron *ng.Neuron) *ng.InboundConnection {
	availableNodeIds := inboundConnectionCandidates(neuron)

	// remove any node id's which have a layer index >= neuron.LayerIndex
	nonRecurrentNodeIds := make([]*ng.NodeId, 0)
	for _, nodeId := range availableNodeIds {
		if nodeId.LayerIndex < neuron.NodeId.LayerIndex {
			nonRecurrentNodeIds = append(nonRecurrentNodeIds, nodeId)
		}

	}

	return neuronAddInlink(neuron, nonRecurrentNodeIds)
}

func NeuronAddInlinkRecurrent(neuron *ng.Neuron) *ng.InboundConnection {

	// choose a random element B, where element B is another
	// neuron or a sensor which is not already connected
	// to this neuron.
	availableNodeIds := inboundConnectionCandidates(neuron)
	return neuronAddInlink(neuron, availableNodeIds)
}

func neuronAddInlink(neuron *ng.Neuron, availableNodeIds []*ng.NodeId) *ng.InboundConnection {

	cortex := neuron.Cortex

	if len(availableNodeIds) == 0 {
		log.Printf("Warning: unable to add inlink to neuron: %v", neuron)
		return nil
	}

	randIndex := ng.RandomIntInRange(0, len(availableNodeIds))
	chosenNodeId := availableNodeIds[randIndex]

	// create weight vector
	weightVectorLength := 1
	if chosenNodeId.NodeType == ng.SENSOR {
		sensor := cortex.FindSensor(chosenNodeId)
		weightVectorLength = sensor.VectorLength
	}
	weights := randomWeights(weightVectorLength)

	// now make a connection
	connection := neuron.ConnectInboundWeighted(chosenNodeId, weights)

	return connection

}

func NeuronMutateWeights(neuron *ng.Neuron) bool {
	didPerturbAnyWeights := false
	probability := parameterPerturbProbability(neuron)
	for _, cxn := range neuron.Inbound {
		didPerturbWeight := possiblyPerturbConnection(cxn, probability)
		if didPerturbWeight == true {
			didPerturbAnyWeights = true
		}
	}
	return didPerturbAnyWeights
}

func NeuronMutateActivation(neuron *ng.Neuron) {

	encodableActivations := ng.AllEncodableActivations()

	for i := 0; i < 100; i++ {

		// pick a random activation function from list
		randomIndex := ng.RandomIntInRange(0, len(encodableActivations))

		chosenActivation := encodableActivations[randomIndex]

		// if we chose a different activation than current one, use it
		if chosenActivation.Name != neuron.ActivationFunction.Name {
			neuron.ActivationFunction = chosenActivation
			return
		}
	}

	// if we got this far, something went wrong
	log.Panicf("Unable to mutate activation function for neuron: %v", neuron)

}

func NeuronResetWeights(neuron *ng.Neuron) {
	for _, cxn := range neuron.Inbound {
		for j, _ := range cxn.Weights {
			cxn.Weights[j] = RandomWeight()
		}
	}
}

func NeuronAddBias(neuron *ng.Neuron) {
	if neuron.Bias == 0 {
		neuron.Bias = RandomBias()
	}
}

func NeuronRemoveBias(neuron *ng.Neuron) {
	if neuron.Bias != 0 {
		neuron.Bias = 0
	}
}

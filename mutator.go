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

	// remove things we already have inbound connections from
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

func AddNeuronNonRecurrent(cortex *ng.Cortex) *ng.Neuron {
	layerMap := cortex.NeuronLayerMap()
	randomLayer := layerMap.ChooseRandomLayer()
	neuron := cortex.CreateNeuronInLayer(randomLayer)
	upstreamNeuron := layerMap.ChooseNeuronPrecedingLayer(randomLayer)
	if upstreamNeuron == nil {
		log.Printf("Unable to find upstream neuron, cannot add neuron")
		return nil
	}

	downstreamNeuron := layerMap.ChooseNeuronFollowingLayer(randomLayer)
	if downstreamNeuron == nil {
		log.Printf("Unable to find downstream neuron, cannot add neuron")
		return nil
	}

	neuronAddInlinkFrom(neuron, upstreamNeuron.NodeId)
	neuronAddOutlinkTo(neuron, downstreamNeuron.NodeId)

	return neuron
}

func AddNeuronRecurrent(cortex *ng.Cortex) *ng.Neuron {
	return nil
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
	return neuronAddInlinkFrom(neuron, chosenNodeId)

}

func neuronAddInlinkFrom(neuron *ng.Neuron, sourceNodeId *NodeId) *ng.InboundConnection {

	cortex := neuron.Cortex

	// create weight vector
	weightVectorLength := 1
	if sourceNodeId.NodeType == ng.SENSOR {
		sensor := cortex.FindSensor(sourceNodeId)
		weightVectorLength = sensor.VectorLength
	}
	weights := randomWeights(weightVectorLength)

	// make an inbound connection sourceNodeId <- neuron
	connection := neuron.ConnectInboundWeighted(sourceNodeId, weights)

	// make an outbound connection sourceNodeId -> neuron
	chosenConnector := cortex.FindConnector(sourceNodeId)
	ng.ConnectOutbound(chosenConnector, neuron)

	return connection

}

func outboundConnectionCandidates(neuron *ng.Neuron) []*ng.NodeId {

	cortex := neuron.Cortex
	neuronNodeIds := cortex.NeuronNodeIds()
	actuatorNodeIds := cortex.ActuatorNodeIds()
	availableNodeIds := append(neuronNodeIds, actuatorNodeIds...)

	// hackish way to delete a vew elements from this slice.
	// put in a map and delete from map, then back to slice. TODO: fixme
	availableNodeIdMap := make(map[string]*ng.NodeId)
	for _, nodeId := range availableNodeIds {
		availableNodeIdMap[nodeId.UUID] = nodeId
	}

	// remove things we are already connected to
	for _, outboundConnection := range neuron.Outbound {
		nodeId := outboundConnection.NodeId
		delete(availableNodeIdMap, nodeId.UUID)
	}

	// remove actuators that can't support any more inbound connections
	for _, actuatorNodeId := range actuatorNodeIds {
		actuator := cortex.FindActuator(actuatorNodeId)
		// does the actuator have capacity for another
		// incoming connection?
		if actuator.CanAddInboundConnection() == false {
			delete(availableNodeIdMap, actuatorNodeId.UUID)
		}
	}

	availableNodeIds = make([]*ng.NodeId, 0)
	for _, nodeId := range availableNodeIdMap {
		availableNodeIds = append(availableNodeIds, nodeId)
	}
	return availableNodeIds

}

func neuronAddOutlink(neuron *ng.Neuron, availableNodeIds []*ng.NodeId) *ng.OutboundConnection {

	cortex := neuron.Cortex

	if len(availableNodeIds) == 0 {
		log.Printf("Warning: unable to add inlink to neuron: %v", neuron)
		return nil
	}

	randIndex := ng.RandomIntInRange(0, len(availableNodeIds))
	chosenNodeId := availableNodeIds[randIndex]

	return neuronAddOutlinkTo(neuron, chosenNodeId)

}

func neuronAddOutlinkTo(neuron *ng.Neuron, targetNodeId ng.NodeId) *ng.OutboundConnection {

	switch targetNodeId.NodeType {
	case ng.NEURON:

		// make an outbound connection neuron -> targetNodeId
		chosenNeuron := cortex.FindNeuron(targetNodeId)
		connection := ng.ConnectOutbound(neuron, chosenNeuron)

		// make an inbound connection targetNodeId <- neuron
		weights := randomWeights(1)
		ng.ConnectInboundWeighted(chosenNeuron, neuron, weights)
		return connection

	case ng.ACTUATOR:

		chosenActuator := cortex.FindActuator(targetNodeId)

		// make an outbound connection neuron -> targetNodeId
		connection := ng.ConnectOutbound(neuron, chosenActuator)

		// make an inbound connection targetNodeId <- neuron
		ng.ConnectInbound(chosenActuator, neuron)
		return connection

	default:
		log.Panicf("unexpected chosen node type")
		return nil
	}

}

func NeuronAddOutlinkRecurrent(neuron *ng.Neuron) *ng.OutboundConnection {

	// choose a random element B, where element B is another
	// neuron or a sensor which is not already connected
	// to this neuron.
	availableNodeIds := outboundConnectionCandidates(neuron)
	return neuronAddOutlink(neuron, availableNodeIds)
}

func NeuronAddOutlinkNonRecurrent(neuron *ng.Neuron) *ng.OutboundConnection {

	availableNodeIds := outboundConnectionCandidates(neuron)

	// remove any node id's which have a layer index >= neuron.LayerIndex
	nonRecurrentNodeIds := make([]*ng.NodeId, 0)
	for _, nodeId := range availableNodeIds {
		if nodeId.LayerIndex > neuron.NodeId.LayerIndex {
			nonRecurrentNodeIds = append(nonRecurrentNodeIds, nodeId)
		}

	}

	return neuronAddOutlink(neuron, nonRecurrentNodeIds)

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

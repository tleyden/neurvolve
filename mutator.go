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

	nodeIdLayerMap := cortex.NodeIdLayerMap()
	neuronLayerMap := cortex.NeuronLayerMap()
	randomLayer := neuronLayerMap.ChooseRandomLayer()
	neuron := cortex.CreateNeuronInLayer(randomLayer)
	upstreamNodeId := nodeIdLayerMap.ChooseNodeIdPrecedingLayer(randomLayer)
	if upstreamNodeId == nil {
		log.Printf("Unable to find upstream neuron, cannot add neuron")
		return nil
	}

	downstreamNodeId := findDownstreamNodeId(cortex, nodeIdLayerMap, randomLayer)
	if downstreamNodeId == nil {
		log.Printf("Unable to find downstream neuron, cannot add neuron")
		return nil
	}

	neuronAddInlinkFrom(neuron, upstreamNodeId)
	neuronAddOutlinkTo(neuron, downstreamNodeId)

	return neuron
}

func OutspliceRecurrent(cortex *ng.Cortex) *ng.Neuron {
	return nil
}

func OutspliceNonRecurrent(cortex *ng.Cortex) *ng.Neuron {

	numAttempts := len(cortex.AllNodeIds())

	for i := 0; i < numAttempts; i++ {
		neuronA := randomNeuron(cortex)
		outbound := randomNonRecurrentOutbound(neuronA)
		if outbound == nil {
			continue
		}

		nodeIdB := outbound.NodeId

		// figure out which layer neuronK will go in
		nodeIdLayerMap := cortex.NodeIdLayerMap()
		layerA := neuronA.NodeId.LayerIndex
		layerB := nodeIdB.LayerIndex
		layerK := nodeIdLayerMap.LayerBetweenOrNew(layerA, layerB)

		// create neuron K
		neuronK := cortex.CreateNeuronInLayer(layerK)

		// disconnect neuronA <-> nodeB
		nodeBConnector := cortex.FindInboundConnector(nodeIdB)
		ng.DisconnectOutbound(neuronA, nodeIdB)
		ng.DisconnectInbound(nodeBConnector, neuronA)

		// connect neuronA -> neuronK
		weights := randomWeights(1)
		ng.ConnectOutbound(neuronA, neuronK)
		ng.ConnectInboundWeighted(neuronK, neuronA, weights)

		// connect neuronK -> nodeB
		switch nodeIdB.NodeType {
		case ng.NEURON:
			neuronB := cortex.FindNeuron(nodeIdB)
			ng.ConnectOutbound(neuronK, neuronB)
			ng.ConnectInboundWeighted(nodeBConnector, neuronK, weights)
		case ng.ACTUATOR:
			actuatorB := cortex.FindActuator(nodeIdB)
			ng.ConnectOutbound(neuronK, actuatorB)
			ng.ConnectInbound(nodeBConnector, neuronK)
		}
		return neuronK

	}
	return nil

}

func randomNonRecurrentOutbound(neuron *ng.Neuron) *ng.OutboundConnection {
	for i := 0; i < len(neuron.Outbound); i++ {
		randIndex := RandomIntInRange(0, len(neuron.Outbound))
		outbound := neuron.Outbound[randIndex]
		if neuron.IsConnectionRecurrent(outbound) {
			continue
		} else {
			return outbound
		}
	}
	return nil
}

func randomNeuron(cortex *ng.Cortex) *ng.Neuron {
	neurons := cortex.Neurons
	randIndex := RandomIntInRange(0, len(neurons))
	return neurons[randIndex]
}

func AddNeuronRecurrent(cortex *ng.Cortex) *ng.Neuron {

	numAttempts := len(cortex.AllNodeIds())

	for i := 0; i < numAttempts; i++ {

		nodeIdLayerMap := cortex.NodeIdLayerMap()
		neuronLayerMap := cortex.NeuronLayerMap()
		randomLayer := neuronLayerMap.ChooseRandomLayer()
		neuron := cortex.CreateNeuronInLayer(randomLayer)
		inboundNodeId := findRecurrentInboundNodeId(cortex,
			nodeIdLayerMap,
			randomLayer)

		if inboundNodeId == nil {
			log.Printf("Warn: unable to find inbound node id")
			continue
		}

		outboundNodeId := findRecurrentOutboundNodeId(cortex,
			nodeIdLayerMap,
			randomLayer)

		if outboundNodeId == nil {
			log.Printf("Warn: unable to find outbound node id")
			continue
		}

		neuronAddInlinkFrom(neuron, inboundNodeId)
		neuronAddOutlinkTo(neuron, outboundNodeId)

		return neuron

	}
	return nil

}

// Find a nodeId suitable for use as an inbound node for a newly created
// neuron.  This can either be a sensor node or another neuron node (including
// the new neuron itself), but it cannot be an actuator node.
func findRecurrentInboundNodeId(cortex *ng.Cortex, layerMap ng.LayerToNodeIdMap, fromLayer float64) *ng.NodeId {

	keys := layerMap.Keys()
	actuatorLayer := keys[len(keys)-1]
	chosenNodeId := layerMap.ChooseNodeIdPrecedingLayer(actuatorLayer)
	return chosenNodeId

}

// Find a nodeId suitable for use as an outbound node for a newly created
// neuron.  This can either be a either another neuron node (including
// the new neuron itself), or an actuator (if it has space), but it cannot
// be a sensor node
func findRecurrentOutboundNodeId(cortex *ng.Cortex, layerMap ng.LayerToNodeIdMap, fromLayer float64) *ng.NodeId {

	numAttempts := len(cortex.AllNodeIds())

	keys := layerMap.Keys()

	sensorLayer := keys[0]

	for i := 0; i < numAttempts; i++ {
		chosenNodeId := layerMap.ChooseNodeIdFollowingLayer(sensorLayer)
		log.Printf("chosenNodeId: %v", chosenNodeId)
		if chosenNodeId.NodeType == ng.ACTUATOR {
			// make sure it has capacity for new incoming
			actuator := cortex.FindActuator(chosenNodeId)
			if actuator.CanAddInboundConnection() == false {
				continue
			}
		}
		return chosenNodeId

	}

	return nil

}

func findDownstreamNodeId(cortex *ng.Cortex, layerMap ng.LayerToNodeIdMap, fromLayer float64) *ng.NodeId {

	numAttempts := len(cortex.AllNodeIds())

	for i := 0; i < numAttempts; i++ {

		downstreamNodeId := layerMap.ChooseNodeIdFollowingLayer(fromLayer)

		if downstreamNodeId == nil {
			log.Printf("Unable to find downstream neuron, cannot add neuron")
			return nil
		}
		if downstreamNodeId.NodeType == ng.ACTUATOR {
			// make sure it has capacity for new incoming
			actuator := cortex.FindActuator(downstreamNodeId)
			if actuator.CanAddInboundConnection() == false {
				continue
			}
		}
		return downstreamNodeId
	}

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

	if len(availableNodeIds) == 0 {
		log.Printf("Warning: unable to add inlink to neuron: %v", neuron)
		return nil
	}

	randIndex := ng.RandomIntInRange(0, len(availableNodeIds))
	chosenNodeId := availableNodeIds[randIndex]
	return neuronAddInlinkFrom(neuron, chosenNodeId)

}

func neuronAddInlinkFrom(neuron *ng.Neuron, sourceNodeId *ng.NodeId) *ng.InboundConnection {

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

	if len(availableNodeIds) == 0 {
		log.Printf("Warning: unable to add inlink to neuron: %v", neuron)
		return nil
	}

	randIndex := ng.RandomIntInRange(0, len(availableNodeIds))
	chosenNodeId := availableNodeIds[randIndex]

	return neuronAddOutlinkTo(neuron, chosenNodeId)

}

func neuronAddOutlinkTo(neuron *ng.Neuron, targetNodeId *ng.NodeId) *ng.OutboundConnection {

	cortex := neuron.Cortex

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

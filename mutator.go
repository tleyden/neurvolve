package neurvolve

import (
	"github.com/couchbaselabs/logg"
	ng "github.com/tleyden/neurgo"
	"log"
	"math"
)

type OutboundChooser func(*ng.Neuron) *ng.OutboundConnection
type NeuronMutator func(*ng.Neuron) (bool, MutateResult)
type MutateResult interface{}
type CortexMutator func(*ng.Cortex) (bool, MutateResult)

func CortexMutatorsNonTopological() []CortexMutator {
	mutators := []CortexMutator{
		AddBias,
		RemoveBias,
		MutateWeights,
		ResetWeights,
		MutateActivation,
	}
	return mutators
}

func CortexMutatorsRecurrent(includeNonTopological bool) []CortexMutator {
	recurrentMutators := []CortexMutator{
		AddNeuronRecurrent,
		AddInlinkRecurrent,
		AddOutlinkRecurrent,
		OutspliceRecurrent,
	}
	if includeNonTopological {
		commonMutators := CortexMutatorsNonTopological()
		return append(recurrentMutators, commonMutators...)
	} else {
		return recurrentMutators
	}
}

func CortexMutatorsNonRecurrent(includeNonTopological bool) []CortexMutator {
	nonRecurrentMutators := []CortexMutator{
		AddNeuronNonRecurrent,
		AddInlinkNonRecurrent,
		AddOutlinkNonRecurrent,
		OutspliceNonRecurrent,
	}
	if includeNonTopological {
		commonMutators := CortexMutatorsNonTopological()
		return append(nonRecurrentMutators, commonMutators...)
	} else {
		return nonRecurrentMutators
	}
}

func inboundConnectionCandidates(neuron *ng.Neuron) []*ng.NodeId {

	if neuron == nil {
		log.Panicf("neuron is nil")
	}
	cortex := neuron.Cortex
	if cortex == nil {
		log.Panicf("neuron has no cortex associated: %v", neuron)
	}

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

func AddNeuronNonRecurrent(cortex *ng.Cortex) (bool, MutateResult) {
	numAttempts := len(cortex.AllNodeIds()) * 5

	for i := 0; i < numAttempts; i++ {

		nodeIdLayerMap := cortex.NodeIdLayerMap()
		neuronLayerMap := cortex.NeuronLayerMap()
		randomLayer := neuronLayerMap.ChooseRandomLayer()

		upstreamNodeId := nodeIdLayerMap.ChooseNodeIdPrecedingLayer(randomLayer)
		if upstreamNodeId == nil {
			continue
		}

		downstreamNodeId := findDownstreamNodeId(cortex, nodeIdLayerMap, randomLayer)
		if downstreamNodeId == nil {
			continue
		}

		neuron := cortex.CreateNeuronInLayer(randomLayer)
		neuronAddInlinkFrom(neuron, upstreamNodeId)
		neuronAddOutlinkTo(neuron, downstreamNodeId)

		return true, neuron

	}
	return false, nil
}

func AddNeuronRecurrent(cortex *ng.Cortex) (bool, MutateResult) {

	numAttempts := len(cortex.AllNodeIds()) * 5

	for i := 0; i < numAttempts; i++ {

		nodeIdLayerMap := cortex.NodeIdLayerMap()
		neuronLayerMap := cortex.NeuronLayerMap()
		randomLayer := neuronLayerMap.ChooseRandomLayer()
		inboundNodeId := findRecurrentInboundNodeId(cortex,
			nodeIdLayerMap,
			randomLayer)

		if inboundNodeId == nil {
			log.Printf("Warn: unable to find inbound node id")
			continue
		}

		if randomLayer == inboundNodeId.LayerIndex {
			continue
		}

		neuron := cortex.CreateNeuronInLayer(randomLayer)

		outboundNodeId := findRecurrentOutboundNodeId(cortex,
			nodeIdLayerMap,
			randomLayer)

		if outboundNodeId == nil {
			log.Printf("Warn: unable to find outbound node id")
			continue
		}

		neuronAddInlinkFrom(neuron, inboundNodeId)
		neuronAddOutlinkTo(neuron, outboundNodeId)

		return true, neuron

	}

	logg.LogTo("NEURVOLVE", "return false, nil")
	return false, nil

}

func Outsplice(cortex *ng.Cortex, chooseOutbound OutboundChooser) (bool, *ng.Neuron) {

	numAttempts := len(cortex.AllNodeIds()) * 5

	for i := 0; i < numAttempts; i++ {
		neuronA := randomNeuron(cortex)
		outbound := chooseOutbound(neuronA)
		if outbound == nil {
			continue
		}

		if neuronA.NodeId.UUID == outbound.NodeId.UUID {
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
		return true, neuronK

	}
	return false, nil

}

func OutspliceRecurrent(cortex *ng.Cortex) (bool, MutateResult) {
	chooseOutboundFunction := randomOutbound
	return Outsplice(cortex, chooseOutboundFunction)
}

func OutspliceNonRecurrent(cortex *ng.Cortex) (bool, MutateResult) {
	chooseOutboundFunction := randomNonRecurrentOutbound
	return Outsplice(cortex, chooseOutboundFunction)
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

func randomOutbound(neuron *ng.Neuron) *ng.OutboundConnection {
	for i := 0; i < len(neuron.Outbound); i++ {
		randIndex := RandomIntInRange(0, len(neuron.Outbound))
		return neuron.Outbound[randIndex]
	}
	return nil
}

func randomNeuron(cortex *ng.Cortex) *ng.Neuron {
	neurons := cortex.Neurons
	randIndex := RandomIntInRange(0, len(neurons))
	return neurons[randIndex]
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

	numAttempts := len(cortex.AllNodeIds()) * 5

	keys := layerMap.Keys()

	sensorLayer := keys[0]

	for i := 0; i < numAttempts; i++ {
		chosenNodeId := layerMap.ChooseNodeIdFollowingLayer(sensorLayer)
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

	numAttempts := len(cortex.AllNodeIds()) * 5

	for i := 0; i < numAttempts; i++ {

		downstreamNodeId := layerMap.ChooseNodeIdFollowingLayer(fromLayer)

		if downstreamNodeId == nil {
			log.Printf("findDownstreamNodeId unable to find downstream neuron, cannot add neuron")
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

func NeuronAddInlinkNonRecurrent(neuron *ng.Neuron) (bool, MutateResult) {

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

func NeuronAddInlinkRecurrent(neuron *ng.Neuron) (bool, MutateResult) {

	// choose a random element B, where element B is another
	// neuron or a sensor which is not already connected
	// to this neuron.
	availableNodeIds := inboundConnectionCandidates(neuron)
	return neuronAddInlink(neuron, availableNodeIds)
}

func neuronAddInlink(neuron *ng.Neuron, availableNodeIds []*ng.NodeId) (bool, *ng.InboundConnection) {

	if len(availableNodeIds) == 0 {
		log.Printf("Warning: unable to add inlink to neuron: %v", neuron)
		return false, nil
	}

	randIndex := ng.RandomIntInRange(0, len(availableNodeIds))
	chosenNodeId := availableNodeIds[randIndex]
	return true, neuronAddInlinkFrom(neuron, chosenNodeId)

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

	if neuron == nil {
		log.Panicf("Neuron is nil")
	}
	cortex := neuron.Cortex
	if cortex == nil {
		log.Panicf("Neuron has no cortex associated with it: %v", neuron)
	}
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

func neuronAddOutlink(neuron *ng.Neuron, availableNodeIds []*ng.NodeId) (bool, *ng.OutboundConnection) {

	if len(availableNodeIds) == 0 {
		log.Printf("Warning: unable to add outlink to neuron: %v", neuron)
		return false, nil
	}

	randIndex := ng.RandomIntInRange(0, len(availableNodeIds))
	chosenNodeId := availableNodeIds[randIndex]

	return true, neuronAddOutlinkTo(neuron, chosenNodeId)

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

func NeuronAddOutlinkRecurrent(neuron *ng.Neuron) (bool, MutateResult) {

	// choose a random element B, where element B is another
	// neuron or a sensor which is not already connected
	// to this neuron.
	availableNodeIds := outboundConnectionCandidates(neuron)
	return neuronAddOutlink(neuron, availableNodeIds)
}

func NeuronAddOutlinkNonRecurrent(neuron *ng.Neuron) (bool, MutateResult) {

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

func NeuronMutateWeights(neuron *ng.Neuron) (bool, MutateResult) {
	didPerturbAnyWeights := false
	probability := parameterPerturbProbability(neuron)
	for _, cxn := range neuron.Inbound {
		saturationBounds := []float64{-100000, 100000}
		didPerturbWeight := possiblyPerturbConnection(cxn, probability, saturationBounds)
		if didPerturbWeight == true {
			didPerturbAnyWeights = true
		}
	}
	return didPerturbAnyWeights, nil
}

func NeuronMutateActivation(neuron *ng.Neuron) (bool, MutateResult) {

	encodableActivations := ng.AllEncodableActivations()

	for i := 0; i < 100; i++ {

		// pick a random activation function from list
		randomIndex := ng.RandomIntInRange(0, len(encodableActivations))

		chosenActivation := encodableActivations[randomIndex]

		// if we chose a different activation than current one, use it
		if chosenActivation.Name != neuron.ActivationFunction.Name {
			neuron.ActivationFunction = chosenActivation
			return true, nil
		}
	}

	// if we got this far, something went wrong
	return false, nil

}

func NeuronResetWeights(neuron *ng.Neuron) (bool, MutateResult) {
	for _, cxn := range neuron.Inbound {
		for j, _ := range cxn.Weights {
			cxn.Weights[j] = RandomWeight()
		}
	}
	return true, nil
}

func NeuronAddBias(neuron *ng.Neuron) (bool, MutateResult) {
	if neuron.Bias == 0 {
		neuron.Bias = RandomBias()
		return true, nil
	}
	return false, nil
}

func NeuronRemoveBias(neuron *ng.Neuron) (bool, MutateResult) {
	if neuron.Bias != 0 {
		neuron.Bias = 0
		return true, nil
	}
	return false, nil
}

func RandomNeuronMutator(c *ng.Cortex, mutator NeuronMutator) (bool, MutateResult) {
	neuron := randomNeuron(c)
	return mutator(neuron)
}

func ReattemptingNeuronMutator(c *ng.Cortex, mutator NeuronMutator) (bool, MutateResult) {

	numAttempts := len(c.AllNodeIds()) * 5

	for i := 0; i < numAttempts; i++ {
		neuron := randomNeuron(c)
		ok, mutateResult := mutator(neuron)
		if ok {
			return ok, mutateResult
		}
	}
	return false, nil
}

func AddBias(cortex *ng.Cortex) (bool, MutateResult) {
	return RandomNeuronMutator(cortex, NeuronAddBias)
}

func RemoveBias(cortex *ng.Cortex) (bool, MutateResult) {
	return RandomNeuronMutator(cortex, NeuronRemoveBias)
}

func MutateWeights(cortex *ng.Cortex) (bool, MutateResult) {
	return RandomNeuronMutator(cortex, NeuronMutateWeights)
}

func ResetWeights(cortex *ng.Cortex) (bool, MutateResult) {
	return RandomNeuronMutator(cortex, NeuronResetWeights)
}

func MutateActivation(cortex *ng.Cortex) (bool, MutateResult) {
	return RandomNeuronMutator(cortex, NeuronMutateActivation)
}

func AddInlinkRecurrent(cortex *ng.Cortex) (bool, MutateResult) {
	return ReattemptingNeuronMutator(cortex, NeuronAddInlinkRecurrent)
}

func AddInlinkNonRecurrent(cortex *ng.Cortex) (bool, MutateResult) {
	return ReattemptingNeuronMutator(cortex, NeuronAddInlinkNonRecurrent)
}

func AddOutlinkRecurrent(cortex *ng.Cortex) (bool, MutateResult) {
	return ReattemptingNeuronMutator(cortex, NeuronAddOutlinkRecurrent)
}

func AddOutlinkNonRecurrent(cortex *ng.Cortex) (bool, MutateResult) {
	return ReattemptingNeuronMutator(cortex, NeuronAddOutlinkNonRecurrent)
}

func NoOpMutator(cortex *ng.Cortex) (success bool, result MutateResult) {
	success = true
	result = "nothing"
	return
}

func MutateAllWeightsBellCurve(cortex *ng.Cortex) (success bool, result MutateResult) {

	stdDev := DEFAULT_STD_DEVIATION

	for _, neuron := range cortex.Neurons {
		for _, inboundConnection := range neuron.Inbound {
			weights := inboundConnection.Weights
			for k, weight := range weights {
				newWeight := perturbParameterBellCurve(weight, stdDev)
				weights[k] = newWeight
			}
		}

		newBias := perturbParameterBellCurve(neuron.Bias, stdDev)
		neuron.Bias = newBias

	}

	success = true
	result = "nothing"
	return
}

func TopologyOrWeightMutator(cortex *ng.Cortex) (success bool, result MutateResult) {

	randomNumber := ng.RandomIntInRange(0, 100)
	if randomNumber > 95 {
		logg.LogTo("NEURVOLVE", "Attempting to mutate topology")

		// before we mutate the cortex, we need to init it,
		// otherwise things like Outsplice will fail because
		// there are no DataChan's.
		cortex.Init()

		// apply topological mutation
		didMutate := false
		includeNonTopological := false
		mutators := CortexMutatorsNonRecurrent(includeNonTopological)
		for i := 0; i <= 100; i++ {
			randInt := RandomIntInRange(0, len(mutators))
			mutator := mutators[randInt]
			didMutate, _ = mutator(cortex)
			if !didMutate {
				logg.LogTo("NEURVOLVE", "Mutate didn't work, retrying...")
				continue
			}
			break
		}
		logg.LogTo("NEURVOLVE", "did mutate: %v", didMutate)
		success = didMutate
	} else {
		logg.LogTo("NEURVOLVE", "Attempting to mutate weights")
		// mutate the weights
		saturationBounds := []float64{-10 * math.Pi, 10 * math.Pi}
		PerturbParameters(cortex, saturationBounds)
		success = true
	}

	result = "nothing"
	return
}

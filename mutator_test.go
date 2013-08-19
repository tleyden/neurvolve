package neurvolve

import (
	"github.com/couchbaselabs/go.assert"
	ng "github.com/tleyden/neurgo"
	"log"
	"testing"
)

func testCortex() *ng.Cortex {

	shouldReInit := false

	sensor := &ng.Sensor{
		NodeId:       ng.NewSensorId("sensor", 0.0),
		VectorLength: 2,
	}
	sensor.Init(shouldReInit)

	hiddenNeuron1 := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("hidden-neuron1", 0.15),
		Bias:               -30,
	}
	hiddenNeuron1.Init(shouldReInit)

	hiddenNeuron2 := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("hidden-neuron2", 0.25),
		Bias:               10,
	}
	hiddenNeuron2.Init(shouldReInit)

	hiddenNeuron3 := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("hidden-neuron3", 0.35),
		Bias:               10,
	}
	hiddenNeuron3.Init(shouldReInit)

	outputNeuron := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("output-neuron", 0.45),
		Bias:               -10,
	}
	outputNeuron.Init(shouldReInit)

	actuator := &ng.Actuator{
		NodeId:       ng.NewActuatorId("actuator", 0.5),
		VectorLength: 1,
	}
	actuator.Init(shouldReInit)

	sensor.ConnectOutbound(hiddenNeuron1)
	hiddenNeuron1.ConnectInboundWeighted(sensor, []float64{20, 20})

	hiddenNeuron1.ConnectOutbound(hiddenNeuron2)
	hiddenNeuron2.ConnectInboundWeighted(hiddenNeuron1, []float64{1})

	hiddenNeuron2.ConnectOutbound(hiddenNeuron3)
	hiddenNeuron3.ConnectInboundWeighted(hiddenNeuron2, []float64{1})

	hiddenNeuron3.ConnectOutbound(outputNeuron)
	outputNeuron.ConnectInboundWeighted(hiddenNeuron3, []float64{1})

	outputNeuron.ConnectOutbound(actuator)
	actuator.ConnectInbound(outputNeuron)

	nodeId := ng.NewCortexId("test-cortex")

	cortex := &ng.Cortex{
		NodeId: nodeId,
	}
	cortex.SetSensors([]*ng.Sensor{sensor})
	cortex.SetNeurons([]*ng.Neuron{hiddenNeuron1, hiddenNeuron2, hiddenNeuron3, outputNeuron})
	cortex.SetActuators([]*ng.Actuator{actuator})

	return cortex

}

func testCortexRecurrent() *ng.Cortex {

	shouldReInit := false

	sensor := &ng.Sensor{
		NodeId:       ng.NewSensorId("sensor", 0.0),
		VectorLength: 2,
	}
	sensor.Init(shouldReInit)

	hiddenNeuron1 := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("hidden-neuron1", 0.15),
		Bias:               -30,
	}
	hiddenNeuron1.Init(shouldReInit)

	hiddenNeuron2 := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("hidden-neuron2", 0.25),
		Bias:               10,
	}
	hiddenNeuron2.Init(shouldReInit)

	hiddenNeuron3 := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("hidden-neuron3", 0.35),
		Bias:               10,
	}
	hiddenNeuron3.Init(shouldReInit)

	outputNeuron := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("output-neuron", 0.45),
		Bias:               -10,
	}
	outputNeuron.Init(shouldReInit)

	actuator := &ng.Actuator{
		NodeId:       ng.NewActuatorId("actuator", 0.5),
		VectorLength: 1,
	}
	actuator.Init(shouldReInit)

	sensor.ConnectOutbound(hiddenNeuron1)
	hiddenNeuron1.ConnectInboundWeighted(sensor, []float64{20, 20})

	hiddenNeuron1.ConnectOutbound(hiddenNeuron2)
	hiddenNeuron2.ConnectInboundWeighted(hiddenNeuron1, []float64{1})

	// jumps over 2nd hidden layer, direct from 1st -> 3rd
	hiddenNeuron1.ConnectOutbound(hiddenNeuron3)
	hiddenNeuron3.ConnectInboundWeighted(hiddenNeuron1, []float64{1})

	hiddenNeuron2.ConnectOutbound(hiddenNeuron3)
	hiddenNeuron3.ConnectInboundWeighted(hiddenNeuron2, []float64{1})

	// recurrent connection
	hiddenNeuron3.ConnectOutbound(hiddenNeuron1)
	hiddenNeuron1.ConnectInboundWeighted(hiddenNeuron3, []float64{1})

	hiddenNeuron3.ConnectOutbound(outputNeuron)
	outputNeuron.ConnectInboundWeighted(hiddenNeuron3, []float64{1})

	outputNeuron.ConnectOutbound(actuator)
	actuator.ConnectInbound(outputNeuron)

	nodeId := ng.NewCortexId("test-cortex")

	cortex := &ng.Cortex{
		NodeId: nodeId,
	}
	cortex.SetSensors([]*ng.Sensor{sensor})
	cortex.SetNeurons([]*ng.Neuron{hiddenNeuron1, hiddenNeuron2, hiddenNeuron3, outputNeuron})
	cortex.SetActuators([]*ng.Actuator{actuator})

	return cortex

}

func verifyWeightsModified(neuron, neuronCopy *ng.Neuron) bool {
	foundModifiedWeight := false

	// make sure the weights have been modified for at least
	// one of the inbound connections
	originalInboundMap := neuron.InboundUUIDMap()
	copyInboundMap := neuronCopy.InboundUUIDMap()

	for uuid, connection := range originalInboundMap {
		connectionCopy := copyInboundMap[uuid]
		for i, weight := range connection.Weights {
			weightCopy := connectionCopy.Weights[i]
			if weight != weightCopy {
				foundModifiedWeight = true
				break
			}
		}
	}
	return foundModifiedWeight

}

func TestOutspliceRecurrent(t *testing.T) {

	ng.SeedRandom()

	numOutspliced := 0
	numOutsplicedWithNewLayer := 0
	numOutsplicedWithExistingLayer := 0
	numIterations := 100

	for i := 0; i < numIterations; i++ {

		cortex := testCortexRecurrent()
		numNeuronsBefore := len(cortex.Neurons)
		neuronLayerMapBefore := cortex.NeuronLayerMap()
		ok, mutateResult := OutspliceRecurrent(cortex)
		neuron := mutateResult.(*ng.Neuron)

		if !ok {
			continue
		} else {
			numOutspliced += 1
		}

		assert.True(t, neuron.ActivationFunction != nil)
		numNeuronsAfter := len(cortex.Neurons)
		assert.Equals(t, numNeuronsAfter, numNeuronsBefore+1)

		// should have 1 outbound and inbound
		assert.Equals(t, len(neuron.Inbound), 1)
		assert.Equals(t, len(neuron.Outbound), 1)

		// increment counter if layer added
		numLayersBefore := len(neuronLayerMapBefore)
		numLayersAfter := len(cortex.NeuronLayerMap())
		if numLayersAfter == numLayersBefore+1 {
			numOutsplicedWithNewLayer += 1
		} else {
			numOutsplicedWithExistingLayer += 1
		}

		// run network make sure it runs
		examples := ng.XnorTrainingSamples()
		fitness := cortex.Fitness(examples)
		assert.True(t, fitness >= 0)

	}

	assert.True(t, numOutspliced > 0)
	assert.True(t, numOutsplicedWithNewLayer > 0)
	assert.True(t, numOutsplicedWithExistingLayer > 0)

}

func TestOutspliceNonRecurrent(t *testing.T) {

	ng.SeedRandom()

	numOutspliced := 0
	numIterations := 100

	for i := 0; i < numIterations; i++ {

		cortex := testCortex()
		numNeuronsBefore := len(cortex.Neurons)
		neuronLayerMapBefore := cortex.NeuronLayerMap()
		ok, mutateResult := OutspliceNonRecurrent(cortex)
		neuron := mutateResult.(*ng.Neuron)

		if !ok {
			continue
		} else {
			numOutspliced += 1
		}

		assert.True(t, neuron.ActivationFunction != nil)
		numNeuronsAfter := len(cortex.Neurons)
		assert.Equals(t, numNeuronsAfter, numNeuronsBefore+1)

		// should have 1 outbound and inbound
		assert.Equals(t, len(neuron.Inbound), 1)
		assert.Equals(t, len(neuron.Outbound), 1)

		// should be no recurrent connections
		assert.Equals(t, len(neuron.RecurrentInboundConnections()), 0)
		assert.Equals(t, len(neuron.RecurrentOutboundConnections()), 0)

		// should have one more layer (this makes an assumption
		// about the testCortex architecture)
		numLayersBefore := len(neuronLayerMapBefore)
		numLayersAfter := len(cortex.NeuronLayerMap())
		assert.Equals(t, numLayersAfter, numLayersBefore+1)

		// run network make sure it runs
		examples := ng.XnorTrainingSamples()
		fitness := cortex.Fitness(examples)
		assert.True(t, fitness >= 0)

	}

	assert.True(t, numOutspliced > 0)

}

func TestAddNeuronNonRecurrent(t *testing.T) {

	ng.SeedRandom()

	numUnableToAdd := 0
	numIterations := 100

	for i := 0; i < numIterations; i++ {

		cortex := testCortex()
		numNeuronsBefore := len(cortex.Neurons)
		ok, mutateResult := AddNeuronNonRecurrent(cortex)
		neuron := mutateResult.(*ng.Neuron)

		if !ok {
			numUnableToAdd += 1
			continue
		}

		assert.True(t, neuron.ActivationFunction != nil)
		numNeuronsAfter := len(cortex.Neurons)
		addedNeuron := numNeuronsAfter == numNeuronsBefore+1
		assert.True(t, addedNeuron)
		if !addedNeuron {
			break
		}

		// should have 1 outbound and inbound
		assert.Equals(t, len(neuron.Inbound), 1)
		assert.Equals(t, len(neuron.Outbound), 1)

		// should be no recurrent connections
		assert.Equals(t, len(neuron.RecurrentInboundConnections()), 0)
		assert.Equals(t, len(neuron.RecurrentOutboundConnections()), 0)

		// run network make sure it runs
		examples := ng.XnorTrainingSamples()
		fitness := cortex.Fitness(examples)
		assert.True(t, fitness >= 0)

	}

	assert.True(t, numUnableToAdd <= (numIterations/3))

}

func TestAddNeuronRecurrent(t *testing.T) {

	ng.SeedRandom()

	numAdded := 0
	numIterations := 10

	for i := 0; i < numIterations; i++ {

		cortex := testCortex()
		numNeuronsBefore := len(cortex.Neurons)
		ok, mutateResult := AddNeuronRecurrent(cortex)
		neuron := mutateResult.(*ng.Neuron)

		if !ok {
			continue
		} else {
			numAdded += 1
		}

		assert.True(t, neuron != nil)
		assert.True(t, neuron.ActivationFunction != nil)
		numNeuronsAfter := len(cortex.Neurons)
		assert.Equals(t, numNeuronsAfter, numNeuronsBefore+1)

		// run network make sure it runs
		examples := ng.XnorTrainingSamples()
		fitness := cortex.Fitness(examples)
		assert.True(t, fitness >= 0)

	}

	assert.True(t, numAdded > 0)

}

func TestNeuronAddInlinkRecurrent(t *testing.T) {

	madeNonRecurrentInlink := false
	madeRecurrentInlink := false

	for i := 0; i < 100; i++ {
		xnorCortex := ng.XnorCortex()
		neuron := xnorCortex.NeuronUUIDMap()["output-neuron"]
		ok, mutateResult := NeuronAddInlinkRecurrent(neuron)
		if !ok {
			continue
		}
		inboundConnection := mutateResult.(*ng.InboundConnection)
		if neuron.IsInboundConnectionRecurrent(inboundConnection) {

			// the first time we make a nonRecurrentInlink,
			// test the network out
			if madeRecurrentInlink == false {
				// make sure the network actually works
				examples := ng.XnorTrainingSamples()
				fitness := xnorCortex.Fitness(examples)
				assert.True(t, fitness >= 0)

			}

			madeRecurrentInlink = true
		} else {

			// the first time we make a nonRecurrentInlink,
			// test the network out
			if madeNonRecurrentInlink == false {
				// make sure the network doesn't totally break
				examples := ng.XnorTrainingSamples()
				fitness := xnorCortex.Fitness(examples)
				assert.True(t, fitness >= 0)
			}

			madeNonRecurrentInlink = true

		}

	}

	assert.True(t, madeNonRecurrentInlink)
	assert.True(t, madeRecurrentInlink)

}

func TestNeuronAddInlinkNonRecurrent(t *testing.T) {

	ng.SeedRandom()

	madeNonRecurrentInlink := false
	madeRecurrentInlink := false
	firstTime := true

	// since it's stochastic, repeat the operation many times and make
	// sure that it always produces expected behavior
	for i := 0; i < 100; i++ {

		xnorCortex := ng.XnorCortex()
		sensor := xnorCortex.Sensors[0]
		neuron := xnorCortex.NeuronUUIDMap()["output-neuron"]
		hiddenNeuron1 := xnorCortex.NeuronUUIDMap()["hidden-neuron1"]
		targetLayerIndex := hiddenNeuron1.NodeId.LayerIndex

		// add a new neuron at the same layer index as the hidden neurons
		hiddenNeuron3 := &ng.Neuron{
			ActivationFunction: ng.EncodableSigmoid(),
			NodeId:             ng.NewNeuronId("hidden-neuron3", targetLayerIndex),
			Bias:               -30,
		}

		shouldReInit := false
		hiddenNeuron3.Init(shouldReInit)
		xnorCortex.Neurons = append(xnorCortex.Neurons, hiddenNeuron3)
		weights := randomWeights(sensor.VectorLength)
		sensor.ConnectOutbound(hiddenNeuron3)
		hiddenNeuron3.ConnectInboundWeighted(sensor, weights)

		ok, mutateResult := NeuronAddInlinkNonRecurrent(neuron)
		if !ok {
			continue
		}
		inboundConnection := mutateResult.(*ng.InboundConnection)

		if neuron.IsInboundConnectionRecurrent(inboundConnection) {
			madeRecurrentInlink = true
		} else {
			madeNonRecurrentInlink = true
		}

		if firstTime == true {

			// only two possibilities - the hiddenNeuron3 or the
			// sensor.  if it was the sensor, then the hiddenNeuron3
			// is "dangliing" and so lets connect it
			if inboundConnection.NodeId.UUID == "sensor" {
				weights2 := randomWeights(1)
				hiddenNeuron3.ConnectOutbound(neuron)
				neuron.ConnectInboundWeighted(hiddenNeuron3, weights2)
			}

			// run network make sure it runs
			examples := ng.XnorTrainingSamples()
			fitness := xnorCortex.Fitness(examples)
			assert.True(t, fitness >= 0)

			firstTime = false
		}

	}

	assert.True(t, madeNonRecurrentInlink)
	assert.False(t, madeRecurrentInlink)

}

func TestNeuronAddOutlinkNonRecurrent(t *testing.T) {

	ng.SeedRandom()

	madeNonRecurrentLink := false
	madeRecurrentLink := false

	for i := 0; i < 100; i++ {
		xnorCortex := testCortex()
		neuron := xnorCortex.NeuronUUIDMap()["hidden-neuron1"]
		ok, mutateResult := NeuronAddOutlinkNonRecurrent(neuron)
		if !ok {
			continue
		}
		outboundConnection := mutateResult.(*ng.OutboundConnection)
		if neuron.IsConnectionRecurrent(outboundConnection) {
			madeRecurrentLink = true
		} else {
			madeNonRecurrentLink = true
		}

	}

	assert.True(t, madeNonRecurrentLink)
	assert.False(t, madeRecurrentLink)

}

func TestNeuronAddOutlinkRecurrent(t *testing.T) {

	ng.SeedRandom()

	madeNonRecurrentLink := false
	madeRecurrentLink := false

	for i := 0; i < 100; i++ {
		xnorCortex := testCortex()

		neuron := xnorCortex.NeuronUUIDMap()["hidden-neuron1"]

		numOutlinksBefore := len(neuron.Outbound)

		ok, mutateResult := NeuronAddOutlinkRecurrent(neuron)
		if !ok {
			continue
		}
		outboundConnection := mutateResult.(*ng.OutboundConnection)

		numOutlinksAfter := len(neuron.Outbound)

		assert.Equals(t, numOutlinksBefore+1, numOutlinksAfter)

		if neuron.IsConnectionRecurrent(outboundConnection) {

			// the first time we make a nonRecurrentInlink,
			// test the network out
			if madeRecurrentLink == false {
				// make sure the network actually works
				examples := ng.XnorTrainingSamples()
				fitness := xnorCortex.Fitness(examples)
				assert.True(t, fitness >= 0)

			}

			madeRecurrentLink = true
		} else {

			// the first time we make a nonRecurrentInlink,
			// test the network out
			if madeNonRecurrentLink == false {
				// make sure the network doesn't totally break
				examples := ng.XnorTrainingSamples()
				fitness := xnorCortex.Fitness(examples)
				assert.True(t, fitness >= 0)
			}

			madeNonRecurrentLink = true

		}

	}

	assert.True(t, madeNonRecurrentLink)
	assert.True(t, madeRecurrentLink)

}

func TestNeuronMutateWeights(t *testing.T) {

	xnorCortex := ng.XnorCortex()
	neuron := xnorCortex.NeuronUUIDMap()["output-neuron"]
	assert.True(t, neuron != nil)
	neuronCopy := neuron.Copy()

	foundModifiedWeight := false
	for i := 0; i < 100; i++ {

		didMutateWeights, _ := NeuronMutateWeights(neuron)
		if didMutateWeights == true {

			foundModifiedWeight = verifyWeightsModified(neuron, neuronCopy)

		}

		if foundModifiedWeight == true {
			break
		}

	}

	assert.True(t, foundModifiedWeight == true)

}

func TestNeuronResetWeights(t *testing.T) {

	xnorCortex := ng.XnorCortex()
	neuron := xnorCortex.NeuronUUIDMap()["output-neuron"]
	assert.True(t, neuron != nil)
	neuronCopy := neuron.Copy()

	foundModifiedWeight := false
	for i := 0; i < 100; i++ {

		NeuronResetWeights(neuron)
		foundModifiedWeight = verifyWeightsModified(neuron, neuronCopy)

		if foundModifiedWeight == true {
			break
		}

	}

	assert.True(t, foundModifiedWeight == true)

}

func TestNeuronMutateActivation(t *testing.T) {

	ng.SeedRandom()
	neuron := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("neuron", 0.25),
		Bias:               10,
	}
	NeuronMutateActivation(neuron)
	assert.True(t, neuron.ActivationFunction != nil)
	assert.True(t, neuron.ActivationFunction.Name != ng.EncodableSigmoid().Name)

}

func TestNeuronRemoveBias(t *testing.T) {

	neuron := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("neuron", 0.25),
		Bias:               10,
	}
	shouldReInit := false
	neuron.Init(shouldReInit)
	NeuronRemoveBias(neuron)
	assert.True(t, neuron.Bias == 0)

}

func TestNeuronAddBias(t *testing.T) {

	// basic case where there is no bias

	neuron := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("neuron", 0.25),
	}
	shouldReInit := false
	neuron.Init(shouldReInit)

	NeuronAddBias(neuron)
	assert.True(t, neuron.Bias != 0)

	// make sure it treats 0 bias as not having a bias

	neuron = &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("neuron", 0.25),
		Bias:               0,
	}
	neuron.Init(shouldReInit)

	NeuronAddBias(neuron)
	assert.True(t, neuron.Bias != 0)

	// make sure it doesn't add a bias if there is an existing one

	neuron = &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("neuron", 0.25),
		Bias:               10,
	}
	neuron.Init(shouldReInit)
	NeuronAddBias(neuron)
	assert.True(t, neuron.Bias == 10)

}

func TestAddBias(t *testing.T) {
	xnorCortex := ng.XnorCortex()
	for _, neuron := range xnorCortex.Neurons {
		neuron.Bias = 0.0
	}
	beforeString := ng.JsonString(xnorCortex)
	AddBias(xnorCortex)
	afterString := ng.JsonString(xnorCortex)
	assert.True(t, beforeString != afterString)
}

func TestMutatorsThatAlwaysMutate(t *testing.T) {

	testCortex := testCortex()
	cortexMutators := []CortexMutator{
		RemoveBias,
		MutateWeights,
		ResetWeights,
		MutateActivation,
		AddInlinkRecurrent,
		AddInlinkNonRecurrent,
		AddOutlinkRecurrent,
		AddOutlinkNonRecurrent,
		AddNeuronNonRecurrent,
		AddNeuronRecurrent,
		OutspliceRecurrent,
		OutspliceNonRecurrent,
	}
	for _, cortexMutator := range cortexMutators {
		beforeString := ng.JsonString(testCortex)
		ok, _ := cortexMutator(testCortex)
		assert.True(t, ok)
		afterString := ng.JsonString(testCortex)
		hasChanged := beforeString != afterString
		if !hasChanged {
			log.Printf("!hasChanged.  beforeString/afterString: %v", beforeString)
		}
		assert.True(t, hasChanged)
		if !hasChanged {
			break
		}

	}

}

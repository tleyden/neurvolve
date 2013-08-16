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

func TestAddNeuronNonRecurrent(t *testing.T) {

	ng.SeedRandom()

	for i := 0; i < 100; i++ {

		cortex := testCortex()
		numNeuronsBefore := len(cortex.Neurons)
		neuron := AddNeuronNonRecurrent(cortex)
		assert.True(t, neuron != nil)
		assert.True(t, neuron.ActivationFunction != nil)
		numNeuronsAfter := len(cortex.Neurons)
		assert.Equals(t, numNeuronsAfter, numNeuronsBefore+1)

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

}

func TestAddNeuronRecurrent(t *testing.T) {

}

func TestNeuronAddInlinkRecurrent(t *testing.T) {

	madeNonRecurrentInlink := false
	madeRecurrentInlink := false

	for i := 0; i < 100; i++ {
		xnorCortex := ng.XnorCortex()
		neuron := xnorCortex.NeuronUUIDMap()["output-neuron"]
		inboundConnection := NeuronAddInlinkRecurrent(neuron)
		if neuron.IsInboundConnectionRecurrent(inboundConnection) {

			log.Printf("added inboundConnection: %v", inboundConnection)

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

		inboundConnection := NeuronAddInlinkNonRecurrent(neuron)
		log.Printf("new inbound: %v", inboundConnection)
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
		outboundConnection := NeuronAddOutlinkNonRecurrent(neuron)
		log.Printf("outbound: %v", outboundConnection)
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

		outboundConnection := NeuronAddOutlinkRecurrent(neuron)
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

		didMutateWeights := NeuronMutateWeights(neuron)
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

	// xnortCortex := ng.XnorCortex()

}

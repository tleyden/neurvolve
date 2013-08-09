package neurvolve

import (
	"github.com/couchbaselabs/go.assert"
	ng "github.com/tleyden/neurgo"
	"testing"
)

func TestNeuronAddBias(t *testing.T) {

	neuron := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("neuron", 0.25),
	}
	neuron.Init()

	neuron = NeuronAddBias(neuron)
	assert.True(t, neuron.Bias != 0)

	neuron = &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("neuron", 0.25),
		Bias:               0,
	}
	neuron.Init()

	neuron = NeuronAddBias(neuron)
	assert.True(t, neuron.Bias != 0)

}

func TestAddBias(t *testing.T) {

	// xnortCortex := ng.XnorCortex()

}

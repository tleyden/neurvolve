package neurvolve

import (
	ng "github.com/tleyden/neurgo"
)

func NeuronAddBias(neuron *ng.Neuron) *ng.Neuron {
	neuron.Bias = RandomBias()
	return neuron
}

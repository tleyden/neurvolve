package neurvolve

import (
	ng "github.com/tleyden/neurgo"
)

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

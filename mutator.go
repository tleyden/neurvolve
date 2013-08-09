package neurvolve

import (
	ng "github.com/tleyden/neurgo"
	"log"
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

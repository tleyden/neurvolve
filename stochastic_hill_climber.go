package neurvolve

import (
	"github.com/couchbaselabs/logg"
	ng "github.com/tleyden/neurgo"
	"log"
	"math"
	"math/rand"
)

type StochasticHillClimber struct {
	FitnessThreshold           float64
	MaxIterationsBeforeRestart int
	MaxAttempts                int
}

func (shc *StochasticHillClimber) Train(cortex *ng.Cortex, scape Scape) (fittestNeuralNet *ng.Cortex, succeeded bool) {

	numAttempts := 0

	fittestNeuralNet = cortex

	// Apply NN to problem and save fitness
	fitness := scape.Fitness(fittestNeuralNet)
	logg.LogTo("DEBUG", "initial fitness: %v", fitness)

	if fitness > shc.FitnessThreshold {
		succeeded = true
		return
	}

	for i := 0; ; i++ {

		// Save the genotype
		candidateNeuralNet := fittestNeuralNet.Copy()

		// Perturb synaptic weights and biases
		shc.perturbParameters(candidateNeuralNet)

		// Re-Apply NN to problem
		candidateFitness := scape.Fitness(candidateNeuralNet)
		logg.LogTo("DEBUG", "candidate fitness: %v", fitness)

		// If fitness of perturbed NN is higher, discard original NN and keep new
		// If fitness of original is higher, discard perturbed and keep old.

		if candidateFitness > fitness {
			logg.LogTo("DEBUG", "i: %v candidateFitness: %v > fitness: %v", i, candidateFitness, fitness)
			i = 0
			fittestNeuralNet = candidateNeuralNet
			fitness = candidateFitness
		}

		if ng.IntModuloProper(i, shc.MaxIterationsBeforeRestart) {
			log.Printf("** restart hill climber.  fitness: %f i/max: %d/%d", fitness, numAttempts, shc.MaxAttempts)
			numAttempts += 1
			i = 0
			shc.resetParametersToRandom(fittestNeuralNet)
		}

		if candidateFitness > shc.FitnessThreshold {
			logg.LogTo("DEBUG", "candidateFitness: %v > Threshold.  Success at i=%v", candidateFitness, i)
			succeeded = true
			break
		}

		if numAttempts >= shc.MaxAttempts {
			succeeded = false
			break
		}

	}

	return

}

func (shc *StochasticHillClimber) TrainExamples(cortex *ng.Cortex, examples []*ng.TrainingSample) (fittestNeuralNet *ng.Cortex, succeeded bool) {

	trainingSampleScape := &TrainingSampleScape{
		examples: examples,
	}
	return shc.Train(cortex, trainingSampleScape)

}

// 1. Each neuron in the neural net (weight or bias) will be chosen for perturbation
//    with a probability of 1/sqrt(nn_size)
// 2. Within the chosen neuron, the weights which will be perturbed will be chosen
//    with probability of 1/sqrt(parameters_size)
// 3. The intensity of the parameter perturbation will chosen with uniform distribution
//    of -pi and pi
func (shc *StochasticHillClimber) perturbParameters(cortex *ng.Cortex) {

	// pick the neurons to perturb (at least one)
	neurons := shc.chooseNeuronsToPerturb(cortex)

	for _, neuron := range neurons {
		shc.perturbNeuron(neuron)
	}

}

func (shc *StochasticHillClimber) resetParametersToRandom(cortex *ng.Cortex) {

	neurons := cortex.Neurons
	for _, neuronNode := range neurons {
		for _, cxn := range neuronNode.Inbound {
			for j, _ := range cxn.Weights {
				cxn.Weights[j] = ng.RandomInRange(-1*math.Pi, math.Pi)
			}
		}
		neuronNode.Bias = ng.RandomInRange(-1*math.Pi, math.Pi)
	}

}

func (shc *StochasticHillClimber) chooseNeuronsToPerturb(cortex *ng.Cortex) []*ng.Neuron {

	neuronsToPerturb := make([]*ng.Neuron, 0)

	// choose some random neurons to perturb.  we need at least one, so
	// keep looping until we've chosen at least one
	didChooseNeuron := false
	for {

		probability := shc.nodePerturbProbability(cortex)
		neurons := cortex.Neurons
		for _, neuronNode := range neurons {
			if rand.Float64() < probability {
				neuronsToPerturb = append(neuronsToPerturb, neuronNode)
				didChooseNeuron = true
			}
		}

		if didChooseNeuron {
			break
		}

	}
	return neuronsToPerturb

}

func (shc *StochasticHillClimber) nodePerturbProbability(cortex *ng.Cortex) float64 {
	neurons := cortex.Neurons
	numNeurons := len(neurons)
	return 1 / math.Sqrt(float64(numNeurons))
}

func (shc *StochasticHillClimber) perturbNeuron(neuron *ng.Neuron) {

	probability := parameterPerturbProbability(neuron)

	// keep trying until we've perturbed at least one parameter
	for {
		didPerturbWeight := false
		for _, cxn := range neuron.Inbound {
			didPerturbWeight = possiblyPerturbConnection(cxn, probability)
		}

		didPerturbBias := shc.possiblyPerturbBias(neuron, probability)

		// did we perturb anything?  if so, we're done
		if didPerturbWeight || didPerturbBias {
			break
		}

	}

}

func parameterPerturbProbability(neuron *ng.Neuron) float64 {
	numWeights := 0
	for _, connection := range neuron.Inbound {
		numWeights += len(connection.Weights)
	}
	return 1 / math.Sqrt(float64(numWeights))
}

func possiblyPerturbConnection(cxn *ng.InboundConnection, probability float64) bool {

	didPerturb := false
	for j, weight := range cxn.Weights {
		if rand.Float64() < probability {
			perturbedWeight := perturbParameter(weight)
			cxn.Weights[j] = perturbedWeight
			didPerturb = true
		}
	}
	return didPerturb

}

func (shc *StochasticHillClimber) possiblyPerturbBias(neuron *ng.Neuron, probability float64) bool {
	didPerturb := false
	if rand.Float64() < probability {
		bias := neuron.Bias
		perturbedBias := perturbParameter(bias)
		neuron.Bias = perturbedBias
		didPerturb = true
	}
	return didPerturb
}

func perturbParameter(parameter float64) float64 {

	parameter += ng.RandomInRange(-1*math.Pi, math.Pi)
	return parameter

}

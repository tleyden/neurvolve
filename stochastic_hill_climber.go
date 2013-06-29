package neurvolve

import (
	ng "github.com/tleyden/neurgo"
	"math"
	"math/rand"
)

type StochasticHillClimber struct {
	currentCandidate *ng.NeuralNetwork
	currentOptimal   *ng.NeuralNetwork
}

const MAX_ITERATIONS_BEFORE_RESTART = 10000

func (shc *StochasticHillClimber) Train(neuralNet *ng.NeuralNetwork, examples []*ng.TrainingSample) *ng.NeuralNetwork {

	fittestNeuralNet := neuralNet

	// Apply NN to problem and save fitness
	fitness := fittestNeuralNet.Fitness(examples)

	if fitness > ng.FITNESS_THRESHOLD {
		return fittestNeuralNet
	}

	for i := 0; ; i++ {

		// Save the genotype
		candidateNeuralNet := fittestNeuralNet.Copy()

		// Perturb synaptic weights and biases
		shc.perturbParameters(candidateNeuralNet)

		// Re-Apply NN to problem
		candidateFitness := candidateNeuralNet.Fitness(examples)

		// If fitness of perturbed NN is higher, discard original NN and keep new
		// If fitness of original is higher, discard perturbed and keep old.
		if candidateFitness > fitness {
			fittestNeuralNet = candidateNeuralNet
			fitness = candidateFitness
		}

		if ng.IntModuloProper(i, MAX_ITERATIONS_BEFORE_RESTART) {
			shc.resetParametersToRandom(fittestNeuralNet)
		}

		if candidateFitness > ng.FITNESS_THRESHOLD {
			break
		}

	}

	return fittestNeuralNet

}

// 1. Each neuron in the neural net (weight or bias) will be chosen for perturbation
//    with a probability of 1/sqrt(nn_size)
// 2. Within the chosen neuron, the weights which will be perturbed will be chosen
//    with probability of 1/sqrt(parameters_size)
// 3. The intensity of the parameter perturbation will chosen with uniform distribution
//    of -pi and pi
func (shc *StochasticHillClimber) perturbParameters(neuralNet *ng.NeuralNetwork) {

	// pick the neurons to perturb (at least one)
	neurons := shc.chooseNeuronsToPerturb(neuralNet)

	for _, neuron := range neurons {
		shc.perturbNeuron(neuron)
	}

}

func (shc *StochasticHillClimber) resetParametersToRandom(neuralNet *ng.NeuralNetwork) {

	neurons := neuralNet.Neurons()
	for _, neuronNode := range neurons {
		for _, cxn := range neuronNode.Inbound() {
			for j, _ := range cxn.Weights() {
				cxn.Weights()[j] = ng.RandomInRange(-1*math.Pi, math.Pi)
			}
		}
		neuronNode.Processor().SetBias(ng.RandomInRange(-1*math.Pi, math.Pi))
	}

}

func (shc *StochasticHillClimber) chooseNeuronsToPerturb(neuralNet *ng.NeuralNetwork) []*ng.Node {

	neuronsToPerturb := make([]*ng.Node, 0)

	// choose some random neurons to perturb.  we need at least one, so
	// keep looping until we've chosen at least one
	didChooseNeuron := false
	for {

		probability := shc.nodePerturbProbability(neuralNet)
		neurons := neuralNet.Neurons()
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

func (shc *StochasticHillClimber) nodePerturbProbability(neuralNet *ng.NeuralNetwork) float64 {
	neurons := neuralNet.Neurons()
	numNeurons := len(neurons)
	return 1 / math.Sqrt(float64(numNeurons))
}

func (shc *StochasticHillClimber) perturbNeuron(node *ng.Node) {

	probability := shc.parameterPerturbProbability(node)

	// keep trying until we've perturbed at least one parameter
	for {
		didPerturbWeight := false
		for _, cxn := range node.Inbound() {
			didPerturbWeight = shc.possiblyPerturbConnection(cxn, probability)
		}

		didPerturbBias := shc.possiblyPerturbBias(node, probability)

		// did we perturb anything?  if so, we're done
		if didPerturbWeight || didPerturbBias {
			break
		}

	}

}

func (shc *StochasticHillClimber) parameterPerturbProbability(node *ng.Node) float64 {
	numWeights := 0
	for _, connection := range node.Inbound() {
		numWeights += len(connection.Weights())
	}
	return 1 / math.Sqrt(float64(numWeights))
}

func (shc *StochasticHillClimber) possiblyPerturbConnection(cxn *ng.Connection, probability float64) bool {

	didPerturb := false
	for j, weight := range cxn.Weights() {
		if rand.Float64() < probability {
			perturbedWeight := shc.perturbParameter(weight)
			cxn.Weights()[j] = perturbedWeight
			didPerturb = true
		}
	}
	return didPerturb

}

func (shc *StochasticHillClimber) possiblyPerturbBias(node *ng.Node, probability float64) bool {
	didPerturb := false
	hasBias := node.Processor().HasBias()
	if hasBias && rand.Float64() < probability {
		bias := node.Processor().BiasValue()
		perturbedBias := shc.perturbParameter(bias)
		node.Processor().SetBias(perturbedBias)
		didPerturb = true
	}
	return didPerturb
}

func (shc *StochasticHillClimber) perturbParameter(parameter float64) float64 {

	parameter += ng.RandomInRange(-1*math.Pi, math.Pi)
	return parameter

}

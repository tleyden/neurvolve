package neurvolve

import (
	ng "github.com/tleyden/neurgo"
	"log"
)

type TopologyMutatingTrainer struct {
	FitnessThreshold           float64
	MaxIterationsBeforeRestart int
	MaxAttempts                int
	NumOutputLayerNodes        int
}

func (tmt *TopologyMutatingTrainer) Train(neuralNet *ng.NeuralNetwork, examples []*ng.TrainingSample) (fittestNeuralNet *ng.NeuralNetwork, succeeded bool) {

	ng.SeedRandom()

	originalNet := neuralNet.Copy()
	currentNeuralNet := neuralNet

	// Apply NN to problem and save fitness
	fitness := currentNeuralNet.Fitness(examples)

	if fitness > tmt.FitnessThreshold {
		succeeded = true
		return
	}

	for i := 0; ; i++ {

		log.Printf("before mutate.  i/max: %d/%d", i, tmt.MaxAttempts)
		printTopology(currentNeuralNet)
		// mutate the network: either add a node to existing layer, or new layer
		randInt := ng.RandomIntInRange(0, 2)
		if randInt == 0 {
			for {
				modifiedNeuralNet := AddNeuronNewRightLayer(currentNeuralNet)
				if tmt.checkConstraints(modifiedNeuralNet) {
					currentNeuralNet = modifiedNeuralNet
					break
				}
			}

		} else {
			for {
				modifiedNeuralNet := AddNeuronExistingLayer(currentNeuralNet)
				if tmt.checkConstraints(modifiedNeuralNet) {
					currentNeuralNet = modifiedNeuralNet
					break
				}
			}
		}

		log.Printf("after mutate.")
		printTopology(currentNeuralNet)

		// memetic step: call stochastic hill climber and see if it can solve it
		shc := &StochasticHillClimber{
			FitnessThreshold:           ng.FITNESS_THRESHOLD,
			MaxIterationsBeforeRestart: 100000,
			MaxAttempts:                4000000,
		}
		fittestNeuralNet, succeeded = shc.Train(currentNeuralNet, examples)

		if succeeded {
			succeeded = true
			break
		}

		if i >= tmt.MaxAttempts {
			succeeded = false
			break
		}

		if ng.IntModuloProper(i, tmt.MaxIterationsBeforeRestart) {
			log.Printf("** restart.  i/max: %d/%d", i, tmt.MaxAttempts)
			currentNeuralNet = originalNet.Copy()
		}

	}

	return

}

func (tmt *TopologyMutatingTrainer) checkConstraints(neuralNet *ng.NeuralNetwork) bool {
	numLayers := neuralNet.NumLayers()
	outputLayerIndex := numLayers - 2
	outputLayerNodes := neuralNet.NodesInLayer(outputLayerIndex)
	return len(outputLayerNodes) == tmt.NumOutputLayerNodes
}

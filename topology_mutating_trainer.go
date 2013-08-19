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

func (tmt *TopologyMutatingTrainer) Train(cortex *ng.Cortex, examples []*ng.TrainingSample) (fittestCortex *ng.Cortex, succeeded bool) {

	ng.SeedRandom()

	mutators := CortexMutatorsNonRecurrent()

	originalCortex := cortex.Copy()
	currentCortex := cortex

	// Apply NN to problem and save fitness
	fitness := currentCortex.Fitness(examples)

	if fitness > tmt.FitnessThreshold {
		succeeded = true
		return
	}

	for i := 0; ; i++ {

		log.Printf("before mutate.  i/max: %d/%d", i, tmt.MaxAttempts)

		// mutate the network
		randInt := RandomIntInRange(0, len(mutators))
		mutator := mutators[randInt]
		ok, _ := mutator(currentCortex)
		if !ok {
			log.Printf("mutate didn't work, retrying...")
			continue
		}

		log.Printf("after mutate.")

		// memetic step: call stochastic hill climber and see if it can solve it
		shc := &StochasticHillClimber{
			FitnessThreshold:           ng.FITNESS_THRESHOLD,
			MaxIterationsBeforeRestart: 100000,
			MaxAttempts:                4000000,
		}
		fittestCortex, succeeded = shc.Train(currentCortex, examples)

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
			currentCortex = originalCortex.Copy()
		}

	}

	return

}

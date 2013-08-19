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
		log.Printf("cortex: %v", ng.JsonString(currentCortex))

		// before we mutate the cortex, we need to init it,
		// otherwise things like Outsplice will fail because
		// there are no DataChan's.
		shouldReInit := false
		currentCortex.Init(shouldReInit)

		// mutate the network
		randInt := RandomIntInRange(0, len(mutators))
		mutator := mutators[randInt]
		ok, _ := mutator(currentCortex)
		if !ok {
			log.Printf("mutate didn't work, retrying...")
			continue
		}

		log.Printf("after mutate. cortex: %v", ng.JsonString(currentCortex))
		log.Printf("run stochastic hill climber")

		// memetic step: call stochastic hill climber and see if it can solve it
		shc := &StochasticHillClimber{
			FitnessThreshold:           ng.FITNESS_THRESHOLD,
			MaxIterationsBeforeRestart: 10000,
			MaxAttempts:                100000,
		}
		fittestCortex, succeeded = shc.Train(currentCortex, examples)
		log.Printf("stochastic hill climber finished.  succeeded: %v", succeeded)

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

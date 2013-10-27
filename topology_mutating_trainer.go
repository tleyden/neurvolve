package neurvolve

import (
	"fmt"
	"github.com/couchbaselabs/logg"
	ng "github.com/tleyden/neurgo"
)

type TopologyMutatingTrainer struct {
	MaxIterationsBeforeRestart int
	MaxAttempts                int
	StochasticHillClimber      *StochasticHillClimber
}

func (tmt *TopologyMutatingTrainer) Train(cortex *ng.Cortex, scape Scape) (fittestCortex *ng.Cortex, succeeded bool) {

	ng.SeedRandom()

	shc := tmt.StochasticHillClimber

	includeNonTopological := false
	mutators := CortexMutatorsNonRecurrent(includeNonTopological)

	originalCortex := cortex.Copy()

	currentCortex := cortex
	currentCortex.RenderSVGFile("/Users/traun/tmp/current.svg")

	// Apply NN to problem and save fitness
	logg.LogTo("MAIN", "Get initial fitness")
	fitness := scape.Fitness(currentCortex)
	logg.LogTo("MAIN", "Initial fitness: %v", fitness)

	if fitness > shc.FitnessThreshold {
		succeeded = true
		return
	}

	for i := 0; ; i++ {

		logg.LogTo("MAIN", "Before mutate.  i/max: %d/%d", i, tmt.MaxAttempts)

		// before we mutate the cortex, we need to init it,
		// otherwise things like Outsplice will fail because
		// there are no DataChan's.
		currentCortex.Init()

		// mutate the network
		randInt := RandomIntInRange(0, len(mutators))
		mutator := mutators[randInt]
		ok, _ := mutator(currentCortex)
		if !ok {
			logg.LogTo("MAIN", "Mutate didn't work, retrying...")
			continue
		}

		isValid := currentCortex.Validate()
		if !isValid {
			logg.LogPanic("Cortex did not validate")
		}

		filenameJson := fmt.Sprintf("cortex-%v.json", i)
		currentCortex.MarshalJSONToFile(filenameJson)
		filenameSvg := fmt.Sprintf("cortex-%v.svg", i)
		currentCortex.RenderSVGFile(filenameSvg)
		logg.LogTo("MAIN", "Post mutate cortex svg: %v json: %v", filenameSvg, filenameJson)

		logg.LogTo("MAIN", "Run stochastic hill climber..")

		// memetic step: call stochastic hill climber and see if it can solve it
		fittestCortex, succeeded = shc.Train(currentCortex, scape)
		logg.LogTo("MAIN", "stochastic hill climber finished.  succeeded: %v", succeeded)

		if succeeded {
			succeeded = true
			break
		}

		if i >= tmt.MaxAttempts {
			succeeded = false
			break
		}

		if ng.IntModuloProper(i, tmt.MaxIterationsBeforeRestart) {
			logg.LogTo("MAIN", "** Restart .  i/max: %d/%d", i, tmt.MaxAttempts)

			currentCortex = originalCortex.Copy()
			isValid := currentCortex.Validate()
			if !isValid {
				currentCortex.Repair() // TODO: remove workaround
				isValid = currentCortex.Validate()
				if !isValid {
					logg.LogPanic("Cortex could not be repaired")
				}
			}

		}

	}

	return

}

func (tmt *TopologyMutatingTrainer) TrainExamples(cortex *ng.Cortex, examples []*ng.TrainingSample) (fittestCortex *ng.Cortex, succeeded bool) {

	trainingSampleScape := &TrainingSampleScape{
		examples: examples,
	}
	return tmt.Train(cortex, trainingSampleScape)

}

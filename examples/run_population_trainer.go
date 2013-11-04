package main

import (
	"fmt"
	"github.com/couchbaselabs/logg"
	ng "github.com/tleyden/neurgo"
	nv "github.com/tleyden/neurvolve"
	"math"
	"time"
)

func RunPopulationTrainer() bool {

	ng.SeedRandom()

	// create population trainer ...
	pt := &nv.PopulationTrainer{
		FitnessThreshold: ng.FITNESS_THRESHOLD,
		MaxGenerations:   1000000,
		// CortexMutator:    nv.MutateAllWeightsBellCurve,
		// CortexMutator: nv.MutateWeights,
		CortexMutator: RandomNeuronMutator,
	}

	population := getInitialPopulation()
	scape := getScape()

	fitPopulation, succeeded := pt.Train(population, scape)

	if succeeded {
		logg.LogTo("MAIN", "Successfully trained!")

		fittestCortex := fitPopulation[0]
		logg.LogTo("MAIN", "Fitness: %v", fittestCortex.Fitness)

		filename := fmt.Sprintf("/tmp/checkerlution-%v.json", time.Now().Unix())
		logg.LogTo("MAIN", "Saving Cortex to %v", filename)
		cortex := fittestCortex.Cortex
		cortex.MarshalJSONToFile(filename)

		// verify it can now solve the training set
		verified := cortex.Verify(ng.XnorTrainingSamples())
		if !verified {
			logg.LogTo("MAIN", "Failed to verify neural net")
			succeeded = false
		}

	}

	if !succeeded {
		logg.LogTo("MAIN", "Failed to train neural net")
	}

	return succeeded

}

func RandomNeuronMutator(cortex *ng.Cortex) (success bool, result nv.MutateResult) {
	saturationBounds := []float64{-2 * math.Pi, 2 * math.Pi}
	nv.PerturbParameters(cortex, saturationBounds)
	success = true
	result = "nothing"
	return
}

type XnorScapeTwoPlayer struct {
	examples []*ng.TrainingSample
}

func (scape XnorScapeTwoPlayer) Fitness(cortex *ng.Cortex, opponent *ng.Cortex) float64 {
	cortexFitness := cortex.Fitness(scape.examples)
	opponentFitness := opponent.Fitness(scape.examples)
	logg.LogTo("DEBUG", "Cortex fitness: %v vs. Opponent: %v", cortexFitness, opponentFitness)
	// return cortexFitness - opponentFitness
	return cortexFitness
}

func getScape() nv.ScapeTwoPlayer {
	return XnorScapeTwoPlayer{
		examples: ng.XnorTrainingSamples(),
	}
}

func getInitialPopulation() []*ng.Cortex {

	population := make([]*ng.Cortex, 0)

	// create 30 of these
	for i := 0; i < 30; i++ {

		cortex := ng.XnorCortexUntrained()
		// cortex := ng.XnorCortex()

		population = append(population, cortex)

	}

	return population

}

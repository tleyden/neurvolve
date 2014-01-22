package main

import (
	"fmt"
	"github.com/couchbaselabs/logg"
	ng "github.com/tleyden/neurgo"
	nv "github.com/tleyden/neurvolve"
	"math"
	"time"
)

func RunPopulationTrainerLoop(maxIterations int) bool {

	// create population trainer ...
	pt := &nv.PopulationTrainer{
		FitnessThreshold: ng.FITNESS_THRESHOLD,
		MaxGenerations:   1000,
		CortexMutator:    nv.MutateAllWeightsBellCurve,
		// CortexMutator: nv.MutateWeights,
		// CortexMutator: RandomNeuronMutator,
		// CortexMutator:       nv.TopologyOrWeightMutator,
		NumOpponents:        5,
		SnapshotRequestChan: make(chan chan nv.EvaluatedCortexes),
	}
	nv.RegisterHandlers(pt)

	for i := 0; i < maxIterations; i++ {
		succeeded := RunPopulationTrainer(pt)
		if !succeeded {
			logg.LogTo("MAIN", "Population trainer succeeded %d times and then failed this time", i)
			return false
		} else {
			logg.LogTo("MAIN", "Population trainer succeeded %d times", i)

		}
	}
	logg.LogTo("MAIN", "Population succeeded %d times", maxIterations)
	return true
}

func RunPopulationTrainer(pt *nv.PopulationTrainer) bool {

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
	// -6pi <-> 6pi or anything lower doesn't work ..
	// -8pi <-> 8pi works but gets stuck sometimes
	saturationBounds := []float64{-100 * math.Pi, 100 * math.Pi}
	nv.PerturbParameters(cortex, saturationBounds)
	success = true
	result = "nothing"
	return
}

type XnorScapeTwoPlayer struct {
	examples []*ng.TrainingSample
}

func (scape XnorScapeTwoPlayer) FitnessAgainst(cortex *ng.Cortex, opponent *ng.Cortex) float64 {
	cortexFitness := cortex.Fitness(scape.examples)
	opponentFitness := opponent.Fitness(scape.examples)
	logg.LogTo("DEBUG", "Cortex fitness: %v vs. Opponent: %v", cortexFitness, opponentFitness)
	// return cortexFitness - opponentFitness
	return cortexFitness
}

func (scape XnorScapeTwoPlayer) Fitness(cortex *ng.Cortex) float64 {
	logg.LogPanic("Fitness not implemented")
	return 0.0
}

func getScape() nv.Scape {
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

package neurvolve

import (
	ng "github.com/tleyden/neurgo"
)

type PopulationTrainer struct {
	CortexMutator    CortexMutator
	FitnessThreshold float64
	MaxGenerations   int
}

type FitCortex struct {
	Cortex  *ng.Cortex
	Fitness float64
}

func (pt *PopulationTrainer) Train(population []*ng.Cortex, scape ScapeTwoPlayer) (population []FitCortex, succeeded bool) {

}

func (pt *PopulationTrainer) computeFitness(population []*ng.Cortex, scape ScapeTwoPlayer) (population []FitCortex) {

	fitCortexes := make([]FitCortex, len(population))
	for i, cortex := range population {
		opponents := pt.chooseRandomOpponents(population, 5)
		fitnessScores := make([]float64, len(opponents))
		for j, opponent := range opponents {
			fitnessScores[j] = scape.Fitness(cortex, opponent)
		}
		fitCortex := FitCortex{
			Cortex:  cortex,
			Fitness: pt.calculateAverage(fitnessScores),
		}
		fitCortexes[i] = fitCortex
	}

}

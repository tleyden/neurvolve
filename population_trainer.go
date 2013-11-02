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

	for i := 0; i < pt.MaxGenerations; i++ {

		fitCortexes := pt.computeFitness(population, scape)

		if pt.exceededFitnessThreshold(fitCortexes) {
			succeeded = true
			population = fitCortexes
			return
		}

		fitCortexes = pt.cullPopulation(fitCortexes)
		fitCortexes = pt.generateOffspring(fitCortexes)

	}

	population = fitCortexes
	return

}

func (pt *PopulationTrainer) computeFitness(population []*ng.Cortex, scape ScapeTwoPlayer) (fitCortexes []FitCortex) {

	fitCortexes = make([]FitCortex, len(population))
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

	fitCortexes = pt.sortByFitness(fitCortexes)

	return
}

func (pt *PopulationTrainer) exceededFitnessThreshold(fitCortexes []FitCortex) bool {

}

func (pt *PopulationTrainer) cullPopulation(population []FitCortex) (culledPopulation []FitCortex) {

}

func (pt *PopulationTrainer) generateOffspring(population []FitCortex) (withOffspring []FitCortex) {

	withOffspring = make([]FitCortex, 2*len(population))
	withOffspring = append(withOffspring, population...)

	for i, fitCortex := range population {

		cortex := fitCortex.Cortex
		offspringCortex := cortex.Copy()
		succeeded, _ := pt.CortexMutator(offspringCortex)
		if !succeeded {
			logg.LogPanic("Unable to mutate cortex: %v", offspringCortex)
		}
		fitCortexOffspring := FitCortex{
			Cortex:  offspringCortex,
			Fitness: 0.0,
		}

		withOffspring = append(withOffspring, fitCortexOffspring)

	}

}

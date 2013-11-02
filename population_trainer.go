package neurvolve

import (
	"github.com/couchbaselabs/logg"
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

func (pt *PopulationTrainer) Train(population []*ng.Cortex, scape ScapeTwoPlayer) (trainedPopulation []FitCortex, succeeded bool) {

	for i := 0; i < pt.MaxGenerations; i++ {

		fitCortexes := pt.computeFitness(population, scape)

		if pt.exceededFitnessThreshold(fitCortexes) {
			succeeded = true
			trainedPopulation = fitCortexes
			return
		}

		fitCortexes = pt.cullPopulation(fitCortexes)
		fitCortexes = pt.generateOffspring(fitCortexes)

		trainedPopulation = fitCortexes
	}

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

func (pt *PopulationTrainer) calculateAverage(fitnessScores []float64) float64 {
	// fixme
	return 0.0
}

func (pt *PopulationTrainer) chooseRandomOpponents(population []*ng.Cortex, numOpponents int) (opponents []*ng.Cortex) {
	// FIXME
	opponents = population
	return
}

func (pt *PopulationTrainer) sortByFitness(population []FitCortex) (sortedPopulation []FitCortex) {

	// FIXME..
	sortedPopulation = population
	return
}

func (pt *PopulationTrainer) exceededFitnessThreshold(fitCortexes []FitCortex) bool {

	// FIXME
	return false
}

func (pt *PopulationTrainer) cullPopulation(population []FitCortex) (culledPopulation []FitCortex) {

	// FIXME
	culledPopulation = population
	return
}

func (pt *PopulationTrainer) generateOffspring(population []FitCortex) (withOffspring []FitCortex) {

	withOffspring = make([]FitCortex, 2*len(population))
	withOffspring = append(withOffspring, population...)

	for _, fitCortex := range population {

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

	return

}

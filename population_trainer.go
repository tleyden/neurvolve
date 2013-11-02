package neurvolve

import (
	"github.com/couchbaselabs/logg"
	ng "github.com/tleyden/neurgo"
	"sort"
)

type PopulationTrainer struct {
	CortexMutator    CortexMutator
	FitnessThreshold float64
	MaxGenerations   int
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
		opponents := pt.chooseRandomOpponents(cortex, population, 5)
		fitnessScores := make([]float64, len(opponents))
		for j, opponent := range opponents {
			logg.LogTo("DEBUG", "Calc fitness of cortex vs opponent")
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

func (pt *PopulationTrainer) chooseRandomOpponents(cortex *ng.Cortex, population []*ng.Cortex, numOpponents int) (opponents []*ng.Cortex) {

	if numOpponents >= len(population) {
		logg.LogPanic("Not enough members of population to choose %d opponents", numOpponents)
	}

	opponents = make([]*ng.Cortex, 0)
	for i := 0; i < numOpponents; i++ {
		for {
			randInt := RandomIntInRange(0, len(population))
			randomCortex := population[randInt]
			if randomCortex == cortex {
				continue
			}
			opponents = append(opponents, randomCortex)
			break
		}

	}
	return

}

func (pt *PopulationTrainer) sortByFitness(population FitCortexArray) (sortedPopulation []FitCortex) {
	sort.Sort(population)
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

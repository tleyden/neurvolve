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
	NumOpponents     int
}

func (pt *PopulationTrainer) Train(population []*ng.Cortex, scape ScapeTwoPlayer) (trainedPopulation []FitCortex, succeeded bool) {

	fitCortexes := pt.fitPopulation(population)

	for i := 0; i < pt.MaxGenerations; i++ {

		fitCortexes = pt.computeFitness(fitCortexes, scape)

		if pt.exceededFitnessThreshold(fitCortexes) {
			succeeded = true
			trainedPopulation = fitCortexes
			return
		}

		fitCortexes = pt.cullPopulation(fitCortexes)

		fitCortexes = pt.generateOffspring(fitCortexes, scape)

		trainedPopulation = fitCortexes
	}

	return

}

func (pt *PopulationTrainer) fitPopulation(population []*ng.Cortex) (fitPopulation []FitCortex) {

	fitPopulation = make([]FitCortex, 0)
	for _, cortex := range population {
		fitCortex := FitCortex{
			Cortex:  cortex,
			Fitness: 0.0,
		}
		fitPopulation = append(fitPopulation, fitCortex)
	}
	return

}

func (pt *PopulationTrainer) computeFitness(population []FitCortex, scape ScapeTwoPlayer) (fitCortexes []FitCortex) {

	fitCortexes = make([]FitCortex, len(population))
	for i, fitCortex := range population {
		cortex := fitCortex.Cortex
		opponents := pt.chooseRandomOpponents(cortex, population, pt.NumOpponents)
		fitnessScores := make([]float64, len(opponents))
		for j, opponent := range opponents {
			fitnessScores[j] = scape.Fitness(cortex, opponent)
			logg.LogTo("DEBUG", "Fitness of cortex vs opponent: %v", fitnessScores[j])
		}
		fitCortexUpdated := FitCortex{
			Cortex:  cortex,
			Fitness: ng.Average(fitnessScores),
		}
		fitCortexes[i] = fitCortexUpdated
	}

	fitCortexes = pt.sortByFitness(fitCortexes)
	logg.LogTo("MAIN", "Highest fitness: %v", fitCortexes[0].Fitness)
	logg.LogTo("MAIN", "Lowest fitness: %v", fitCortexes[len(fitCortexes)-1].Fitness)

	return
}

func (pt *PopulationTrainer) chooseRandomOpponents(cortex *ng.Cortex, population []FitCortex, numOpponents int) (opponents []*ng.Cortex) {

	if numOpponents >= len(population) {
		logg.LogPanic("Not enough members of population to choose %d opponents", numOpponents)
	}

	opponents = make([]*ng.Cortex, 0)
	for i := 0; i < numOpponents; i++ {
		for {
			randInt := RandomIntInRange(0, len(population))
			randomFitCortex := population[randInt]
			if randomFitCortex.Cortex == cortex {
				continue
			}
			opponents = append(opponents, randomFitCortex.Cortex)
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
	for _, fitCortex := range fitCortexes {
		if fitCortex.Fitness >= pt.FitnessThreshold {
			return true
		}
	}
	return false
}

func (pt *PopulationTrainer) cullPopulation(population []FitCortex) (culledPopulation []FitCortex) {

	population = pt.sortByFitness(population)

	if len(population)%2 != 0 {
		logg.LogPanic("population size must be even")
	}

	logg.LogTo("DEBUG", "Before cull, highest fitness %v", population[0].Fitness)
	logg.LogTo("DEBUG", "Before cull, lowest fitness %v", population[len(population)-1].Fitness)

	culledPopulationSize := len(population) / 2
	culledPopulation = make([]FitCortex, 0)

	for i, fitCortex := range population {
		culledPopulation = append(culledPopulation, fitCortex)
		if i >= (culledPopulationSize - 1) {
			break
		}
	}

	logg.LogTo("DEBUG", "After cull, highest fitness %v", culledPopulation[0].Fitness)
	logg.LogTo("DEBUG", "After cull, lowest fitness %v", culledPopulation[len(culledPopulation)-1].Fitness)

	return
}

func (pt *PopulationTrainer) generateOffspring(population []FitCortex, scape ScapeTwoPlayer) (withOffspring []FitCortex) {

	logg.LogTo("DEBUG", "generateOffspring called with pop len %d", len(population))
	withOffspring = make([]FitCortex, 0)
	withOffspring = append(withOffspring, population...)

	for i, fitCortex := range population {

		cortex := fitCortex.Cortex
		offspringCortex := cortex.Copy()
		succeeded, _ := pt.CortexMutator(offspringCortex)
		if !succeeded {
			logg.LogPanic("Unable to mutate cortex: %v", offspringCortex)
		}

		// offspring should have different fitness than parent
		if scape != nil {

			offspringFitness := scape.Fitness(offspringCortex, cortex)
			parentFitness := scape.Fitness(cortex, offspringCortex)
			if offspringFitness == parentFitness {
				logg.LogPanic("%v == %v", offspringFitness, parentFitness)
			} else {

				if offspringFitness > parentFitness && i == 0 {
					logg.LogTo("MAIN", "%v > %v", offspringFitness, parentFitness)
					logg.LogTo("MAIN", "fittest parent just produced more fit offspring")
				}
			}

		}

		fitCortexOffspring := FitCortex{
			Cortex:  offspringCortex,
			Fitness: 0.0,
		}

		logg.LogTo("DEBUG", "original: %v", cortex)
		logg.LogTo("DEBUG", "offspring: %v", offspringCortex)

		withOffspring = append(withOffspring, fitCortexOffspring)

	}

	logg.LogTo("DEBUG", "generateOffspring returning pop len %d", len(withOffspring))

	return

}

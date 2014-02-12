package neurvolve

import (
	"fmt"
	"github.com/couchbaselabs/logg"
	ng "github.com/tleyden/neurgo"
	"sort"
)

func init() {
}

type PopulationTrainer struct {
	CortexMutator       CortexMutator
	FitnessThreshold    float64
	MaxGenerations      int
	CurrentGeneration   int
	NumOpponents        int
	SnapshotRequestChan chan chan EvaluatedCortexes
}

func (pt *PopulationTrainer) Train(population []*ng.Cortex, scape Scape, recorder Recorder) (trainedPopulation []EvaluatedCortex, succeeded bool) {

	evaldCortexes := pt.addEmptyFitnessScores(population)
	recorder.AddGeneration(evaldCortexes)

	for i := 0; i < pt.MaxGenerations; i++ {

		pt.CurrentGeneration = i
		pt.publishSnapshot(evaldCortexes)

		evaldCortexes = pt.computeFitness(evaldCortexes, scape, recorder)

		if pt.exceededFitnessThreshold(evaldCortexes) {
			succeeded = true
			trainedPopulation = evaldCortexes
			return
		}

		evaldCortexes = pt.cullPopulation(evaldCortexes)

		evaldCortexes = pt.generateOffspring(evaldCortexes)

		recorder.AddGeneration(evaldCortexes)

		trainedPopulation = evaldCortexes
	}

	return

}

func (pt *PopulationTrainer) publishSnapshot(evaldPopulation EvaluatedCortexes) {
	select {
	case responseChan := <-pt.SnapshotRequestChan:
		evaldPopulationCopy := make(EvaluatedCortexes, 0)
		for _, evaldCortex := range evaldPopulation {
			evaldCortexCopy := EvaluatedCortex{
				Fitness: evaldCortex.Fitness,
				Cortex:  evaldCortex.Cortex.Copy(),
			}
			evaldPopulationCopy = append(evaldPopulationCopy, evaldCortexCopy)
		}
		responseChan <- evaldPopulationCopy
	default:
	}
}

func (pt *PopulationTrainer) GetPopulationSnapshot() EvaluatedCortexes {
	responseChan := make(chan EvaluatedCortexes)
	pt.SnapshotRequestChan <- responseChan
	return <-responseChan
}

func (pt *PopulationTrainer) addEmptyFitnessScores(population []*ng.Cortex) (evaldPopulation []EvaluatedCortex) {

	evaldPopulation = make([]EvaluatedCortex, 0)
	for _, cortex := range population {

		evaldCortex := EvaluatedCortex{
			Cortex:   cortex,
			ParentId: cortex.NodeId.UUID, // no parent, set to self
			Fitness:  0.0,
		}
		evaldPopulation = append(evaldPopulation, evaldCortex)

	}
	return

}

func (pt *PopulationTrainer) computeFitness(population []EvaluatedCortex, scape Scape, recorder Recorder) (evaldCortexes []EvaluatedCortex) {

	evaldCortexes = make([]EvaluatedCortex, len(population))
	for i, evaldCortex := range population {
		cortex := evaldCortex.Cortex

		averageFitness := 0.0
		if pt.NumOpponents > 0 {

			opponents := pt.chooseRandomOpponents(cortex, population, pt.NumOpponents)

			fitnessScores := make([]float64, len(opponents))
			for j, opponent := range opponents {
				score := scape.FitnessAgainst(cortex, opponent)
				fitnessScores[j] = score
				recorder.AddFitnessScore(score, cortex, opponent)
			}
			averageFitness = ng.Average(fitnessScores)

		} else {
			averageFitness = scape.Fitness(cortex)
		}

		evaldCortexUpdated := EvaluatedCortex{
			Cortex:   cortex,
			ParentId: evaldCortex.ParentId,
			Fitness:  averageFitness,
		}
		evaldCortexes[i] = evaldCortexUpdated

	}

	evaldCortexes = pt.sortByFitness(evaldCortexes)

	return
}

func (pt *PopulationTrainer) chooseRandomOpponents(cortex *ng.Cortex, population []EvaluatedCortex, numOpponents int) (opponents []*ng.Cortex) {

	if numOpponents >= len(population) {
		logg.LogPanic("Not enough members of population to choose %d opponents", numOpponents)
	}

	opponents = make([]*ng.Cortex, 0)
	for i := 0; i < numOpponents; i++ {
		for {
			randInt := RandomIntInRange(0, len(population))
			randomEvaluatedCortex := population[randInt]
			if randomEvaluatedCortex.Cortex == cortex {
				continue
			}
			opponents = append(opponents, randomEvaluatedCortex.Cortex)
			break
		}

	}
	return

}

func (pt *PopulationTrainer) sortByFitness(population EvaluatedCortexes) (sortedPopulation []EvaluatedCortex) {
	sort.Sort(population)
	sortedPopulation = population
	return
}

func (pt *PopulationTrainer) exceededFitnessThreshold(evaldCortexes []EvaluatedCortex) bool {
	for _, evaldCortex := range evaldCortexes {
		if evaldCortex.Fitness >= pt.FitnessThreshold {
			return true
		}
	}
	return false
}

func (pt *PopulationTrainer) cullPopulation(population []EvaluatedCortex) (culledPopulation []EvaluatedCortex) {

	population = pt.sortByFitness(population)

	if len(population)%2 != 0 {
		logg.LogPanic("population size must be even")
	}

	culledPopulationSize := len(population) / 2
	culledPopulation = make([]EvaluatedCortex, 0)

	for i, evaldCortex := range population {
		culledPopulation = append(culledPopulation, evaldCortex)
		if i >= (culledPopulationSize - 1) {
			break
		}
	}

	return
}

func (pt *PopulationTrainer) generateOffspring(population []EvaluatedCortex) (withOffspring []EvaluatedCortex) {

	withOffspring = make([]EvaluatedCortex, 0)
	withOffspring = append(withOffspring, population...)

	for _, evaldCortex := range population {

		cortex := evaldCortex.Cortex
		offspringCortex := cortex.Copy()

		offspringNodeIdStr := fmt.Sprintf("cortex-%s", ng.NewUuid())
		offspringCortex.NodeId = ng.NewCortexId(offspringNodeIdStr)

		succeeded, _ := pt.CortexMutator(offspringCortex)
		if !succeeded {
			logg.LogPanic("Unable to mutate cortex: %v", offspringCortex)
		}

		evaldCortexOffspring := EvaluatedCortex{
			Cortex:              offspringCortex,
			ParentId:            cortex.NodeId.UUID,
			CreatedInGeneration: pt.CurrentGeneration,
			Fitness:             0.0,
		}

		withOffspring = append(withOffspring, evaldCortexOffspring)

	}

	return

}

func (pt *PopulationTrainer) dumpPopulationToLog(population []EvaluatedCortex) {

	for _, evaluatedCortex := range population {
		msg := fmt.Sprintf("%v f: %v", evaluatedCortex.Cortex.ExtraCompact(), evaluatedCortex.Fitness)
		logg.LogTo("NEURVOLVE", msg)
	}

}

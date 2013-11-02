package neurvolve

import (
	"github.com/couchbaselabs/go.assert"
	ng "github.com/tleyden/neurgo"
	"testing"
)

func TestChooseRandomOpponents(t *testing.T) {

	pt := &PopulationTrainer{}

	cortex := BasicCortex()
	opponent := BasicCortex()
	population := []*ng.Cortex{cortex, opponent}

	opponents := pt.chooseRandomOpponents(cortex, population, 1)
	assert.Equals(t, len(opponents), 1)
	assert.Equals(t, opponents[0], opponent)

}

func TestSortByFitness(t *testing.T) {

	pt := &PopulationTrainer{}

	fitCortexHigh := FitCortex{Fitness: 100.0}
	fitCortexLow := FitCortex{Fitness: -100.0}
	population := []FitCortex{fitCortexLow, fitCortexHigh}

	sortedPopulation := pt.sortByFitness(population)
	assert.Equals(t, len(population), len(sortedPopulation))
	assert.Equals(t, sortedPopulation[0], fitCortexHigh)
	assert.Equals(t, sortedPopulation[1], fitCortexLow)

}

func TestCullPopulation(t *testing.T) {
	fitCortexHigh := FitCortex{Fitness: 100.0}
	fitCortexLow := FitCortex{Fitness: -100.0}
	population := []FitCortex{fitCortexLow, fitCortexHigh}

	pt := &PopulationTrainer{}
	culledPopulation := pt.cullPopulation(population)
	assert.Equals(t, len(culledPopulation), 1)
	assert.Equals(t, culledPopulation[0], fitCortexHigh)

}

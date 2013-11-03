package neurvolve

import (
	"github.com/couchbaselabs/go.assert"
	"github.com/couchbaselabs/logg"
	ng "github.com/tleyden/neurgo"
	"testing"
)

func init() {
	logg.LogKeys["MAIN"] = true
	logg.LogKeys["DEBUG"] = true
	logg.LogKeys["NEURGO"] = true
	logg.LogKeys["SENSOR_SYNC"] = true
	logg.LogKeys["ACTUATOR_SYNC"] = true
	logg.LogKeys["NODE_PRE_SEND"] = true
	logg.LogKeys["NODE_POST_SEND"] = true
	logg.LogKeys["NODE_POST_RECV"] = true
	logg.LogKeys["NODE_STATE"] = true
}

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

func TestGenerateOffspring(t *testing.T) {

	fakeCortexMutator := func(cortex *ng.Cortex) (success bool, result MutateResult) {
		cortex.SetSensors(make([]*ng.Sensor, 0))
		result = "nothing"
		success = true
		return
	}

	pt := &PopulationTrainer{
		CortexMutator: fakeCortexMutator,
	}

	cortex1 := BasicCortex()
	cortex2 := BasicCortex()

	fitCortex1 := FitCortex{Fitness: 100.0, Cortex: cortex1}
	fitCortex2 := FitCortex{Fitness: -100.0, Cortex: cortex2}

	population := []FitCortex{fitCortex1, fitCortex2}
	offspringPopulation := pt.generateOffspring(population)
	assert.Equals(t, len(offspringPopulation), 2*len(population))

	offspringFitCortex := offspringPopulation[3]
	assert.Equals(t, offspringFitCortex.Fitness, 0.0)
	assert.Equals(t, len(offspringFitCortex.Cortex.Sensors), 0)

}

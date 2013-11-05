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
	logg.LogKeys["TEST"] = true
	logg.LogKeys["NEURGO"] = false
	logg.LogKeys["SENSOR_SYNC"] = false
	logg.LogKeys["ACTUATOR_SYNC"] = false
	logg.LogKeys["NODE_PRE_SEND"] = false
	logg.LogKeys["NODE_POST_SEND"] = false
	logg.LogKeys["NODE_POST_RECV"] = false
	logg.LogKeys["NODE_STATE"] = false
}

func TestTrain(t *testing.T) {

	fakeCortexMutator := func(cortex *ng.Cortex) (success bool, result MutateResult) {
		for _, neuron := range cortex.Neurons {
			neuron.Bias += 1
		}
		result = "nothing"
		success = true
		return
	}

	pt := &PopulationTrainer{
		FitnessThreshold: 1000,
		MaxGenerations:   1000000,
		CortexMutator:    fakeCortexMutator,
		NumOpponents:     1,
	}

	cortex1 := SingleNeuronCortex("cortex1")
	cortex2 := SingleNeuronCortex("cortex2")

	population := []*ng.Cortex{cortex1, cortex2}

	// inputs + expected outputs
	examples := []*ng.TrainingSample{
		{SampleInputs: [][]float64{[]float64{1}},
			ExpectedOutputs: [][]float64{[]float64{100}}},
	}

	scape := FakeScapeTwoPlayer{
		examples: examples,
	}
	trainedPopulation, succeeded := pt.Train(population, scape)
	logg.LogTo("TEST", "succeeded: %v", succeeded)
	logg.LogTo("TEST", "trainedPopulation: %v", trainedPopulation)

}

type FakeScapeTwoPlayer struct {
	examples []*ng.TrainingSample
}

func (scape FakeScapeTwoPlayer) Fitness(cortex *ng.Cortex, opponent *ng.Cortex) float64 {
	cortexFitness := cortex.Fitness(scape.examples)
	logg.LogTo("TEST", "getting fitness of cortex: %v", cortex)
	logg.LogTo("TEST", "cortexFitness: %v", cortexFitness)
	return cortexFitness
}

func TestChooseRandomOpponents(t *testing.T) {

	pt := &PopulationTrainer{}

	cortex := BasicCortex()
	fitCortex := FitCortex{
		Cortex:  cortex,
		Fitness: 0.0,
	}

	opponent := BasicCortex()
	fitOpponent := FitCortex{
		Cortex:  opponent,
		Fitness: 0.0,
	}

	population := []FitCortex{fitCortex, fitOpponent}

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
	offspringPopulation := pt.generateOffspring(population, nil)
	assert.Equals(t, len(offspringPopulation), 2*len(population))

	offspringFitCortex := offspringPopulation[3]
	assert.Equals(t, offspringFitCortex.Fitness, 0.0)
	assert.Equals(t, len(offspringFitCortex.Cortex.Sensors), 0)

}

package neurvolve

import (
	"github.com/couchbaselabs/go.assert"
	"github.com/couchbaselabs/logg"
	ng "github.com/tleyden/neurgo"
	"testing"
)

func init() {
	logg.LogKeys["MAIN"] = true
	logg.LogKeys["DEBUG"] = false
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
		NumOpponents:     0,
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
	recorder := NewNullRecorder()
	trainedPopulation, succeeded := pt.Train(population, scape, recorder)
	logg.LogTo("TEST", "succeeded: %v", succeeded)
	logg.LogTo("TEST", "trainedPopulation: %v", trainedPopulation)

}

type FakeScapeTwoPlayer struct {
	examples []*ng.TrainingSample
}

func (scape FakeScapeTwoPlayer) FitnessAgainst(cortex *ng.Cortex, opponent *ng.Cortex) float64 {
	logg.LogPanic("FitnessAgainst not implemented")
	return 0.0
}

func (scape FakeScapeTwoPlayer) Fitness(cortex *ng.Cortex) float64 {
	cortexFitness := cortex.Fitness(scape.examples)
	return cortexFitness
}

func TestChooseRandomOpponents(t *testing.T) {

	pt := &PopulationTrainer{}

	cortex := BasicCortex()
	evaldCortex := EvaluatedCortex{
		Cortex:  cortex,
		Fitness: 0.0,
	}

	opponent := BasicCortex()
	fitOpponent := EvaluatedCortex{
		Cortex:  opponent,
		Fitness: 0.0,
	}

	population := []EvaluatedCortex{evaldCortex, fitOpponent}

	opponents := pt.chooseRandomOpponents(cortex, population, 1)
	assert.Equals(t, len(opponents), 1)
	assert.Equals(t, opponents[0], opponent)

}

func TestSortByFitness(t *testing.T) {

	pt := &PopulationTrainer{}

	evaldCortexHigh := EvaluatedCortex{Fitness: 100.0}
	evaldCortexLow := EvaluatedCortex{Fitness: -100.0}
	population := []EvaluatedCortex{evaldCortexLow, evaldCortexHigh}

	sortedPopulation := pt.sortByFitness(population)
	assert.Equals(t, len(population), len(sortedPopulation))
	assert.Equals(t, sortedPopulation[0], evaldCortexHigh)
	assert.Equals(t, sortedPopulation[1], evaldCortexLow)

}

func TestCullPopulation(t *testing.T) {
	evaldCortexHigh := EvaluatedCortex{Fitness: 100.0}
	evaldCortexLow := EvaluatedCortex{Fitness: -100.0}
	population := []EvaluatedCortex{evaldCortexLow, evaldCortexHigh}

	pt := &PopulationTrainer{}
	culledPopulation := pt.cullPopulation(population)
	assert.Equals(t, len(culledPopulation), 1)
	assert.Equals(t, culledPopulation[0], evaldCortexHigh)

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

	evaldCortex1 := EvaluatedCortex{Fitness: 100.0, Cortex: cortex1}
	evaldCortex2 := EvaluatedCortex{Fitness: -100.0, Cortex: cortex2}

	population := []EvaluatedCortex{evaldCortex1, evaldCortex2}
	offspringPopulation := pt.generateOffspring(population)
	assert.Equals(t, len(offspringPopulation), 2*len(population))

	offspringEvaluatedCortex := offspringPopulation[3]
	assert.Equals(t, offspringEvaluatedCortex.Fitness, 0.0)
	assert.Equals(t, len(offspringEvaluatedCortex.Cortex.Sensors), 0)

}

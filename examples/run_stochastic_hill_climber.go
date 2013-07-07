package main

import (
	ng "github.com/tleyden/neurgo"
	"github.com/tleyden/neurvolve"
)

func RunWeightTraining() {

	ng.SeedRandom()

	// training set -- todo: examples := ng.XnorTrainingSamples()
	examples := []*ng.TrainingSample{
		// TODO: how to wrap this?
		{SampleInputs: [][]float64{[]float64{0, 1}}, ExpectedOutputs: [][]float64{[]float64{0}}},
		{SampleInputs: [][]float64{[]float64{1, 1}}, ExpectedOutputs: [][]float64{[]float64{1}}},
		{SampleInputs: [][]float64{[]float64{1, 0}}, ExpectedOutputs: [][]float64{[]float64{0}}},
		{SampleInputs: [][]float64{[]float64{0, 0}}, ExpectedOutputs: [][]float64{[]float64{1}}}}

	// create netwwork with topology capable of solving XNOR
	neuralNet := neurvolve.XnorNetworkUntrained()

	// verify it can not yet solve the training set (since training would be useless in that case)
	verified := neuralNet.Verify(examples)
	if verified {
		panic("neural net already trained, nothing to do")
	}

	shc := &neurvolve.StochasticHillClimber{
		FitnessThreshold:           ng.FITNESS_THRESHOLD,
		MaxIterationsBeforeRestart: 100000,
		MaxAttempts:                4000000,
	}
	neuralNetTrained, succeeded := shc.Train(neuralNet, examples)
	if !succeeded {
		panic("could not train neural net")
	}

	// verify it can now solve the training set
	verified = neuralNetTrained.Verify(examples)
	if !verified {
		panic("could not verify neural net")
	}

}

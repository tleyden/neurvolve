package main

import (
	"encoding/json"
	"fmt"
	ng "github.com/tleyden/neurgo"
	nv "github.com/tleyden/neurvolve"
	"log"
)

func RunTopologyMutatingTrainer() {

	ng.SeedRandom()

	// training set
	examples := ng.XnorTrainingSamples()

	// create netwwork with topology capable of solving XNOR
	cortex := ng.BasicCortex()

	// verify it can not yet solve the training set (since training would be useless in that case)
	verified := cortex.Verify(examples)
	if verified {
		panic("neural net already trained, nothing to do")
	}

	tmt := &nv.TopologyMutatingTrainer{
		FitnessThreshold:           ng.FITNESS_THRESHOLD,
		MaxAttempts:                25,
		MaxIterationsBeforeRestart: 5,
		NumOutputLayerNodes:        1,
	}
	cortexTrained, succeeded := tmt.Train(cortex, examples)
	if succeeded {
		nnJson, _ := json.Marshal(cortex)
		nnJsonString := fmt.Sprintf("%s", nnJson)
		log.Printf("Successfully trained net: %v", nnJsonString)
	}
	if !succeeded {
		panic("failed to train neural net")
	}

	// verify it can now solve the training set
	verified = cortexTrained.Verify(examples)
	if !verified {
		panic("failed to verify neural net")
	}

}

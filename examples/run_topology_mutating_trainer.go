package main

import (
	"encoding/json"
	"fmt"
	ng "github.com/tleyden/neurgo"
	"github.com/tleyden/neurvolve"
	"log"
	"math/rand"
	"time"
)

func main() {
	TestTopologyMutatingTrainer()
}

func TestTopologyMutatingTrainer() {

	rand.Seed(time.Now().UTC().UnixNano())

	// training set
	examples := ng.XnorTrainingSamples()

	// create netwwork with topology capable of solving XNOR
	neuralNet := neurvolve.BasicNetwork()

	// verify it can not yet solve the training set (since training would be useless in that case)
	verified := neuralNet.Verify(examples)
	if verified {
		panic("neural net already trained, nothing to do")
	}

	tmt := &neurvolve.TopologyMutatingTrainer{
		FitnessThreshold:           ng.FITNESS_THRESHOLD,
		MaxAttempts:                25,
		MaxIterationsBeforeRestart: 5,
		NumOutputLayerNodes:        1,
	}
	neuralNetTrained, succeeded := tmt.Train(neuralNet, examples)
	if succeeded {
		nnJson, _ := json.Marshal(neuralNet)
		nnJsonString := fmt.Sprintf("%s", nnJson)
		log.Printf("Successfully trained net: %v", nnJsonString)
	}
	if !succeeded {
		panic("failed to train neural net")
	}

	// verify it can now solve the training set
	verified = neuralNetTrained.Verify(examples)
	if !verified {
		panic("failed to verify neural net")
	}

}

package neurvolve

import (
	"encoding/json"
	"fmt"
	"github.com/couchbaselabs/go.assert"
	ng "github.com/tleyden/neurgo"
	"log"
	"math/rand"
	"testing"
	"time"
)

func TestTopologyMutatingTrainer(t *testing.T) {

	rand.Seed(time.Now().UTC().UnixNano())

	// training set
	examples := ng.XnorTrainingSamples()

	// create netwwork with topology capable of solving XNOR
	neuralNet := BasicNetwork()

	// verify it can not yet solve the training set (since training would be useless in that case)
	verified := neuralNet.Verify(examples)
	assert.False(t, verified)

	tmt := &TopologyMutatingTrainer{
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
	assert.True(t, succeeded)

	// verify it can now solve the training set
	verified = neuralNetTrained.Verify(examples)
	assert.True(t, verified)

}

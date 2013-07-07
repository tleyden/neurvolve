package neurvolve

import (
	"encoding/json"
	"fmt"
	"github.com/couchbaselabs/go.assert"
	ng "github.com/tleyden/neurgo"
	"log"
	"testing"
)

func TestAddNeuronNewRightLayer(t *testing.T) {
	neuralNet := BasicNetwork()
	nnCopy := AddNeuronNewRightLayer(neuralNet)
	origLayers := neuralNet.NumLayers()
	copyLayers := nnCopy.NumLayers()
	assert.Equals(t, origLayers+1, copyLayers)

	nodesByLayer := nnCopy.NodesByLayer()
	assert.Equals(t, len(nodesByLayer[1]), 1)
	assert.Equals(t, len(nodesByLayer[2]), 1)

}

func TestAddNeuronExistingLayer(t *testing.T) {
	neuralNet := BasicNetwork()
	nnCopy := AddNeuronExistingLayer(neuralNet)

	assert.Equals(t, len(neuralNet.Neurons())+1, len(nnCopy.Neurons()))
	nodesByLayer := nnCopy.NodesByLayer()
	assert.Equals(t, len(nodesByLayer[1]), 2)

}

func TestGenerateXnorTopology(t *testing.T) {

	ng.SeedRandom()

	succeeded := false
	neuralNet := BasicNetwork()
	for i := 0; i < 100; i++ {

		log.Printf("mutating neural net.  i: %d", i)
		// mutate the network: either add a node to existing layer, or new layer
		randInt := ng.RandomIntInRange(0, 2)
		if randInt == 0 {
			neuralNet = AddNeuronNewRightLayer(neuralNet)
		} else {
			neuralNet = AddNeuronExistingLayer(neuralNet)
		}

		printTopology(neuralNet)

		nodesByLayer := neuralNet.NodesByLayer()
		numLayer1 := len(nodesByLayer[1])
		numLayer2 := len(nodesByLayer[2])
		if numLayer1 == 2 && numLayer2 == 1 {
			log.Printf("stumbled upon xnor topology!")
			nnJson, _ := json.Marshal(neuralNet)
			nnJsonString := fmt.Sprintf("%s", nnJson)
			log.Printf("xnor topology: %v", nnJsonString)
			succeeded = true
			break
		} else {
			log.Printf("numLayer1: %d.  numLayer2: %d", numLayer1, numLayer2)

		}

		if i%5 == 0 { // <--- if this is 10, things slow to crawl.  why??
			// restart from beginning
			log.Printf("restart")
			neuralNet = BasicNetwork()
		}

	}
	assert.True(t, succeeded)

}

func TestAddInlink(t *testing.T) {
	assert.True(t, true)
}

func TestAddOutlink(t *testing.T) {
	assert.True(t, true)
}

func TestOutsplice(t *testing.T) {
	assert.True(t, true)
}

func TestInsplice(t *testing.T) {
	assert.True(t, true)
}

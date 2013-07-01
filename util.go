package neurvolve

import (
	"fmt"
	ng "github.com/tleyden/neurgo"
	"math"
)

func randomWeights(length int) []float64 {
	weights := []float64{}
	for i := 0; i < length; i++ {
		weights = append(weights, ng.RandomInRange(-1*math.Pi, math.Pi))
	}
	return weights
}

func printTopology(neuralNet *ng.NeuralNetwork) {
	nodesByLayer := neuralNet.NodesByLayer()
	for _, nodesInLayer := range nodesByLayer {
		fmt.Printf("[%d]-", len(nodesInLayer))
	}
	fmt.Printf("|\n")
}

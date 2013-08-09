package neurvolve

import (
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

func RandomBias() float64 {
	return ng.RandomInRange(-1*math.Pi, math.Pi)
}

func RandomWeight() float64 {
	return ng.RandomInRange(-1*math.Pi, math.Pi)
}

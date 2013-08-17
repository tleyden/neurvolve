package neurvolve

import (
	ng "github.com/tleyden/neurgo"
	"math"
)

func randomWeights(length int) []float64 {
	return ng.RandomWeights()
}

func RandomBias() float64 {
	return ng.RandomBias()
}

func RandomWeight() float64 {
	return ng.RandomWeight()
}

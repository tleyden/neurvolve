package neurvolve

import (
	ng "github.com/tleyden/neurgo"
)

func randomWeights(length int) []float64 {
	return ng.RandomWeights(length)
}

func RandomBias() float64 {
	return ng.RandomBias()
}

func RandomWeight() float64 {
	return ng.RandomWeight()
}

func RandomIntInRange(min, max int) int {
	return ng.RandomIntInRange(min, max)
}

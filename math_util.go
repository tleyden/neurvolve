package neurvolve

import (
	ng "github.com/tleyden/neurgo"
	"math"
	"math/rand"
)

const DEFAULT_STD_DEVIATION = 1.5

func perturbParameter(parameter float64, saturationBounds []float64) float64 {

	parameter += ng.RandomInRange(-1*math.Pi, math.Pi)

	lowerBound := saturationBounds[0]
	upperBound := saturationBounds[1]

	if parameter < lowerBound {
		parameter = lowerBound
	} else if parameter > upperBound {
		parameter = upperBound
	}

	return parameter

}

func perturbParameterBellCurve(parameter float64, desiredStdDev float64) float64 {

	desiredMean := parameter
	return rand.NormFloat64()*desiredStdDev + desiredMean

}

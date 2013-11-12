package neurvolve

import (
	ng "github.com/tleyden/neurgo"
	"math"
	"math/rand"
)

const DEFAULT_STD_DEVIATION = 1.5

func perturbParameter(parameter float64, saturationBounds []float64) float64 {

	parameter += ng.RandomInRange(-2*math.Pi, 2*math.Pi)
	return saturate(parameter, saturationBounds)

}

func saturate(parameter float64, saturationBounds []float64) float64 {

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
	parameter = rand.NormFloat64()*desiredStdDev + desiredMean

	saturationBounds := []float64{-10 * math.Pi, 10 * math.Pi} // todo: pass this in as a parameter

	return saturate(parameter, saturationBounds)

}

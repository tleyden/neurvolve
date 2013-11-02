package neurvolve

import (
	ng "github.com/tleyden/neurgo"
)

type FitCortex struct {
	Cortex  *ng.Cortex
	Fitness float64
}

type FitCortexArray []FitCortex

func (fca FitCortexArray) Len() int {
	return len(fca)
}

func (fca FitCortexArray) Less(i, j int) bool {
	return fca[i].Fitness > fca[j].Fitness
}

func (fca FitCortexArray) Swap(i, j int) {
	fca[i], fca[j] = fca[j], fca[i]
}

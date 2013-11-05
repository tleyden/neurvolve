package neurvolve

import (
	ng "github.com/tleyden/neurgo"
)

// A Cortex along with a fitness evaluation
type EvaluatedCortex struct {
	Cortex  *ng.Cortex
	Fitness float64
}

type EvaluatedCortexArray []EvaluatedCortex

func (fca EvaluatedCortexArray) Len() int {
	return len(fca)
}

func (fca EvaluatedCortexArray) Less(i, j int) bool {
	return fca[i].Fitness > fca[j].Fitness
}

func (fca EvaluatedCortexArray) Swap(i, j int) {
	fca[i], fca[j] = fca[j], fca[i]
}

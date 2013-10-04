package neurvolve

import (
	ng "github.com/tleyden/neurgo"
)

type TrainingSampleScape struct {
	examples []*ng.TrainingSample
}

func (scape TrainingSampleScape) Fitness(cortex *ng.Cortex) float64 {
	return cortex.Fitness(scape.examples)
}

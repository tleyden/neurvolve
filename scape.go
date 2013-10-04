package neurvolve

import (
	ng "github.com/tleyden/neurgo"
)

type Scape interface {
	Fitness(cortex *ng.Cortex) float64
}

package neurvolve

import (
	ng "github.com/tleyden/neurgo"
)

// Record the evolutionary history of a population
type Recorder interface {

	// This is called as soon as a generation is created.  It could
	// be either the initial generation, or an offspring generation
	AddGeneration(evaldCortexes []EvaluatedCortex)

	// This is called after two cortexes face off
	AddFitnessScore(score float64, cortex *ng.Cortex, opponent *ng.Cortex)
}

package neurvolve

import (
	"github.com/couchbaselabs/logg"
	ng "github.com/tleyden/neurgo"
)

type TrainingSampleScape struct {
	examples []*ng.TrainingSample
}

func (scape TrainingSampleScape) Fitness(cortex *ng.Cortex) float64 {
	return cortex.Fitness(scape.examples)
}

func (scape TrainingSampleScape) FitnessAgainst(cortex *ng.Cortex, opponentCortex *ng.Cortex) (fitness float64) {
	// return cortex.Fitness(scape.examples) - opponentCortex.Fitness(scape.examples)
	logg.LogPanic("Cannot calculate fitness against another cortex")
	return 0.0
}

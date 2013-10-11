package neurvolve

import (
	ng "github.com/tleyden/neurgo"
)

// The idea behind a "scape", aka "simulation landscape", is that
// the scape puts the neural network in a simulation and measures
// it's fitness.  An example simulation might be a "predator vs prey"
// simulation, where the fitness would be higher if the neural network
// survived longer.  Another example would be a checkers game, where
// the fitness would be higher if it was able to beat an opponent.
type Scape interface {
	Fitness(cortex *ng.Cortex) float64
}

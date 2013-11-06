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

	// This is appropriate when the opponent is baked into
	// the scape, or it's just being evaluated against
	// training examples (as in the xor case)
	Fitness(cortex *ng.Cortex) float64

	// Calculate the fitness against an actual opponent
	FitnessAgainst(cortex *ng.Cortex, opponent *ng.Cortex) float64
}

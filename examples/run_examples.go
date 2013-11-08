package main

import (
	"github.com/couchbaselabs/logg"
	ng "github.com/tleyden/neurgo"
)

func init() {
	logg.LogKeys["MAIN"] = true
	logg.LogKeys["DEBUG"] = false
	ng.SeedRandom()
}

// How to run this code:
// $ cd examples
// $ go build -v && go run run_examples.go run_stochastic_hill_climber.go run_topology_mutating_trainer.go
func main() {

	// RunStochasticHillClimber()
	// success := MultiRunTopologyMutatingTrainer()

	/*
		success := RunTopologyMutatingTrainer()
		if !success {
			logg.LogPanic("Failed to run example")
		}
	*/

	success := RunPopulationTrainerLoop(1)
	if !success {
		logg.LogPanic("Failed to run population trainer")
	}

}

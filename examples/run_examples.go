package main

import (
	"github.com/couchbaselabs/logg"
)

// How to run this code:
// $ cd examples
// $ go build -v && go run run_examples.go run_stochastic_hill_climber.go run_topology_mutating_trainer.go
func main() {

	RunStochasticHillClimber()
	// success := MultiRunTopologyMutatingTrainer()
	success := RunTopologyMutatingTrainer()
	if !success {
		logg.LogPanic("Failed to run example")
	}

}

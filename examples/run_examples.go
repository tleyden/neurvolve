package main

import (
	_ "expvar"
	"github.com/couchbaselabs/logg"
	ng "github.com/tleyden/neurgo"
	"net/http"
)

func init() {
	logg.LogKeys["MAIN"] = true
	logg.LogKeys["DEBUG"] = false
	logg.LogKeys["NEURVOLVE"] = true
	logg.LogKeys["NODE_STATE"] = false
	ng.SeedRandom()
}

// How to run this code:
// $ cd examples
// $ go build -v && go run run_examples.go run_stochastic_hill_climber.go run_topology_mutating_trainer.go
func main() {

	go http.ListenAndServe(":8080", nil)

	// RunStochasticHillClimber()
	// success := MultiRunTopologyMutatingTrainer()

	/*
		success := RunTopologyMutatingTrainer()
		if !success {
			logg.LogPanic("Failed to run example")
		}
	*/

	// success := RunPopulationTrainerLoop(150)
	success := RunPopulationTrainer()
	if !success {
		logg.LogFatal("Failed to run population trainer")
	}

}

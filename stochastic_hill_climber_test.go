package neurvolve

import (
	"encoding/json"
	"fmt"
	"github.com/couchbaselabs/go.assert"
	ng "github.com/tleyden/neurgo"
	"log"
	"testing"
)

func TestPerturbParameters(t *testing.T) {

	cortex := ng.XnorCortex()

	nnJson, _ := json.Marshal(cortex)
	nnJsonString := fmt.Sprintf("%s", nnJson)

	shc := new(StochasticHillClimber)

	shc.perturbParameters(cortex)

	nnJsonAfter, _ := json.Marshal(cortex)
	nnJsonStringAfter := fmt.Sprintf("%s", nnJsonAfter)

	// the json should be different after we perturb it
	assert.NotEquals(t, nnJsonString, nnJsonStringAfter)

}

func ExpensiveTestUnmarshalCortexFitness(t *testing.T) {

	// this test is disabled by default since it can take a long time

	// this was a real net that was evolved by the topological mutator
	// before it went through the memetic step.
	jsonString := `{"NodeId":{"UUID":"cortex","NodeType":"CORTEX","LayerIndex":0},"Sensors":[{"NodeId":{"UUID":"sensor","NodeType":"SENSOR","LayerIndex":0},"VectorLength":2,"Outbound":[{"NodeId":{"UUID":"neuron","NodeType":"NEURON","LayerIndex":0.25}},{"NodeId":{"UUID":"todo=394057419","NodeType":"NEURON","LayerIndex":0.25}}]}],"Neurons":[{"NodeId":{"UUID":"neuron","NodeType":"NEURON","LayerIndex":0.25},"Bias":0,"Inbound":[{"NodeId":{"UUID":"sensor","NodeType":"SENSOR","LayerIndex":0},"Weights":[20,20]}],"Outbound":[{"NodeId":{"UUID":"todo=3342881449","NodeType":"NEURON","LayerIndex":0.375}}],"ActivationFunction":{"Name":"sigmoid"}},{"NodeId":{"UUID":"todo=3342881449","NodeType":"NEURON","LayerIndex":0.375},"Bias":-0.36421027459743627,"Inbound":[{"NodeId":{"UUID":"neuron","NodeType":"NEURON","LayerIndex":0.25},"Weights":[1.8610633947514623]},{"NodeId":{"UUID":"todo=394057419","NodeType":"NEURON","LayerIndex":0.25},"Weights":[0.45355591271633067]}],"Outbound":[{"NodeId":{"UUID":"actuator","NodeType":"ACTUATOR","LayerIndex":0.5}}],"ActivationFunction":{"Name":"sigmoid"}},{"NodeId":{"UUID":"todo=394057419","NodeType":"NEURON","LayerIndex":0.25},"Bias":0.4665982575854781,"Inbound":[{"NodeId":{"UUID":"sensor","NodeType":"SENSOR","LayerIndex":0},"Weights":[0.6618610678776169,3.065256532332561]}],"Outbound":[{"NodeId":{"UUID":"todo=3342881449","NodeType":"NEURON","LayerIndex":0.375}}],"ActivationFunction":{"Name":"tanh"}}],"Actuators":[{"NodeId":{"UUID":"actuator","NodeType":"ACTUATOR","LayerIndex":0.5},"VectorLength":1,"Inbound":[{"NodeId":{"UUID":"todo=3342881449","NodeType":"NEURON","LayerIndex":0.375},"Weights":null}]}]}`

	jsonBytes := []byte(jsonString)

	cortex := &ng.Cortex{}
	err := json.Unmarshal(jsonBytes, cortex)
	if err != nil {
		log.Fatal(err)
	}
	assert.True(t, err == nil)

	shc := &StochasticHillClimber{
		FitnessThreshold:           ng.FITNESS_THRESHOLD,
		MaxIterationsBeforeRestart: 100000,
		MaxAttempts:                10,
	}
	examples := ng.XnorTrainingSamples()
	cortexTrained, succeeded := shc.Train(cortex, examples)
	assert.True(t, succeeded)

	// verify it can now solve the training set
	verified := cortexTrained.Verify(examples)
	assert.True(t, verified)

	fitness := cortexTrained.Fitness(examples)
	log.Printf("Final fitness: %v", fitness)

}

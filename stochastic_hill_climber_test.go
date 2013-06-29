package neurgo

import (
	"encoding/json"
	"fmt"
	"github.com/couchbaselabs/go.assert"
	ng "github.com/tleyden/neurgo"
	"math"
	"testing"
)

func TestPerturbParameters(t *testing.T) {

	neuralNet := ng.XnorCondensedNetwork()

	nnJson, _ := json.Marshal(neuralNet)
	nnJsonString := fmt.Sprintf("%s", nnJson)

	shc := new(StochasticHillClimber)

	shc.perturbParameters(neuralNet)

	nnJsonAfter, _ := json.Marshal(neuralNet)
	nnJsonStringAfter := fmt.Sprintf("%s", nnJsonAfter)

	// the json should be different after we perturb it
	assert.NotEquals(t, nnJsonString, nnJsonStringAfter)

}

// create netwwork with topology capable of solving XNOR, but which
// has not been trained yet
func xnorNetworkUntrained() *ng.NeuralNetwork {

	randBias_1 := ng.RandomInRange(-1*math.Pi, math.Pi)
	randBias_2 := ng.RandomInRange(-1*math.Pi, math.Pi)
	randBias_3 := ng.RandomInRange(-1*math.Pi, math.Pi)

	randWeights_1 := []float64{
		ng.RandomInRange(-1*math.Pi, math.Pi),
		ng.RandomInRange(-1*math.Pi, math.Pi),
	}

	randWeights_2 := []float64{
		ng.RandomInRange(-1*math.Pi, math.Pi),
		ng.RandomInRange(-1*math.Pi, math.Pi),
	}

	randWeights_3 := []float64{
		ng.RandomInRange(-1*math.Pi, math.Pi),
	}

	randWeights_4 := []float64{
		ng.RandomInRange(-1*math.Pi, math.Pi),
	}

	// create network nodes
	hn1_processor := &ng.Neuron{Bias: randBias_1, ActivationFunction: ng.Sigmoid}
	hidden_neuron1 := &ng.Node{Name: "hidden_neuron1"}
	hidden_neuron1.SetProcessor(hn1_processor)

	hn2_processor := &ng.Neuron{Bias: randBias_2, ActivationFunction: ng.Sigmoid}
	hidden_neuron2 := &ng.Node{Name: "hidden_neuron2"}
	hidden_neuron2.SetProcessor(hn2_processor)

	outn_processor := &ng.Neuron{Bias: randBias_3, ActivationFunction: ng.Sigmoid}
	output_neuron := &ng.Node{Name: "output_neuron"}
	output_neuron.SetProcessor(outn_processor)

	sensor := &ng.Node{Name: "sensor"}
	sensor.SetProcessor(&ng.Sensor{})
	actuator := &ng.Node{Name: "actuator"}
	actuator.SetProcessor(&ng.Actuator{})

	// connect nodes together
	sensor.ConnectBidirectionalWeighted(hidden_neuron1, randWeights_1)
	sensor.ConnectBidirectionalWeighted(hidden_neuron2, randWeights_2)
	hidden_neuron1.ConnectBidirectionalWeighted(output_neuron, randWeights_3)
	hidden_neuron2.ConnectBidirectionalWeighted(output_neuron, randWeights_4)
	output_neuron.ConnectBidirectional(actuator)

	// create neural network
	sensors := []*ng.Node{sensor}
	actuators := []*ng.Node{actuator}
	neuralNet := &ng.NeuralNetwork{}
	neuralNet.SetSensors(sensors)
	neuralNet.SetActuators(actuators)

	return neuralNet

}

func TestWeightTraining(t *testing.T) {

	// training set
	examples := []*ng.TrainingSample{
		// TODO: how to wrap this?
		{SampleInputs: [][]float64{[]float64{0, 1}}, ExpectedOutputs: [][]float64{[]float64{0}}},
		{SampleInputs: [][]float64{[]float64{1, 1}}, ExpectedOutputs: [][]float64{[]float64{1}}},
		{SampleInputs: [][]float64{[]float64{1, 0}}, ExpectedOutputs: [][]float64{[]float64{0}}},
		{SampleInputs: [][]float64{[]float64{0, 0}}, ExpectedOutputs: [][]float64{[]float64{1}}}}

	// create netwwork with topology capable of solving XNOR
	neuralNet := xnorNetworkUntrained()

	// verify it can not yet solve the training set (since training would be useless in that case)
	verified := neuralNet.Verify(examples)
	assert.False(t, verified)

	shc := new(StochasticHillClimber)
	neuralNetTrained := shc.Train(neuralNet, examples)
	// verify it can now solve the training set
	verified = neuralNetTrained.Verify(examples)
	assert.True(t, verified)

}

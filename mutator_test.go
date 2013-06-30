package neurvolve

import (
	"github.com/couchbaselabs/go.assert"
	ng "github.com/tleyden/neurgo"
	"math"
	"testing"
)

func BasicNetwork() *ng.NeuralNetwork {

	neuron_processor := &ng.Neuron{Bias: -10, ActivationFunction: ng.Sigmoid}
	neuron := &ng.Node{Name: "neuron"}
	neuron.SetProcessor(neuron_processor)

	sensor := &ng.Node{Name: "sensor"}
	sensor.SetProcessor(&ng.Sensor{})

	actuator := &ng.Node{Name: "actuator"}
	actuator.SetProcessor(&ng.Actuator{})

	randWeights := []float64{
		ng.RandomInRange(-1*math.Pi, math.Pi),
		ng.RandomInRange(-1*math.Pi, math.Pi),
	}

	// connect nodes together
	sensor.ConnectBidirectionalWeighted(neuron, randWeights)
	neuron.ConnectBidirectional(actuator)

	// create neural network
	sensors := []*ng.Node{sensor}
	actuators := []*ng.Node{actuator}
	neuralNet := &ng.NeuralNetwork{}
	neuralNet.SetSensors(sensors)
	neuralNet.SetActuators(actuators)

	return neuralNet
}

func TestAddNeuron(t *testing.T) {
	neuralNet := BasicNetwork()
	nnCopy := AddNeuron(neuralNet)
	assert.Equals(t, len(neuralNet.Neurons())+1, len(nnCopy.Neurons()))
}

func TestAddInlink(t *testing.T) {
	assert.True(t, true)
}

func TestAddOutlink(t *testing.T) {
	assert.True(t, true)
}

func TestOutsplice(t *testing.T) {
	assert.True(t, true)
}

func TestInsplice(t *testing.T) {
	assert.True(t, true)
}

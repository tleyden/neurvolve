package neurvolve

import (
	ng "github.com/tleyden/neurgo"
	"math"
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

// create netwwork with topology capable of solving XNOR, but which
// has not been trained yet
func XnorNetworkUntrained() *ng.NeuralNetwork {

	randBias_1 := ng.RandomInRange(-1*math.Pi, math.Pi)
	randBias_2 := ng.RandomInRange(-1*math.Pi, math.Pi)
	randBias_3 := ng.RandomInRange(-1*math.Pi, math.Pi)

	// TODO: use util.randomWeights
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

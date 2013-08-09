package main

import (
	"fmt"
	ng "github.com/tleyden/neurgo"
	nv "github.com/tleyden/neurvolve"
)

func main() {
	fmt.Println("RunWeightTraining")
	RunWeightTraining()
}

func RunWeightTraining() {

	ng.SeedRandom()

	// training set -- todo: examples := ng.XnorTrainingSamples()
	examples := ng.XnorTrainingSamples()

	// create netwwork with topology capable of solving XNOR
	cortex := XnorCortexUntrained()

	// verify it can not yet solve the training set (since training would be useless in that case)
	verified := cortex.Verify(examples)
	if verified {
		panic("neural net already trained, nothing to do")
	}

	shc := &nv.StochasticHillClimber{
		FitnessThreshold:           ng.FITNESS_THRESHOLD,
		MaxIterationsBeforeRestart: 100000,
		MaxAttempts:                4000000,
	}
	cortexTrained, succeeded := shc.Train(cortex, examples)
	if !succeeded {
		panic("could not train neural net")
	}

	// verify it can now solve the training set
	verified = cortexTrained.Verify(examples)
	if !verified {
		panic("could not verify neural net")
	}

}

func XnorCortexUntrained() *ng.Cortex {

	sensorNodeId := ng.NewSensorId("sensor", 0.0)
	hiddenNeuron1NodeId := ng.NewNeuronId("hidden-neuron1", 0.25)
	hiddenNeuron2NodeId := ng.NewNeuronId("hidden-neuron2", 0.25)
	outputNeuronNodeIde := ng.NewNeuronId("output-neuron", 0.35)

	actuatorNodeId := ng.NewActuatorId("actuator", 0.5)

	hiddenNeuron1 := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             hiddenNeuron1NodeId,
		Bias:               nv.RandomBias(),
	}
	hiddenNeuron1.Init()

	hiddenNeuron2 := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             hiddenNeuron2NodeId,
		Bias:               nv.RandomBias(),
	}
	hiddenNeuron2.Init()

	outputNeuron := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             outputNeuronNodeIde,
		Bias:               nv.RandomBias(),
	}
	outputNeuron.Init()

	sensor := &ng.Sensor{
		NodeId:       sensorNodeId,
		VectorLength: 2,
	}
	sensor.Init()

	actuator := &ng.Actuator{
		NodeId:       actuatorNodeId,
		VectorLength: 1,
	}
	actuator.Init()

	sensor.ConnectOutbound(hiddenNeuron1)
	hiddenNeuron1.ConnectInboundWeighted(sensor, []float64{20, 20})

	sensor.ConnectOutbound(hiddenNeuron2)
	hiddenNeuron2.ConnectInboundWeighted(sensor, []float64{-20, -20})

	hiddenNeuron1.ConnectOutbound(outputNeuron)
	outputNeuron.ConnectInboundWeighted(hiddenNeuron1, []float64{20})

	hiddenNeuron2.ConnectOutbound(outputNeuron)
	outputNeuron.ConnectInboundWeighted(hiddenNeuron2, []float64{20})

	outputNeuron.ConnectOutbound(actuator)
	actuator.ConnectInbound(outputNeuron)

	nodeId := ng.NewCortexId("cortex")

	cortex := &ng.Cortex{
		NodeId:    nodeId,
		Sensors:   []*ng.Sensor{sensor},
		Neurons:   []*ng.Neuron{hiddenNeuron1, hiddenNeuron2, outputNeuron},
		Actuators: []*ng.Actuator{actuator},
	}

	return cortex

}

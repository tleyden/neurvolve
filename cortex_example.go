package neurvolve

import (
	ng "github.com/tleyden/neurgo"
)

func BasicCortex() *ng.Cortex {

	shouldReInit := false

	sensor := &ng.Sensor{
		NodeId:       ng.NewSensorId("sensor", 0.0),
		VectorLength: 2,
	}
	sensor.Init(shouldReInit)

	hiddenNeuron1 := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("hidden-neuron1", 0.15),
		Bias:               -30,
	}
	hiddenNeuron1.Init(shouldReInit)

	hiddenNeuron2 := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("hidden-neuron2", 0.25),
		Bias:               10,
	}
	hiddenNeuron2.Init(shouldReInit)

	hiddenNeuron3 := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("hidden-neuron3", 0.35),
		Bias:               10,
	}
	hiddenNeuron3.Init(shouldReInit)

	outputNeuron := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("output-neuron", 0.45),
		Bias:               -10,
	}
	outputNeuron.Init(shouldReInit)

	actuator := &ng.Actuator{
		NodeId:       ng.NewActuatorId("actuator", 0.5),
		VectorLength: 1,
	}
	actuator.Init(shouldReInit)

	sensor.ConnectOutbound(hiddenNeuron1)
	hiddenNeuron1.ConnectInboundWeighted(sensor, []float64{20, 20})

	hiddenNeuron1.ConnectOutbound(hiddenNeuron2)
	hiddenNeuron2.ConnectInboundWeighted(hiddenNeuron1, []float64{1})

	hiddenNeuron2.ConnectOutbound(hiddenNeuron3)
	hiddenNeuron3.ConnectInboundWeighted(hiddenNeuron2, []float64{1})

	hiddenNeuron3.ConnectOutbound(outputNeuron)
	outputNeuron.ConnectInboundWeighted(hiddenNeuron3, []float64{1})

	outputNeuron.ConnectOutbound(actuator)
	actuator.ConnectInbound(outputNeuron)

	nodeId := ng.NewCortexId("test-cortex")

	cortex := &ng.Cortex{
		NodeId: nodeId,
	}
	cortex.SetSensors([]*ng.Sensor{sensor})
	cortex.SetNeurons([]*ng.Neuron{hiddenNeuron1, hiddenNeuron2, hiddenNeuron3, outputNeuron})
	cortex.SetActuators([]*ng.Actuator{actuator})

	return cortex

}

func BasicCortexRecurrent() *ng.Cortex {

	shouldReInit := false

	sensor := &ng.Sensor{
		NodeId:       ng.NewSensorId("sensor", 0.0),
		VectorLength: 2,
	}
	sensor.Init(shouldReInit)

	hiddenNeuron1 := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("hidden-neuron1", 0.15),
		Bias:               -30,
	}
	hiddenNeuron1.Init(shouldReInit)

	hiddenNeuron2 := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("hidden-neuron2", 0.25),
		Bias:               10,
	}
	hiddenNeuron2.Init(shouldReInit)

	hiddenNeuron3 := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("hidden-neuron3", 0.35),
		Bias:               10,
	}
	hiddenNeuron3.Init(shouldReInit)

	outputNeuron := &ng.Neuron{
		ActivationFunction: ng.EncodableSigmoid(),
		NodeId:             ng.NewNeuronId("output-neuron", 0.45),
		Bias:               -10,
	}
	outputNeuron.Init(shouldReInit)

	actuator := &ng.Actuator{
		NodeId:       ng.NewActuatorId("actuator", 0.5),
		VectorLength: 1,
	}
	actuator.Init(shouldReInit)

	sensor.ConnectOutbound(hiddenNeuron1)
	hiddenNeuron1.ConnectInboundWeighted(sensor, []float64{20, 20})

	hiddenNeuron1.ConnectOutbound(hiddenNeuron2)
	hiddenNeuron2.ConnectInboundWeighted(hiddenNeuron1, []float64{1})

	// jumps over 2nd hidden layer, direct from 1st -> 3rd
	hiddenNeuron1.ConnectOutbound(hiddenNeuron3)
	hiddenNeuron3.ConnectInboundWeighted(hiddenNeuron1, []float64{1})

	hiddenNeuron2.ConnectOutbound(hiddenNeuron3)
	hiddenNeuron3.ConnectInboundWeighted(hiddenNeuron2, []float64{1})

	// recurrent connection
	hiddenNeuron3.ConnectOutbound(hiddenNeuron1)
	hiddenNeuron1.ConnectInboundWeighted(hiddenNeuron3, []float64{1})

	hiddenNeuron3.ConnectOutbound(outputNeuron)
	outputNeuron.ConnectInboundWeighted(hiddenNeuron3, []float64{1})

	outputNeuron.ConnectOutbound(actuator)
	actuator.ConnectInbound(outputNeuron)

	nodeId := ng.NewCortexId("test-cortex")

	cortex := &ng.Cortex{
		NodeId: nodeId,
	}
	cortex.SetSensors([]*ng.Sensor{sensor})
	cortex.SetNeurons([]*ng.Neuron{hiddenNeuron1, hiddenNeuron2, hiddenNeuron3, outputNeuron})
	cortex.SetActuators([]*ng.Actuator{actuator})

	return cortex

}

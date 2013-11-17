package neurvolve

import (
	ng "github.com/tleyden/neurgo"
)

type CortexPopulation []*ng.Cortex

func (population CortexPopulation) Find(uuid string) *ng.Cortex {
	for _, cortex := range population {
		if cortex.NodeId.UUID == uuid {
			return cortex
		}
	}
	return nil
}

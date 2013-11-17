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

func (population CortexPopulation) Uuids() []string {
	uuids := make([]string, 0)
	for _, cortex := range population {
		uuids = append(uuids, cortex.NodeId.UUID)
	}
	return uuids
}

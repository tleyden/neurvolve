package neurvolve

import (
	"fmt"
	ng "github.com/tleyden/neurgo"
)

// A Cortex along with other data
type EvaluatedCortex struct {
	Cortex              *ng.Cortex
	Fitness             float64
	ParentId            string
	CreatedInGeneration int
}

type EvaluatedCortexes []EvaluatedCortex

func (fca EvaluatedCortexes) Len() int {
	return len(fca)
}

func (fca EvaluatedCortexes) Less(i, j int) bool {
	return fca[i].Fitness > fca[j].Fitness
}

func (fca EvaluatedCortexes) Swap(i, j int) {
	fca[i], fca[j] = fca[j], fca[i]
}

func (evaldCortexes EvaluatedCortexes) Uuids() [][]string {
	uuids := make([][]string, 0)
	for _, evaldCortex := range evaldCortexes {
		fitnessStr := fmt.Sprintf("%v", evaldCortex.Fitness)
		uuidFitness := []string{evaldCortex.Cortex.NodeId.UUID, fitnessStr}
		uuids = append(uuids, uuidFitness)
	}
	return uuids
}

func (evaldCortexes EvaluatedCortexes) Find(uuid string) EvaluatedCortex {
	for _, evaldCortex := range evaldCortexes {
		if evaldCortex.Cortex.NodeId.UUID == uuid {
			return evaldCortex
		}
	}
	return EvaluatedCortex{}
}

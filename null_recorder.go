package neurvolve

import (
	ng "github.com/tleyden/neurgo"
)

type NullRecorder struct {
}

func NewNullRecorder() *NullRecorder {
	return &NullRecorder{}
}

func (r NullRecorder) AddGeneration(evaldCortexes []EvaluatedCortex) {

}

func (r NullRecorder) AddFitnessScore(score float64, cortex *ng.Cortex, opponent *ng.Cortex) {

}

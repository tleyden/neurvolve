package neurvolve

import (
	"encoding/json"
	"fmt"
	"github.com/couchbaselabs/go.assert"
	ng "github.com/tleyden/neurgo"
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

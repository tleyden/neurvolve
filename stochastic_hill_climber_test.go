package neurvolve

import (
	"encoding/json"
	"fmt"
	"github.com/couchbaselabs/go.assert"
	ng "github.com/tleyden/neurgo"
	"log"
	"testing"
)

func TestPerturbParameters(t *testing.T) {

	cortex := ng.XnorCortex()

	nnJson, _ := json.Marshal(cortex)
	nnJsonString := fmt.Sprintf("%s", nnJson)
	log.Printf("before: %s", nnJsonString)

	shc := new(StochasticHillClimber)

	shc.perturbParameters(cortex)

	nnJsonAfter, _ := json.Marshal(cortex)
	nnJsonStringAfter := fmt.Sprintf("%s", nnJsonAfter)
	log.Printf("before: %s", nnJsonStringAfter)

	// the json should be different after we perturb it
	assert.NotEquals(t, nnJsonString, nnJsonStringAfter)

}

package neurvolve

import (
	"github.com/couchbaselabs/go.assert"
	ng "github.com/tleyden/neurgo"
	"testing"
)

func TestChooseRandomOpponents(t *testing.T) {

	pt := &PopulationTrainer{}

	cortex := BasicCortex()
	opponent := BasicCortex()
	population := []*ng.Cortex{cortex, opponent}

	opponents := pt.chooseRandomOpponents(cortex, population, 1)
	assert.Equals(t, len(opponents), 1)
	assert.Equals(t, opponents[0], opponent)

}

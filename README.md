
# Neurvolve

[![Build Status](https://drone.io/github.com/tleyden/neurvolve/status.png)](https://drone.io/github.com/tleyden/neurvolve/latest)

Evolution-based training for [neurgo](https://github.com/tleyden/neurgo), a neural network framework in Go.

# Status

* Learning via Stochastic Hill Climbing works 
* Topological mutatation operators implemented
* Example which evolves a network capable of solving XOR 

# Running Examples

```
$ cd examples
$ go build -v && go run run_examples.go run_stochastic_hill_climber.go
```

# Related Work

[DXNN2](https://github.com/CorticalComputer/DXNN2) - Pure Erlang TPEULN (Topology & Parameter Evolving Universal Learning Network).  


# Related Publications

[Handbook of Neuroevolution Through Erlang](http://www.amazon.com/Handbook-Neuroevolution-Through-Erlang-Gene/dp/1461444624) _by Gene Sher_.
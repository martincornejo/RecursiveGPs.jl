# RecursiveGPs.jl

RecursiveGPs.jl implements recursive Gaussian process (RGP) regression
[(Huber, 2014)](https://doi.org/10.1016/j.patrec.2014.03.004) for learning unknown
functions online. The package depends on
[AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) for kernel
definitions and
[LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl)
for the Kalman Filter backend.

An RGP approximates a Gaussian process by its values at a fixed set of basis points.
These values form the state of a Kalman filter, which is updated with each
observation at a constant cost. Past observations are not stored, and the posterior
mean and variance of the function are available at every step.

The GP state can be augmented with the states of a physical model. An extended Kalman
filter then estimates the model states and an unknown function in the model, for
example a friction law or the open-circuit voltage curve of a battery, from the same
measurements.

## Installation

To install RecursiveGPs.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("RecursiveGPs")
```

## Contents

| Section | Description |
|---------|-------------|
| [Getting Started](@ref) | A first example with figure, step by step |
| [Mathematical Background](@ref) | GP regression, the recursive GP, and coupling to physical models |
| [Tutorials](@ref "Multi-Component RGPs") | Worked examples with executed code and figures |
| [API Reference](@ref) | Docstrings of all exported functions |

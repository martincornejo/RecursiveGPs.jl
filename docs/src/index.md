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

RecursiveGPs.jl is not yet registered. Install it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/martincornejo/RecursiveGPs.jl")
```

## Example

Learn a function from noisy samples, one sample at a time:

```julia
using RecursiveGPs, AbstractGPs, StaticArrays

# Noisy samples of a function to learn
f(u) = 0.5u + 0.1 * sinpi(2u)
us = 0.1 .+ 0.7 .* rand(100)
ys = [SA[f(u) + 0.005 * randn()] for u in us]

# GP prior, represented at 21 basis points
rgp = RGP(0.1 * with_lengthscale(SEKernel(), 0.4), collect(range(0, 1, length = 21)))

# Kalman filter whose state is the GP at the basis points
kf = ExtendedKalmanFilter(rgp; σn = 0.005)

# Learn online
for (u, y) in zip(us, ys)
    kf(u, y)
end

# Posterior mean and covariance of f
post = predict_gp(kf, range(0, 1, length = 200))
```

Models with several GPs or with physical states use the multi-component constructor,
`ExtendedKalmanFilter(components, dynamics, measurement, R2)`, with arbitrary
dynamics and measurement functions.

## Contents

| Section | Description |
|---------|-------------|
| [Getting Started](@ref) | Installation and a first example with figure, step by step |
| [Mathematical Background](@ref) | GP regression, the recursive GP, and coupling to physical models |
| [Tutorials](@ref "Multi-Component RGPs") | Worked examples with executed code and figures |
| [API Reference](@ref) | Docstrings of all exported functions |

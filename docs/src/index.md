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
rgp = RGP(0.01 * with_lengthscale(SEKernel(), 0.3), collect(range(0, 1, length = 21)))

# Kalman filter whose state is the GP at the basis points
dynamics(x, u, p, t) = x
measurement(x, u, p, t) = SA[measurement_gp(p.f, x, u)]
R2(x, u, p, t) = @SMatrix [uncertainty_gp(p.f, u) + 0.005^2]
kf = ExtendedKalmanFilter((; f = rgp), dynamics, measurement, R2)

# Learn online
for (u, y) in zip(us, ys)
    kf(u, y)
end

# Posterior mean and covariance of f
post = predict_gp(kf, range(0, 1, length = 200), :f)
```

`R2` is the sum of the residual variance of the GP between basis points and the
sensor noise variance. The same constructor accepts further components, such as
physical states, together with arbitrary dynamics and measurement functions.

## Contents

| Section | Description |
|---------|-------------|
| [Getting Started](@ref) | Installation and a first example, step by step |
| [Mathematical Background](@ref) | GP regression, the recursive GP, and coupling to physical models |
| [Tutorials](@ref "Basic RGP with Kalman Filter") | Worked examples with executed code and figures |
| [API Reference](@ref) | Docstrings of all exported functions |

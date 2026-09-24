# RecursiveGPs.jl

RecursiveGPs.jl learns unknown functions online with Gaussian processes, inside a
Kalman filter. It implements the recursive Gaussian process (RGP) of
[Huber (2014)](https://doi.org/10.1016/j.patrec.2014.03.004) on top of the filters in
[LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl),
with kernels from [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl).

An RGP represents a function by its values at a fixed set of basis points. These
values are the state of a Kalman filter, so every new sample updates the function
estimate at a constant cost, without storing past data. The result is a Gaussian
posterior over the function: its mean is the estimate, and its variance shows where
the data constrains the function and where it does not.

Because the function is an ordinary filter state, it can be combined with the
physical states of a model. A single extended Kalman filter then estimates the states
and learns an unknown term of the model, such as a friction law or a battery's
open-circuit voltage curve, from the same measurements.

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

`R2` combines the GP's residual variance between basis points with the sensor noise.
The same constructor accepts physical states next to the GP, with arbitrary dynamics
and measurement functions.

## Contents

| Section | Description |
|---------|-------------|
| [Getting Started](@ref) | Installation and a first example, step by step |
| [Mathematical Background](@ref) | GP regression, the recursive GP, and coupling to physical models |
| [Tutorials](@ref "Basic RGP with Kalman Filter") | Worked examples with executed code and figures |
| [API Reference](@ref) | Docstrings of all exported functions |

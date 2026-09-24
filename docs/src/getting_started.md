# Getting Started

## Installation

RecursiveGPs.jl is not yet registered in the General registry. Install it from
GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/martincornejo/RecursiveGPs.jl")
```

The package depends on [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl)
for kernel definitions and
[LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl)
for the Kalman Filter backend.

## A first example

The following example covers the basic workflow of the package. A GP prior is placed
on a set of basis points and wrapped in a Kalman filter. The filter learns an unknown
scalar function from noisy samples, one sample at a time, and the learned function
is then evaluated at arbitrary inputs together with its uncertainty.

```julia
using RecursiveGPs     # RGP, ExtendedKalmanFilter, measurement_gp, uncertainty_gp, predict_gp
using AbstractGPs      # kernels
using StaticArrays     # static vectors and matrices for the filter
using LinearAlgebra    # diag
```

### 1. Data

100 samples of ``f(u) = 0.5u + 0.1\sin(2\pi u)`` on ``[0.1, 0.8]``, with sensor noise
of standard deviation 0.005:

```julia
f(u) = 0.5u + 0.1 * sinpi(2u)
σn = 0.005

us = 0.1 .+ 0.7 .* rand(100)
ys = [SA[f(u) + σn * randn()] for u in us]
```

### 2. GP prior

An [`RGP`](@ref) is defined by a kernel and a set of basis points that covers the
input range. Here the kernel is a squared exponential with variance 0.01 and length
scale 0.3, and the basis has 21 points on ``[0, 1]``:

```julia
kernel = 0.01 * with_lengthscale(SEKernel(), 0.3)
b0 = collect(range(0, 1, length = 21))

rgp = RGP(kernel, b0)
```

A mean function can be passed as the first argument, for example
`RGP(u -> 0.5u, kernel, b0)`.

### 3. Filter

The filter state is the vector of GP values at the basis points, initialised with the
prior mean and covariance. The function is constant in time, so the dynamics are the
identity. The measurement is the GP mean at the input, and the measurement noise is
the residual variance of the GP plus the sensor noise variance:

```julia
dynamics(x, u, p, t) = x
measurement(x, u, p, t) = SA[measurement_gp(p.f, x, u)]
R2(x, u, p, t) = @SMatrix [uncertainty_gp(p.f, u) + σn^2]

kf = ExtendedKalmanFilter((; f = rgp), dynamics, measurement, R2)
```

The named tuple `(; f = rgp)` names the component `f`. The component is available as
`p.f` inside the model functions and selects it in [`predict_gp`](@ref).

### 4. Learning

Each call `kf(u, y)` runs one prediction and one correction step:

```julia
for (u, y) in zip(us, ys)
    kf(u, y)
end
```

### 5. Prediction

[`predict_gp`](@ref) returns the posterior mean and covariance of the function at any
query points:

```julia
b = range(0, 1, length = 200)
post = predict_gp(kf, b, :f)

μ = post.μ               # posterior mean
σ = sqrt.(diag(post.Σ))  # posterior standard deviation
```

The standard deviation is small within ``[0.1, 0.8]`` and grows towards the ends of
the basis. [`predict_kf`](@ref) returns the predicted measurement at a single input,
including the measurement noise.

## Next steps

- [Tutorials](@ref "Basic RGP with Kalman Filter"): figures, several GPs in one
  model, hyperparameter tuning, and an RGP inside a physical model.
- [Mathematical Background](@ref): derivation of the recursive GP.
- [API Reference](@ref): all exported functions.

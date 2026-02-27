# Getting Started

## Installation

RecursiveGPs.jl is a registered Julia package. Install it from the Julia REPL:

```julia
using Pkg
Pkg.add("RecursiveGPs")
```

The package depends on [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl)
for kernel definitions and
[LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl)
for the Kalman Filter backend.

---

## Required Imports

```julia
using RecursiveGPs       # RGP, ExtendedKalmanFilter, predict_gp, predict_kf
using AbstractGPs        # GP, SEKernel, with_lengthscale, …
using StaticArrays       # SA[y] for static observation vectors
using LinearAlgebra      # diag, etc.
```

---

## Minimal Working Example

This section walks through fitting a single RGP to noisy scalar observations.

### 1 — Generate Data

```julia
f(b) = 0.5 * b + 0.1 * sinpi(b * 2)   # ground-truth function

n  = 100
us = 0.1 .+ rand(n) / 1.5             # inputs  ∈ [0.1, 0.77]
ys = [SA[f(u) + 5e-3 * randn()] for u in us]  # noisy scalar observations
```

### 2 — Define the Model

An [`RGP`](@ref) is parameterized by:
- a **kernel** (here a scaled squared-exponential),
- a set of **basis points** `b0` covering the input domain.

```julia
kernel = 0.01 * with_lengthscale(SEKernel(), 0.3)
b0     = collect(0:0.05:1)   # 21 evenly-spaced basis points

rgp = RGP(kernel, b0)
```

Optionally, you can supply a mean function:

```julia
m(x)   = 0.1 + 0.5x
rgp_m  = RGP(m, kernel, b0)
```

### 3 — Wrap in an Extended Kalman Filter

```julia
kf = ExtendedKalmanFilter(rgp)
```

The state of the KF is the vector of GP function values at `b0`, initialized
with the prior mean and covariance of the GP.

### 4 — Train Online

Call the filter for each (input, observation) pair:

```julia
for (u, y) in zip(us, ys)
    kf(u, y)
end
```

Each call runs one predict-correct cycle, updating the posterior distribution
over GP values at the basis points.

### 5 — Predict

After training, query the fitted function at any set of test points:

```julia
b_test = collect(range(0.0, 1.0, length = 200))
pred   = predict_gp(kf, b_test)

μ = pred.μ                        # posterior mean vector (length 200)
σ = sqrt.(diag(pred.Σ))           # posterior std deviation
```

To get the predicted *measurement* at a single input (including measurement
noise modelled by the GP uncertainty):

```julia
y_hat = predict_kf(kf, 0.5)      # returns (; μ, Σ)
```

---

## Key Objects

| Symbol | Type | Description |
|--------|------|-------------|
| `rgp` | [`RGP`](@ref) | Recursive GP model: kernel + basis points + precomputed matrices |
| `kf` | `ExtendedKalmanFilter` | KF whose state = GP values at `b0` |
| `predict_gp` | function | Project posterior onto arbitrary query points |
| `predict_kf` | function | Predicted measurement + innovation covariance |

---

## Next Steps

- See the [Tutorials](@ref "Basic RGP with Kalman Filter") for plots and
  multi-component models.
- See [Mathematical Background](@ref) for the equations behind the scenes.
- See [API Reference](@ref) for the full function signatures.
- For **hyperparameter tuning**, see the
  [Hyperparameter Tuning](@ref "Hyperparameter Tuning") tutorial.

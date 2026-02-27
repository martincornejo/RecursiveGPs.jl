# RecursiveGPs.jl

**RecursiveGPs.jl** implements Recursive Gaussian Process (RGP) regression
([Huber 2013](https://doi.org/10.1109/ICASSP.2013.6638281),
[Huber 2014](https://doi.org/10.1016/j.patrec.2014.03.004)) with seamless
integration into [LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl)'s
`ExtendedKalmanFilter`.

A Recursive GP represents a function as a Gaussian distribution over values at
a fixed set of *basis points*. This distribution acts as the Kalman Filter
state, enabling online, sequential updates as new observations arrive — with no
matrix inversion at inference time.

---

## Installation

```julia
using Pkg
Pkg.add("RecursiveGPs")
```

Or from the Julia REPL:

```
] add RecursiveGPs
```

---

## Quick Start

```julia
using RecursiveGPs, AbstractGPs, StaticArrays

# 1. Define a kernel and a set of basis points
kernel = 0.01 * with_lengthscale(SEKernel(), 0.3)
b0     = collect(0:0.05:1)

# 2. Build an RGP and wrap it in a Kalman Filter
rgp = RGP(kernel, b0)
kf  = ExtendedKalmanFilter(rgp)

# 3. Train online (one observation at a time)
for (u, y) in zip(inputs, outputs)
    kf(u, SA[y])
end

# 4. Predict
b_test = collect(range(0.0, 1.0, 200))
pred   = predict_gp(kf, b_test)   # returns (; μ, Σ)
```

---

## Documentation Overview

| Section | Description |
|---------|-------------|
| [Getting Started](@ref) | Step-by-step installation and minimal working example |
| [Mathematical Background](@ref) | RGP theory, key equations, state-space interpretation |
| [Tutorials](@ref "Basic RGP with Kalman Filter") | Worked examples with full code |
| [API Reference](@ref) | Complete docstrings for all exported symbols |

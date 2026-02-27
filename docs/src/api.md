# API Reference

## Type

```@docs
RGP
```

---

## Constructors

### `RGP`

There are three constructor overloads (docstrings are attached to each method):

| Signature | Description |
|-----------|-------------|
| `RGP(gp, b0)` | Build from a full `AbstractGPs.GP` object |
| `RGP(kernel, b0)` | Build from a kernel (zero mean prior) |
| `RGP(mean, kernel, b0)` | Build from a mean function and kernel |

All overloads accept an optional positional argument `cov_jitter = 1e-6` for
numerical stability when inverting ``\Sigma_0``.

### `ExtendedKalmanFilter`

```@docs
ExtendedKalmanFilter(::RecursiveGPs.RGP)
ExtendedKalmanFilter(::NamedTuple, ::Any, ::Function, ::Function)
```

---

## Inference and Prediction

```@docs
measurement_gp
uncertainty_gp
predict_gp
predict_kf
```

---

## State Accessors (multi-component models)

These methods extend the `LowLevelParticleFilters` accessors with a
component-index overload. They are available after constructing a filter with
the multi-component `ExtendedKalmanFilter(components, ...)` constructor.

```@docs
LowLevelParticleFilters.state(::Any, ::Symbol)
LowLevelParticleFilters.covariance(::Any, ::Symbol)
```

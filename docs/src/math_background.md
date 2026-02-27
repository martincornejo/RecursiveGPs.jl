# Mathematical Background

## Gaussian Process Regression (brief review)

A **Gaussian Process** (GP) ``f \sim \mathcal{GP}(m, k)`` is a distribution
over functions: any finite collection of function values is jointly Gaussian.
Given a prior mean ``m(\cdot)`` and covariance kernel ``k(\cdot, \cdot)``, the
posterior conditioned on noisy observations ``\{(u_i, y_i)\}`` is also a GP,
with analytically computable mean and covariance.

Standard batch GP inference requires inverting the ``n \times n`` Gram matrix
of the training set, which is ``\mathcal{O}(n^3)`` — prohibitive for large
``n`` or online settings.

---

## Recursive GP Representation

The **Recursive GP** (Huber 2013, 2014) sidesteps the growing matrix inversion
by fixing a set of ``N`` **basis points** ``b_0 \in \mathbb{R}^N`` and
maintaining a Gaussian distribution over function values *at those points*:

```math
g = f(b_0) \sim \mathcal{N}(\mu_0,\, \Sigma_0)
```

where ``\mu_0 = m(b_0)`` and ``\Sigma_0 = k(b_0, b_0)``.

The *state* ``g \in \mathbb{R}^N`` is updated by a **Kalman Filter** each time
a new observation arrives. No new basis points are added; the cost per update is
``\mathcal{O}(N^2)``.

### Conditional Mean (measurement model)

Given the current state ``g`` (the GP values at ``b_0``), the predicted
function value at a new query point ``b`` is:

```math
\mu_\text{post}(b) = m(b) + k(b, b_0)\,\Sigma_0^{-1}\,(g - \mu_0)
```

This is the **measurement function** ``h(g, b)`` used by the EKF.

### Conditional Variance (measurement noise)

The uncertainty at ``b`` conditioned on the basis is:

```math
\sigma^2_\text{post}(b) = k(b, b) - k(b, b_0)\,\Sigma_0^{-1}\,k(b_0, b)
```

This is the **observation noise covariance** ``R_2`` seen by the EKF.

### KF State Covariance Projection

After ``n`` updates, the KF state covariance ``R^-`` captures the posterior
uncertainty over ``g``. The uncertainty at a vector of query points ``b`` is:

```math
\Sigma = R_2(b) + H\, R^-\, H^\top, \qquad H = k(b, b_0)\,\Sigma_0^{-1}
```

where ``R_2(b)`` is the diagonal GP conditional variance and ``H`` is the
projection gain matrix.

---

## State-Space Interpretation

The RGP maps directly onto a linear Gaussian state-space model:

| KF component | RGP meaning |
|---|---|
| State ``x`` | GP values at basis points ``g = f(b_0)`` |
| Dynamics ``f(x,u)`` | Identity (GP prior is stationary in index space) |
| Measurement ``h(x,u)`` | Conditional mean ``\mu_\text{post}(b)`` |
| Process noise ``R_1`` | Zero (prior is fixed) |
| Measurement noise ``R_2`` | Conditional variance ``\sigma^2_\text{post}(b)`` |
| Initial distribution | ``\mathcal{N}(\mu_0, \Sigma_0)`` |

At each step the KF prediction pass is a no-op (identity dynamics), and the
correction pass is a standard EKF update using the linearized measurement
function ``H``.

---

## Multi-Component Block-Diagonal Formulation

When the output depends on several independent GPs — e.g.:

```math
y = f_1(u_1) + u_2\, f_2(u_1)
```

each ``f_i`` is represented by its own RGP with basis ``b_0^{(i)}``. The joint
state is the concatenation of all component states:

```math
x = \begin{bmatrix} g^{(a)} \\ g^{(b)} \end{bmatrix}
```

with a block-diagonal prior covariance and process noise. The measurement
function combines the component predictions:

```math
h(x, u) = \mu_1(u_1;\, g^{(a)}) + u_2\,\mu_2(u_1;\, g^{(b)})
```

The EKF Jacobian ``C = \partial h / \partial x`` is block-sparse, making each
update efficient. Component-wise posteriors are recovered via [`predict_gp`](@ref).

---

## References

1. M. F. Huber, "Recursive Gaussian process regression," *2013 IEEE
   International Conference on Acoustics, Speech and Signal Processing*,
   Vancouver, BC, Canada, 2013, pp. 3362–3366.
   DOI: [10.1109/ICASSP.2013.6638281](https://doi.org/10.1109/ICASSP.2013.6638281)

2. M. F. Huber, "Recursive Gaussian process: On-line regression and learning,"
   *Pattern Recognition Letters*, 2014.
   DOI: [10.1016/j.patrec.2014.03.004](https://doi.org/10.1016/j.patrec.2014.03.004)

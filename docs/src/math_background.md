# Mathematical Background

Physical models often contain functions that are difficult to derive from first
principles, such as a friction law, the open-circuit voltage curve of a battery, or
the efficiency map of a motor. A parametric form for such a function has to be chosen
in advance, and a wrong choice biases the model. This page describes how a Gaussian
process represents such a function instead, how its recursive formulation allows online
learning, and how it is combined with the states of a physical model in a Kalman
filter.

## Gaussian process regression

Gaussian process (GP) regression is a Bayesian, nonparametric machine learning method
[1]. A GP is a probability distribution over functions. It generalises the
multivariate Gaussian distribution, defined by a mean vector and a covariance matrix,
to functions, defined by a mean function and a covariance function. The prior encodes
assumptions about the function, for example its smoothness. Conditioning the prior on
observations yields a posterior distribution over the function, from which an estimate
and its uncertainty follow at any input.

GP regression requires no parametric form of the function, and the posterior variance
quantifies the uncertainty of the estimate. Applications include system
identification, surrogate modelling and Bayesian optimisation.

### The prior

A GP is written

```math
f \sim \mathcal{GP}\big(m(u),\, k(u, u')\big).
```

The mean function ``m(u)`` is the expected value of ``f`` before any observation. It
is commonly zero or a simple prior model. The kernel ``k(u, u')`` is the covariance
between ``f(u)`` and ``f(u')``. It determines which functions are likely under the
prior, and with it how the information of an observation propagates to neighbouring
inputs. A common choice is the squared exponential kernel,

```math
k(u, u') = \sigma^2 \exp\left(-\frac{(u - u')^2}{2\ell^2}\right),
```

with prior variance ``\sigma^2`` and length scale ``\ell``. The correlation between
``f(u)`` and ``f(u')`` is ``e^{-1/2} \approx 0.61`` at ``|u - u'| = \ell`` and
negligible beyond about ``3\ell``. Short length scales permit rapid variation, long
ones enforce smooth functions. Other kernels express other properties, such as
periodicity or lower smoothness, and kernels can be combined by addition and
multiplication.

Formally, a GP is a collection of random variables ``f(u)``, any finite number of
which have a joint Gaussian distribution. For inputs ``U = \{u_1, \ldots, u_n\}``,

```math
f(U) \sim \mathcal{N}\big(m(U),\, K_{UU}\big), \qquad [K_{UU}]_{ij} = k(u_i, u_j).
```

Regression with a GP therefore reduces to conditioning a multivariate Gaussian, which
has a closed-form solution.

### Conditioning on data

The observations are ``y_i = f(u_i) + \epsilon_i`` with sensor noise
``\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)``. The observations ``Y`` and the
function values ``f(U^*)`` at test inputs ``U^*`` are jointly Gaussian,

```math
\begin{bmatrix} Y \\ f(U^*) \end{bmatrix}
\sim \mathcal{N}\left(
\begin{bmatrix} m(U) \\ m(U^*) \end{bmatrix},
\begin{bmatrix} K_{UU} + \sigma_n^2 I & K_{UU^*} \\ K_{U^*U} & K_{U^*U^*} \end{bmatrix}
\right),
```

where ``K_{U^*U}`` contains the kernel evaluations between test and training inputs.
Conditioning on ``Y`` gives the posterior ``f(U^*) \mid Y \sim \mathcal{N}(\mu^*, \Sigma^*)``,

```math
\begin{aligned}
\mu^* &= m(U^*) + K_{U^*U}\,(K_{UU} + \sigma_n^2 I)^{-1}\,(Y - m(U)) \\
\Sigma^* &= K_{U^*U^*} - K_{U^*U}\,(K_{UU} + \sigma_n^2 I)^{-1}\,K_{UU^*}
\end{aligned}
```

The posterior mean adds to the prior mean a correction that is weighted by the
covariance between test and training inputs. Close to the training inputs, the
posterior variance falls below the prior variance. Several length scales away,
``K_{U^*U} \approx 0`` and the posterior reverts to the prior. The noise variance
``\sigma_n^2`` sets the degree of smoothing: for small ``\sigma_n^2`` the posterior
mean interpolates the observations, for large ``\sigma_n^2`` it averages over them.

The hyperparameters ``\sigma^2``, ``\ell`` and ``\sigma_n^2`` are commonly chosen by
maximising the marginal likelihood of the observations.

### Limitations for online use

Inverting the ``n \times n`` matrix costs ``\mathcal{O}(n^3)``, and all observations
must be stored for prediction. At a sampling rate of 1 kHz, one hour of operation
yields ``3.6 \cdot 10^6`` samples, which rules out batch GP regression for online use.

Batch GP regression also requires direct observations of ``f``. In a physical model,
the unknown function is often observed only indirectly. A friction force, for
instance, affects the measured velocity through the equation of motion, and its
argument, the velocity, is itself an estimated state. Learning such a function
requires including it in the state estimation problem.

## The recursive GP

Huber [2] formulated GP regression as a Kalman filtering problem. The function is
represented by a Gaussian state of fixed dimension, which is updated with each
observation. Past observations are not stored.

### Basis points

The recursive GP (RGP) fixes ``N`` basis points ``b_0`` and represents the function
by its values there, ``g = f(b_0)``. Under the GP prior,

```math
g \sim \mathcal{N}(\mu_0, \Sigma_0), \qquad \mu_0 = m(b_0), \quad \Sigma_0 = K_{b_0 b_0} + \varepsilon I,
```

where a small jitter ``\varepsilon`` keeps ``\Sigma_0`` invertible. The dimension of
``g`` is fixed by ``N``. The basis points take the role of training inputs whose
function values are unknown and estimated.

### Function values from the basis values

An observation at an arbitrary input ``u`` is related to ``g`` through the conditional
distribution of ``f(u)`` given ``g``. Since ``g`` and ``f(u)`` are jointly Gaussian
under the prior, ``f(u) \mid g`` is Gaussian with

```math
\begin{aligned}
\mathbb{E}[f(u) \mid g] &= m(u) + H(u)\,(g - \mu_0), \qquad H(u) = K_{u b_0}\,\Sigma_0^{-1} \\
\operatorname{Var}[f(u) \mid g] &= r(u) = k(u, u) - H(u)\,K_{b_0 u}
\end{aligned}
```

This is the GP posterior of the previous section, with the basis points as noise-free
training inputs. ``H(u)`` contains interpolation weights, and the conditional mean is
a weighted sum of the basis values. The RGP is thus a linear model in the ``N``
weights ``g`` with basis functions ``H(u)`` derived from the kernel, which corresponds
to the weight-space view of GP regression [1]. The residual variance ``r(u)`` is the
variance of ``f(u)`` not explained by ``g``. It vanishes at the basis points and
increases between them, and it remains small for a spacing well below ``\ell``.

### Observation model

An observation ``y_t`` at input ``u_t`` is a linear measurement of ``g``,

```math
y_t = m(u_t) + H(u_t)\,(g - \mu_0) + \epsilon_t, \qquad
\epsilon_t \sim \mathcal{N}(0,\, R_2), \qquad R_2 = r(u_t) + \sigma_n^2.
```

The measurement noise ``R_2`` comprises the residual variance of the basis
representation and the sensor noise variance. Omitting either term makes the filter
overconfident.

### Kalman filter recursion

Let ``\hat g_t`` and ``P_t`` denote the mean and covariance of ``g`` after ``t``
observations, with ``\hat g_0 = \mu_0`` and ``P_0 = \Sigma_0``. The function is
modelled as a random walk,

```math
g_t = g_{t-1} + w_t, \qquad w_t \sim \mathcal{N}(0, R_1).
```

For ``R_1 = 0`` the function is constant in time, and the filter performs GP
regression on the basis. For ``R_1 > 0`` the covariance increases between
observations. Older data is discounted, and the filter tracks a time-varying
function, for example friction that changes with temperature or wear.

The correction step for an observation at ``u_t``, with ``H_t = H(u_t)``, is

```math
\begin{aligned}
S_t &= H_t P_{t|t-1} H_t^\top + R_2 \\
K_t &= P_{t|t-1} H_t^\top S_t^{-1} \\
\hat g_t &= \hat g_{t|t-1} + K_t \big(y_t - m(u_t) - H_t(\hat g_{t|t-1} - \mu_0)\big) \\
P_t &= P_{t|t-1} - K_t H_t P_{t|t-1}
\end{aligned}
```

``S_t`` is the variance of the predicted measurement. The gain ``K_t`` is large for
basis values that are strongly correlated with ``f(u_t)`` and still uncertain, and an
observation changes the function mainly in the vicinity of ``u_t``. Since the
measurement is linear in ``g``, the update is exact.

### Prediction at new inputs

The posterior at query points ``b`` follows from ``\hat g_t`` and ``P_t`` through the
same conditional,

```math
\begin{aligned}
\mu^* &= m(b) + H^*(\hat g_t - \mu_0), \qquad H^* = K_{b b_0}\,\Sigma_0^{-1} \\
\Sigma^* &= H^* P_t H^{*\top} + K_{bb} - H^* K_{b_0 b}.
\end{aligned}
```

The first term of ``\Sigma^*`` is the uncertainty of the basis values, the second the
residual between basis points. ``\Sigma^*`` does not include sensor noise and thus
describes ``f``. For ``P_0 = \Sigma_0``, ``\Sigma^*`` equals the prior covariance
``K_{bb}``.

### Approximation and basis choice

The RGP replaces the GP by a finite-dimensional model in which ``f`` is determined by
``g`` up to the residual ``r``. For a basis that is dense relative to ``\ell``, ``r``
vanishes and the RGP coincides with full GP regression. Two requirements on the basis
follow:

- The basis must cover the inputs visited by the data. More than a few length scales
  outside the basis, ``H(u) \approx 0`` and ``f(u)`` remains at the prior regardless
  of the data.
- The spacing must be small compared to ``\ell``. For a coarse spacing, ``r(u)`` is
  large between basis points, and features narrower than the spacing cannot be
  represented.

The number of basis points sets the balance between approximation accuracy and the
dimension of the state.

## Coupling a GP to a physical model

The basis values ``g`` can be stacked with the states ``s`` of a physical model,
``x = [s;\; g]``. A single filter then estimates the states and the unknown function
from the same measurements, for example the velocity of a mass and the friction force
acting on it.

The GP input is often a state itself, such as the velocity ``v`` in a friction model.
The model is then nonlinear in ``x``, and the extended Kalman filter linearises it
around the current estimate. The Jacobian with respect to ``g`` is ``H(v)``, and the
Jacobian with respect to ``v`` contains the derivative of the GP mean.

The residual variance ``r`` is assigned according to where the function enters the
model:

- In the measurement equation, ``r(u)`` is added to ``R_2`` as above.
- In the dynamics, ``r`` is added to the process noise of the state it drives. For
  ``v_{t+1} = v_t + \frac{\Delta t}{m}\big(F_u - F_f(v_t)\big)``, the contribution to
  the process noise of ``v`` is ``(\Delta t / m)^2\, r(v_t)``.

In the second case, ``g`` does not enter any measurement. The dynamics Jacobian
couples ``v`` and ``g`` and creates a cross-covariance between them. A velocity
innovation then corrects ``g`` through this cross-covariance, mainly at the basis
points near the current velocity.

## Multiple GPs in one model

A measurement can depend on several unknown functions. The terminal voltage of a
battery, for example, depends on the open-circuit voltage and on the product of a
resistance and the current, both functions of the state of charge. A generic form is

```math
y = f_a(u_1) + u_2\, f_b(u_1).
```

Each function is represented by its own RGP, and the state contains both sets of
basis values, ``x = [g_a;\; g_b]``. The measurement row is
``[H_a(u_1),\; u_2 H_b(u_1)]``, and the measurement noise is

```math
R_2 = r_a(u_1) + u_2^2\, r_b(u_1) + \sigma_n^2.
```

A single measurement constrains only ``f_a + u_2 f_b``. The prior covariance is block
diagonal, but each update introduces cross-covariance between ``g_a`` and ``g_b``.
The two functions are identifiable only if ``u_2`` varies across measurements with
similar ``u_1``.

## Computational cost

The state has ``N`` entries per GP, independent of the number of observations. The
correction step for a scalar measurement costs ``\mathcal{O}(N^2)``. The prediction
step of an extended Kalman filter with a dense Jacobian ``A`` computes
``A P A^\top`` at ``\mathcal{O}(N^3)``. Processing ``n`` observations thus requires
``\mathcal{O}(n N^3)`` time and ``\mathcal{O}(N^2)`` memory, compared with
``\mathcal{O}(n^3)`` time and ``\mathcal{O}(n^2)`` memory for batch GP regression.
With a constant cost per observation, the filter can run at the sampling rate of the
system.

## Correspondence to the package

| Quantity | Package |
|---|---|
| ``b_0``, ``\mu_0``, ``\Sigma_0``, jitter ``\varepsilon`` | [`RGP`](@ref), fields `b0`, `μ0`, `Σ0`, argument `cov_jitter` (default ``10^{-6}``) |
| ``m(u) + H(u)(g - \mu_0)`` | [`measurement_gp`](@ref) |
| ``r(u)`` | [`uncertainty_gp`](@ref) |
| ``\mu^*``, ``\Sigma^*`` | [`predict_gp`](@ref) |
| Predicted measurement and ``S_t`` | [`predict_kf`](@ref) |
| ``\hat g``, ``P`` | `state(kf)`, `covariance(kf)` (fields `x`, `R` of the filter) |
| ``R_1``, ``R_2`` | `R1`, `R2` of the `ExtendedKalmanFilter` |
| ``\sigma_n`` | keyword `σn` of `ExtendedKalmanFilter(rgp; σn)` |

## References

1. C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine
   Learning*, MIT Press, 2006.
   Available online: [gaussianprocess.org/gpml](https://gaussianprocess.org/gpml/)

2. M. F. Huber, "Recursive Gaussian process: On-line regression and learning,"
   *Pattern Recognition Letters*, vol. 45, pp. 85-91, 2014.
   DOI: [10.1016/j.patrec.2014.03.004](https://doi.org/10.1016/j.patrec.2014.03.004)

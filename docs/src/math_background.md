# Mathematical Background

Many models contain a function that is hard to write down from first principles: a
friction law, the open-circuit voltage curve of a battery, the efficiency map of a
motor. Choosing a parametric form for it means guessing the shape in advance, and a
wrong guess biases everything the model is used for. This page builds up a method
that learns such a function from data while the system runs, reports how certain it
is, and can sit inside a state estimator next to the rest of the model.

## Gaussian process regression

Gaussian process (GP) regression is a Bayesian machine learning method for learning a
function from data [1]. It places a probability distribution over functions: a
Gaussian distribution describes a random vector by a mean vector and a covariance
matrix, and a GP describes a random function by a mean function and a covariance
function. The prior distribution encodes assumptions such as how smooth the function
is. Conditioning it on observations gives the posterior, whose mean is the estimate
of the function and whose variance shows how well the data determines the function
at each input.

Unlike a parametric model, a GP does not fix the form of the function, and it
reports the uncertainty of its estimate. GPs are therefore common where a function
has to be learned from limited data and the reliability of the result matters, for
example in system identification, surrogate modelling, Bayesian optimisation, and
for unknown terms in physical models.

### The prior

A GP is written

```math
f \sim \mathcal{GP}\big(m(u),\, k(u, u')\big).
```

The mean function ``m(u)`` is the expected value of ``f`` before any data is seen. It
is often zero, or a simple model that the data should correct. The kernel
``k(u, u')`` is the covariance between ``f(u)`` and ``f(u')``, and it carries the
modelling assumptions: a large covariance between two inputs means their function
values tend to move together, so the kernel decides how information from one
observation spreads to nearby inputs. A common choice is the squared exponential
kernel,

```math
k(u, u') = \sigma^2 \exp\left(-\frac{(u - u')^2}{2\ell^2}\right).
```

The variance ``\sigma^2`` sets how far ``f`` is expected to deviate from ``m``. The
length scale ``\ell`` sets the distance over which values stay correlated: at
``|u - u'| = \ell`` the correlation is ``e^{-1/2} \approx 0.61``, and beyond about
``3\ell`` it is negligible. A short ``\ell`` allows fast variations, a long ``\ell``
enforces a smooth function. Other kernels encode other assumptions, such as periodic
or less smooth behaviour, and sums and products of kernels combine them.

Formally, a GP is a collection of random variables ``f(u)``, one per input, any
finite number of which are jointly Gaussian. For inputs ``U = \{u_1, \ldots, u_n\}``,

```math
f(U) \sim \mathcal{N}\big(m(U),\, K_{UU}\big), \qquad [K_{UU}]_{ij} = k(u_i, u_j).
```

Learning a function therefore reduces to conditioning a Gaussian vector, which has a
closed-form solution.

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

where ``K_{U^*U}`` holds the kernel evaluations between test and training inputs.
Conditioning on ``Y`` gives the posterior ``f(U^*) \mid Y \sim \mathcal{N}(\mu^*, \Sigma^*)``,

```math
\begin{aligned}
\mu^* &= m(U^*) + K_{U^*U}\,(K_{UU} + \sigma_n^2 I)^{-1}\,(Y - m(U)) \\
\Sigma^* &= K_{U^*U^*} - K_{U^*U}\,(K_{UU} + \sigma_n^2 I)^{-1}\,K_{UU^*}
\end{aligned}
```

The posterior mean is the prior mean plus a correction, weighted by how strongly each
test input correlates with the training inputs. Near the data the correction is large
and the variance drops below the prior. More than a few length scales away,
``K_{U^*U}`` is close to zero, so the mean returns to ``m`` and the variance to
``\sigma^2``. The noise variance ``\sigma_n^2`` decides how closely the mean follows
individual observations: with a small ``\sigma_n^2`` it interpolates them, with a
large one it averages over them.

The hyperparameters ``\sigma^2``, ``\ell`` and ``\sigma_n^2`` are usually chosen by
maximising the likelihood of the observations under this model.

### Limits for online use

The inverse is ``n \times n`` for ``n`` observations, so the cost grows as
``\mathcal{O}(n^3)`` and all past observations have to be stored to predict at new
inputs. A system sampled at 1 kHz produces 3.6 million samples per hour, so batch GP
regression cannot keep up with it.

Batch GP regression also needs direct samples of ``f``. Inside a physical model,
``f`` is usually not measured directly. A friction force, for example, only shows up
through the velocity it causes, and its input, the velocity, is itself an estimated
state. Learning ``f`` in that setting requires treating the function as part of the
state estimation problem.

## The recursive GP

Huber [2] reformulated GP regression as a Kalman filtering problem, which addresses
both limits. The function is summarised by a fixed-size Gaussian state, and each new
observation updates that state and is then discarded.

### Basis points

The recursive GP (RGP) fixes ``N`` basis points ``b_0`` and represents the function
by its values there, ``g = f(b_0)``. Under the GP prior,

```math
g \sim \mathcal{N}(\mu_0, \Sigma_0), \qquad \mu_0 = m(b_0), \quad \Sigma_0 = K_{b_0 b_0} + \varepsilon I,
```

where a small jitter ``\varepsilon`` keeps ``\Sigma_0`` invertible. The size of ``g``
is set by ``N`` and does not grow with the data. The basis points act as fixed
training inputs whose function values are unknown and estimated.

### Function values from the basis values

To learn ``g`` from an observation at an arbitrary input ``u``, the function value at
``u`` has to be expressed through ``g``. Because ``g`` and ``f(u)`` are jointly
Gaussian under the prior, ``f(u)`` given ``g`` is Gaussian with

```math
\begin{aligned}
\mathbb{E}[f(u) \mid g] &= m(u) + H(u)\,(g - \mu_0), \qquad H(u) = K_{u b_0}\,\Sigma_0^{-1} \\
\operatorname{Var}[f(u) \mid g] &= r(u) = k(u, u) - H(u)\,K_{b_0 u}
\end{aligned}
```

This is the GP posterior from above, with the basis points as noise-free training
inputs. ``H(u)`` is a row of interpolation weights: the function at ``u`` is a
weighted sum of the nearby basis values. The RGP is therefore a linear model in the
``N`` weights ``g``, with basis functions ``H(u)`` derived from the kernel, which
corresponds to the weight-space view of GP regression [1]. The residual variance ``r(u)`` is the part
of ``f(u)`` that the basis values cannot explain. It is zero at the basis points,
grows between them, and stays small when the basis spacing is well below the length
scale.

### Observation model

An observation ``y_t`` of ``f`` at input ``u_t`` then becomes a linear measurement of
the state ``g``:

```math
y_t = m(u_t) + H(u_t)\,(g - \mu_0) + \epsilon_t, \qquad
\epsilon_t \sim \mathcal{N}(0,\, R_2), \qquad R_2 = r(u_t) + \sigma_n^2
```

The measurement noise ``R_2`` has two sources: the error of representing ``f`` by its
basis values, and the sensor noise. Leaving out either one makes the filter treat the
data as more precise than it is.

### Kalman filter recursion

Let ``\hat g_t`` and ``P_t`` be the mean and covariance of ``g`` after ``t``
observations, starting from ``\hat g_0 = \mu_0`` and ``P_0 = \Sigma_0``. Between
observations, the function is modelled as a random walk,

```math
g_t = g_{t-1} + w_t, \qquad w_t \sim \mathcal{N}(0, R_1).
```

With ``R_1 = 0`` the function is constant in time, and the recursion performs GP
regression on the basis. A nonzero ``R_1`` lets the function drift. The covariance
then grows between observations, so recent data weighs more than old data. This
tracks a function that changes over time, such as friction that varies with
temperature or wear.

Each observation corrects the state with the Kalman update, using ``H_t = H(u_t)``:

```math
\begin{aligned}
S_t &= H_t P_{t|t-1} H_t^\top + R_2 \\
K_t &= P_{t|t-1} H_t^\top S_t^{-1} \\
\hat g_t &= \hat g_{t|t-1} + K_t \big(y_t - m(u_t) - H_t(\hat g_{t|t-1} - \mu_0)\big) \\
P_t &= P_{t|t-1} - K_t H_t P_{t|t-1}
\end{aligned}
```

``S_t`` is the variance of the predicted measurement. The gain ``K_t`` is large for
basis values that correlate strongly with ``u_t`` and are still uncertain, so each
observation updates the function near its input and leaves distant parts unchanged.
The measurement is linear in ``g``, so the update is exact and involves no
linearisation.

### Prediction at new inputs

The function at any query points ``b`` follows from ``\hat g_t`` and ``P_t``
through the same conditional:

```math
\begin{aligned}
\mu^* &= m(b) + H^*(\hat g_t - \mu_0), \qquad H^* = K_{b b_0}\,\Sigma_0^{-1} \\
\Sigma^* &= H^* P_t H^{*\top} + K_{bb} - H^* K_{b_0 b}
\end{aligned}
```

The first term of ``\Sigma^*`` is the remaining uncertainty in the basis values, the
second the residual between basis points. ``\Sigma^*`` excludes the sensor noise, so
it describes ``f`` itself. Before any data, ``P_0 = \Sigma_0`` and ``\Sigma^*``
reduces to the prior ``K_{bb}``.

### Approximation and basis choice

The RGP replaces the GP by a finite-dimensional model in which ``f`` is determined by
``g`` up to the residual ``r``. It agrees with full GP regression when the basis is
dense relative to ``\ell``, because ``r`` then vanishes. This gives two rules for
choosing the basis:

- The basis has to span the inputs the data visits. A few length scales outside the
  basis range, ``H(u)`` is close to zero, so ``f(u)`` stays at the prior no matter
  how much data arrives there.
- The spacing has to be small compared to ``\ell``. With coarse spacing, ``r(u)``
  between basis points becomes large and features narrower than the spacing cannot
  be represented.

A denser basis improves the approximation and enlarges the state, so the choice
trades accuracy against computation.

## Coupling a GP to a physical model

Because ``g`` is a Gaussian state like any other, it can be stacked with the physical
states ``s`` of a model, ``x = [s;\; g]``. A single filter then estimates the states
and the unknown function together, from the same measurements. A model with known
structure and one unknown term, such as an equation of motion with unknown friction,
is completed this way while the system runs.

The GP input is often itself a state, for example friction evaluated at the estimated
velocity ``v``. The model is then nonlinear in ``x``, and an extended Kalman filter
linearises it at the current estimate. The Jacobian with respect to ``g`` is
``H(v)``, and the Jacobian with respect to ``v`` contains the slope of the GP mean.

The unknown function can enter the model in two places, and ``r`` goes with it:

- In the measurement equation, ``r(u)`` is added to ``R_2`` as above.
- In the dynamics, ``r`` enters the process noise of the state it drives. For
  ``v_{t+1} = v_t + \frac{\Delta t}{m}\big(F_u - F_f(v_t)\big)``, the contribution is
  ``(\Delta t / m)^2\, r(v_t)`` on ``v``.

In the second case no measurement depends on ``g`` directly, and the function is
learned indirectly. The dynamics Jacobian couples ``v`` and ``g`` and builds a
cross-covariance between them. When the measured velocity deviates from its
prediction, the filter corrects ``g`` through that cross-covariance, in the part of
the function evaluated at the current velocity.

## Several GPs in one model

A measurement can depend on more than one unknown function. The terminal voltage of a
battery, for example, depends on the open-circuit voltage and on a resistance times
the current, both functions of the state of charge. In general form,

```math
y = f_a(u_1) + u_2\, f_b(u_1).
```

Each function gets its own RGP, and the state stacks their basis values,
``x = [g_a;\; g_b]``. The measurement row is ``[H_a(u_1),\; u_2 H_b(u_1)]``, and the
residual variances combine as

```math
R_2 = r_a(u_1) + u_2^2\, r_b(u_1) + \sigma_n^2.
```

A single measurement constrains only the combination ``f_a + u_2 f_b``, so one sample
cannot separate the two functions. The prior covariance is block diagonal, and each
update adds cross-covariance between ``g_a`` and ``g_b``. Measurements with different
values of ``u_2`` at the same ``u_1`` separate them, so the data has to vary ``u_2``
for both functions to be identifiable.

## Computational cost

The state has ``N`` entries per GP, independent of the number of observations. The
correction for a scalar measurement costs ``\mathcal{O}(N^2)``. An extended Kalman
filter that propagates the covariance with a dense Jacobian ``A`` forms
``A P A^\top`` at every step, which costs ``\mathcal{O}(N^3)``. Processing ``n``
observations therefore costs ``\mathcal{O}(n N^3)`` time and ``\mathcal{O}(N^2)``
memory, compared to ``\mathcal{O}(n^3)`` time and ``\mathcal{O}(n^2)`` memory for
batch GP regression. The cost per observation is constant, so the filter can run at
the sampling rate of the system.

## Correspondence to the package

| Quantity | Package |
|---|---|
| ``b_0``, ``\mu_0``, ``\Sigma_0``, jitter ``\varepsilon`` | [`RGP`](@ref), fields `b0`, `μ0`, `Σ0`, argument `cov_jitter` (default ``10^{-6}``) |
| ``m(u) + H(u)(g - \mu_0)`` | [`measurement_gp`](@ref) |
| ``r(u)`` | [`uncertainty_gp`](@ref) |
| ``\mu^*``, ``\Sigma^*`` | [`predict_gp`](@ref) |
| Predicted measurement and ``S_t`` | [`predict_kf`](@ref) |
| ``R_1``, ``R_2`` | `R1`, `R2` of the `ExtendedKalmanFilter` |

!!! note
    The single-RGP constructor `ExtendedKalmanFilter(rgp)` sets ``R_2 = r(u_t)`` and
    leaves out the sensor noise. For noisy data, build the filter with the
    component constructor and return `uncertainty_gp(rgp, u) + σn^2` from `R2`.

## References

1. C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine
   Learning*, MIT Press, 2006.
   Available online: [gaussianprocess.org/gpml](https://gaussianprocess.org/gpml/)

2. M. F. Huber, "Recursive Gaussian process: On-line regression and learning,"
   *Pattern Recognition Letters*, vol. 45, pp. 85-91, 2014.
   DOI: [10.1016/j.patrec.2014.03.004](https://doi.org/10.1016/j.patrec.2014.03.004)

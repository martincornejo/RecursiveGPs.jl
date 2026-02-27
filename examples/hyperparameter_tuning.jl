# # Hyperparameter Tuning
#
# This tutorial shows how to tune the RGP kernel hyperparameters (variance
# ``\sigma^2``, length-scale ``\ell``, and measurement noise ``R_2``)
# using gradient-based optimisation via
# [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) and
# [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/).
#
# ## Setup

using RecursiveGPs
using AbstractGPs
using StaticArrays
using ComponentArrays
using LinearAlgebra
using ForwardDiff
using Optimization
using OptimizationOptimJL
using LineSearches
using LowLevelParticleFilters
using Random
using CairoMakie

# ## Positivity Transformation
#
# Hyperparameters must be positive, but unconstrained optimisers work on
# ``\mathbb{R}``.  We use a sigmoid (softplus-style) map
# ``\theta_+ = \sigma(\theta) = 1/(1 + e^{-\theta})`` together with its
# inverse to transform between the two spaces.

sigmoid(x)     = 1 / (1 + exp(-x))
inv_sigmoid(x) = log(x / (1 - x))

# ## Model Factory
#
# `build_kf` constructs a fresh KF from a parameter vector.  This is called
# inside the loss function and must be compatible with ForwardDiff dual numbers.

function build_kf(θ, ϑ)
    b0  = collect(range(0, 1, length = ϑ.n_basis))
    gp  = GP(ConstMean(ϑ.mean), θ.σ * with_lengthscale(SEKernel(), θ.ℓ))
    rgp = RGP(gp, b0)

    dynamics(x, u, p, t)    = x
    measurement(x, u, p, t) = measurement_gp(p.rgp1, x, u) |> SVector{1}
    R2(x, u, p, t)           = SMatrix{1, 1}(θ.R2)

    return ExtendedKalmanFilter((; rgp1 = rgp), dynamics, measurement, R2)
end

# ## Dataset

Random.seed!(7)

f(b) = 0.5 * b + 0.1 * sinpi(b * 2)

n  = 100
us = 0.1 .+ rand(n) / 1.5
ys = [SA[f(u) + 5e-3 * randn()] for u in us];

# ## Loss Function
#
# We minimise the sum of squared innovations.
# The parameters are stored in sigmoid-space; we convert them back to positive
# values inside the loss.

function loss_function(θ_raw, p)
    (; ϑ, us, ys) = p
    θ  = sigmoid.(θ_raw)
    kf = build_kf(θ, ϑ)

    cost = zero(eltype(θ_raw))
    for (u, y) in zip(us, ys)
        ll, e = correct!(kf, u, y, kf.p)
        predict!(kf, u)
        cost += dot(e, e)
    end
    return cost
end

# ## Optimisation Setup
#
# Initial hyperparameters (in positive space → map to unconstrained space):

ϑ  = (; n_basis = 20, mean = 0.0)
θ0 = ComponentVector(σ = 0.2, ℓ = 0.8, R2 = 0.01)
θ0_raw = inv_sigmoid.(θ0)

p = (; ϑ, us, ys)

adtype = AutoForwardDiff()
optf   = OptimizationFunction(loss_function, adtype)
prob   = OptimizationProblem(optf, θ0_raw, p)

# Run LBFGS with backtracking line search:
alg = LBFGS(linesearch = LineSearches.BackTracking())
sol = solve(prob, alg; reltol = 1e-4, show_trace = true)

θ_opt = sigmoid.(sol.u)
@info "Optimal hyperparameters" σ=θ_opt.σ  ℓ=θ_opt.ℓ  R2=θ_opt.R2

# ## Retrain with Optimal Parameters

kf_opt = build_kf(θ_opt, ϑ)

for (u, y) in zip(us, ys)
    kf_opt(u, y)
end

# ## Plot Results

b_plt  = collect(range(0.0, 0.9, length = 100))
pred_μ = Float64[]
pred_σ = Float64[]

for u in b_plt
    p = predict_kf(kf_opt, u)
    push!(pred_μ, p.μ[1])
    push!(pred_σ, sqrt(p.Σ[1, 1]))
end

fig = Figure()
ax  = CairoMakie.Axis(fig[1, 1]; title = "RGP with tuned hyperparameters")

lines!(ax, b_plt, f.(b_plt);   label = "Ground truth")
lines!(ax, b_plt, pred_μ;      color = :orange, label = "Posterior mean")
band!(ax,  b_plt,
      pred_μ .+ 2 .* pred_σ,
      pred_μ .- 2 .* pred_σ;
      color = (:orange, 0.3), label = "±2σ")
scatter!(ax, us, [y[1] for y in ys]; color = :red, label = "Training data")

ylims!(ax, 0.05, 0.35)
axislegend(ax; position = :rb)
fig

# !!! tip
#     For a log-likelihood loss (MLE), replace the squared-error accumulation
#     with `cost -= ll` where `ll` is the log-likelihood returned by
#     `correct!`.
#
#     To tune the number of basis points as well, add it to `ϑ` and rebuild
#     `b0` accordingly — though `n_basis` is discrete and cannot be
#     differentiated.

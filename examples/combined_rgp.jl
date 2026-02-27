# # Multi-Component RGPs
#
# This tutorial shows how to compose multiple RGPs into a single Extended
# Kalman Filter to model outputs that depend on several latent functions.
#
# ## Problem Setup
#
# We observe
# ```math
# y_t = f_1(u_1) + u_2 \cdot f_2(u_1)
# ```
# where ``f_1(b) = e^b`` and ``f_2(b) = 0.1 + 0.5b + 0.1\sin(2\pi b)``.
# The input ``u_2`` is a scalar gain, so neither function is directly
# observed — only their weighted sum is.

using RecursiveGPs
using AbstractGPs
using StaticArrays
using ComponentArrays
using LinearAlgebra
using Random
using CairoMakie

# ## Dataset

Random.seed!(123)

f1(b) = exp(b)
f2(b) = 0.1 + 0.5 * b + 0.1 * sinpi(b * 2)

n   = 100
ts  = collect(range(0, 100, n))
u1  = 0.1 .+ rand(n) / 1.5
u2  = 0.2 .* randn(n)
gt  = @. f1(u1) + u2 * f2(u1)

ys  = [SA[y] for y in gt]
us  = [(; u1 = u1[i], u2 = u2[i]) for i in 1:n];

# ## Define RGP Components
#
# Each latent function gets its own [`RGP`](@ref) with a separate kernel.
# Both share the same basis-point grid.

b0 = collect(0:0.05:1)

rgp_a = RGP(0.8 * with_lengthscale(SEKernel(), 0.3), b0)   # models f1
rgp_b = RGP(0.1 * with_lengthscale(SEKernel(), 0.3), b0)   # models f2

components = (; a = rgp_a, b = rgp_b);

# ## Dynamics, Measurement, and Noise
#
# The KF state is the concatenation of the two component states:
# ``x = [g^{(a)}; g^{(b)}]``.
#
# **Dynamics**: identity (both GPs are stationary in the index domain).

dynamics(x, u, p, t) = x

# **Measurement**: combine component outputs according to the observation model.
# `xid` axes are stored in `p` for slicing `x` into named components.

function measurement(x, u, p, t)
    (; xid) = p
    xc = ComponentVector(x, xid)
    μ1 = measurement_gp(p.a, xc.a, u[1])
    μ2 = measurement_gp(p.b, xc.b, u[1])
    return (μ1 + u[2] * μ2) |> SVector{1}
end

# **Noise covariance**: GP conditional variances, propagated through the
# observation model using the chain rule of variance.

function R2(x, u, p, t)
    r1 = uncertainty_gp(p.a, u[1])
    r2 = uncertainty_gp(p.b, u[1])
    return (r1 + u[2]^2 * r2) |> SMatrix{1, 1}
end

# ## Construct and Train the Filter

kf = ExtendedKalmanFilter(components, dynamics, measurement, R2)

for (u, y) in zip(us, ys)
    kf(u, y)
end

# ## Predict Output

pred_μ = Float64[]
pred_σ = Float64[]

for u in us
    p = predict_kf(kf, u)
    push!(pred_μ, p.μ[1])
    push!(pred_σ, sqrt(p.Σ[1, 1]))
end

# ## Plot Output vs Ground Truth

fig1 = Figure()
ax1  = CairoMakie.Axis(fig1[1, 1]; title = "Combined RGP output")

lines!(ax1,  ts, gt;     label = "Ground truth")
lines!(ax1,  ts, pred_μ; color = :orange, label = "Posterior mean")
band!(ax1,   ts,
      pred_μ .+ 2 .* pred_σ,
      pred_μ .- 2 .* pred_σ;
      color = (:orange, 0.3))
scatter!(ax1, ts, gt; color = :red, markersize = 4, label = "Training data")

axislegend(ax1)
fig1

# ## Extract Individual Component Predictions
#
# Use [`predict_gp`](@ref) with a component symbol to query each latent
# function independently.

b_plot = collect(range(0.0, 1.0, length = 100))

pred_a = predict_gp(kf, b_plot, :a)
pred_b = predict_gp(kf, b_plot, :b)
σ_a    = sqrt.(diag(pred_a.Σ))
σ_b    = sqrt.(diag(pred_b.Σ))

fig2 = Figure()
axs  = [CairoMakie.Axis(fig2[i, 1]) for i in 1:2]

axs[1].title = "Component a:  f₁(b) = exp(b)"
lines!(axs[1], b_plot, f1.(b_plot);   label = "Ground truth")
lines!(axs[1], b_plot, pred_a.μ;      color = :orange, label = "Posterior mean")
band!(axs[1],  b_plot,
      pred_a.μ .+ 2 .* σ_a,
      pred_a.μ .- 2 .* σ_a;
      color = (:orange, 0.3))
scatter!(axs[1], u1, f1.(u1); color = :red, label = "Training inputs")

axs[2].title = "Component b:  f₂(b) = 0.1 + 0.5b + 0.1sin(2πb)"
lines!(axs[2], b_plot, f2.(b_plot);   label = "Ground truth")
lines!(axs[2], b_plot, pred_b.μ;      color = :orange, label = "Posterior mean")
band!(axs[2],  b_plot,
      pred_b.μ .+ 2 .* σ_b,
      pred_b.μ .- 2 .* σ_b;
      color = (:orange, 0.3))
scatter!(axs[2], u1, f2.(u1); color = :red, label = "Training inputs")

axislegend.(axs; position = :rb)
fig2

# !!! note
#     `state(kf, :a)` and `covariance(kf, :a)` give direct access to the
#     component sub-state and covariance block without projecting onto query
#     points. See the [API Reference](@ref "API Reference") for full signatures.

# # Basic RGP with Kalman Filter
#
# This tutorial demonstrates how to fit a single Recursive Gaussian Process
# (RGP) to noisy scalar observations using an Extended Kalman Filter.
#
# ## Setup
#
# Load the required packages:

using RecursiveGPs
using AbstractGPs
using StaticArrays
using LinearAlgebra
using Random
using CairoMakie

# ## Dataset
#
# We approximate the function ``f(b) = 0.5b + 0.1\sin(2\pi b)`` using 100
# noisy observations drawn uniformly from ``[0.1, 0.77]``.

Random.seed!(42)
f(b) = 0.5 * b + 0.1 * sinpi(b * 2)

n  = 100
us = 0.1 .+ rand(n) / 1.5
ys = [SA[f(u) + 5e-3 * randn()] for u in us];

# ## Model Setup
#
# We define a scaled squared-exponential kernel and 21 evenly-spaced basis
# points covering the input domain ``[0, 1]``:

kernel = 0.01 * with_lengthscale(SEKernel(), 0.3)
b0     = collect(0:0.05:1)

# Optionally add a mean function (here a linear prior):
m1(x) = 0.1 + 0.5 * x
rgp1  = RGP(m1, kernel, b0)

# Construct the Extended Kalman Filter.  The KF state is the vector of GP
# values at the basis points, initialized at the prior mean and covariance.

kf = ExtendedKalmanFilter(rgp1)

# ## Training
#
# Process all observations sequentially.  Each call to `kf(u, y)` runs one
# predict–correct cycle and updates the posterior over the basis-point values.

for (u, y) in zip(us, ys)
    kf(u, y)
end

# ## Prediction
#
# Query the fitted function on a test grid using [`predict_gp`](@ref).
# This projects the KF posterior onto arbitrary input locations.

b_test  = collect(range(0.0, 0.9, length = 100))
pred_μ  = Float64[]
pred_σ  = Float64[]

for u in b_test
    p = predict_kf(kf, u)
    push!(pred_μ, p.μ[1])
    push!(pred_σ, sqrt(p.Σ[1, 1]))
end

# ## Plot
#
# The orange band shows ±2σ posterior uncertainty.

fig = Figure()
ax  = CairoMakie.Axis(fig[1, 1]; title = "RGP fit: f(b) = 0.5b + 0.1sin(2πb)")

lines!(ax, b_test, f.(b_test);   label = "Ground truth")
lines!(ax, b_test, pred_μ;       color = :orange, label = "Posterior mean")
band!(ax,  b_test,
      pred_μ .+ 2 .* pred_σ,
      pred_μ .- 2 .* pred_σ;
      color = (:orange, 0.3), label = "±2σ")
scatter!(ax, us, [y[1] for y in ys]; color = :red, label = "Training data")

ylims!(ax, 0.05, 0.35)
axislegend(ax; position = :rb, merge = true)
fig

# The fit closely tracks the ground truth with narrow uncertainty bands in
# the region covered by training data.
#
# !!! tip
#     Increase the number of basis points `b0` for a more flexible model, or
#     decrease for faster updates.  See the
#     [Hyperparameter Tuning](@ref "Hyperparameter Tuning") tutorial to learn
#     how to optimise kernel parameters automatically.

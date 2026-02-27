using RecursiveGPs
using Random
using AbstractGPs
using LinearAlgebra
using ComponentArrays
using StaticArrays
using ForwardDiff
using Optimization
using OptimizationOptimJL
using LineSearches
using LowLevelParticleFilters
using CairoMakie


# === optimization functions
function softplus(x)
    return 1 / (1 + exp(-x))
end

function inv_softplus(x)
    return log(x / (1 - x))
end

function build_kf(θ, ϑ)
    b0 = collect(range(0, 1, length = ϑ.n_basis))
    gp = GP(ConstMean(ϑ.mean), θ.σ * with_lengthscale(SEKernel(), θ.ℓ))
    rgp1 = RGP(gp, b0)
    components = (; rgp1)

    function dynamics(x, u, p, t)
        return x
    end

    function measurement(x, u, p, t)
        return measurement_gp(p.rgp1, x, u) |> SVector{1}
    end

    function R2(x, u, p, t)
        return SMatrix{1, 1}(θ.R2)
    end
    return ExtendedKalmanFilter(components, dynamics, measurement, R2)
end


function loss_function(θ, p)
    """Squared error"""
    (; ϑ, us, ys) = p
    θ_ = softplus.(θ)
    kf = build_kf(θ_, ϑ)

    cost = 0.0
    for (u, y) in zip(us, ys)
        ll, e = correct!(kf, u, y, kf.p)
        predict!(kf, u)
        cost += dot(e, 1, e)
    end
    return cost
end

# === dataset
begin
    f(b) = 0.5 * b + 0.1 * sinpi(b * 2) # <- function to infer
    data = let n = 100
        b = 0.1 .+ rand(n) / 1.5
        y = f.(b) .+ 5.0e-3 .* randn(n)
        (; b, y)
    end
    ys = [SA[y] for y in data.y]
    us = data.b
end

# === optimization model
begin
    ϑ = (;
        n_basis = 20,
        mean = 0.0,
    )
    p = (; ϑ, us, ys)

    θ0 = ComponentVector(;
        σ = 0.2,
        ℓ = 0.8,
        R2 = 0.1^2
    )

    θ0 = inv_softplus.(θ0)

    loss_function(θ0, p)
    adtype = AutoForwardDiff()
    fs = OptimizationFunction(loss_function, adtype)
    prob = OptimizationProblem(fs, θ0, p)

    alg = LBFGS(linesearch = LineSearches.BackTracking())
    sol = solve(
        prob,
        alg,
        reltol = 1.0e-4,
        show_trace = true
    )

    θ = softplus.(sol.u)
end

# === Training
begin
    kf = build_kf(θ, ϑ)
    for (u, y) in zip(us, ys)
        kf(u, y)
    end
end


# === plots
begin
    ys_pred = (; μ = [], σ = [])
    us_plt = collect(range(0.0, 0.9, length = 100))
    gt = f.(us_plt)

    for u in us_plt
        y_pred = predict_kf(kf, u)
        push!(ys_pred.μ, y_pred.μ[1])
        push!(ys_pred.σ, sqrt(y_pred.Σ[1, 1]))
    end


    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1])
    ax.title = "RGP: 0.5 * b + 0.1 * sinpi(b * 2)"
    lines!(ax, us_plt, gt, label = "GT")
    lines!(ax, us_plt, ys_pred.μ, color = :orange, label = "Prediction")
    band!(
        ax,
        us_plt,
        ys_pred.μ .+ 2ys_pred.σ,
        ys_pred.μ .- 2ys_pred.σ,
        color = :orange,
        alpha = 0.3
    )

    scatter!(ax, us, data.y, color = :red, label = "Train Points")

    ylims!(ax, 0.05, 0.35)
    axislegend(ax, position = :rb)
    fig
end

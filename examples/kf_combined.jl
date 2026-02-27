using RecursiveGPs
using Random
using AbstractGPs
using LinearAlgebra
using ComponentArrays
using StaticArrays
using CairoMakie


#### KF COMBINED-RGPs-####
# === dataset
begin
    f1(b) = exp(b)
    f2(b) = 0.1 + 0.5 * b + 0.1 * sinpi(b * 2) # <- function to infer
    data = let n = 100
        rng = Xoshiro(123)
        ts = collect(range(0, 100, n))
        u1 = 0.1 .+ rand(rng, n) / 1.5
        u2 = 0.2 .* randn(rng, n)
        gt = @. f1(u1) + u2 * f2(u1)
        y = gt
        (; ts, gt, u1, u2, y)
    end

    ys = [SA[y] for y in data.y]
    us = [(; u1 = data.u1[idx], u2 = data.u2[idx]) for idx in 1:length(ys)]
end

# Generating RGP and Component
begin
    b0 = collect(0:0.05:1)
    kernel1 = 0.8 * with_lengthscale(SEKernel(), 0.3)
    rgp1 = RGP(kernel1, b0)


    kernel2 = 0.1 * with_lengthscale(SEKernel(), 0.3)
    rgp2 = RGP(kernel2, b0)

    components = (; a = rgp1, b = rgp2)

    dynamics(x, u, p, t) = x

    function measurement(x, u, p, t)
        (; xid) = p
        xc = ComponentVector(x, xid)
        μ1 = measurement_gp(p.a, xc.a, u[1])
        μ2 = measurement_gp(p.b, xc.b, u[1])
        return μ1 + u[2] * μ2 |> SVector{1}
    end

    function R2(x, u, p, t)
        R1 = uncertainty_gp(p.a, u[1])
        R2 = uncertainty_gp(p.b, u[1])
        return R1 + u[2]^2 * R2 |> SMatrix{1, 1}
    end

end

# Instantiation and Training
begin
    kf = ExtendedKalmanFilter(components, dynamics, measurement, R2)
    for (u, y) in zip(us, ys)
        kf(u, y)
    end
end

# Plot Output
begin
    ys_pred = (; μ = [], σ = [])
    for u in us
        y_pred = predict_kf(kf, u)
        push!(ys_pred.μ, y_pred.μ[1])
        push!(ys_pred.σ, sqrt(y_pred.Σ[1, 1]))
    end

    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1])
    lines!(ax, data.ts, data.gt, label = "GT")
    lines!(ax, data.ts, ys_pred.μ, color = :orange, label = "Prediction")
    band!(
        ax,
        data.ts,
        ys_pred.μ .+ 2ys_pred.σ,
        ys_pred.μ .- 2ys_pred.σ,
        color = :orange,
        alpha = 0.3
    )

    scatter!(ax, data.ts, data.y, color = :red, label = "Train Points")
    axislegend(ax)
    fig
end

# Plot Components
begin
    b_plot = collect(range(0.0, 1.0, length = 100))

    pred_a = predict_gp(kf, b_plot, :a)
    pred_b = predict_gp(kf, b_plot, :b)
    σ_a = sqrt.(diag(pred_a.Σ))
    σ_b = sqrt.(diag(pred_b.Σ))

    fig = Figure()
    axs = [CairoMakie.Axis(fig[i, 1]) for i in 1:2]

    axs[1].title = "RGP 1: exp(b) function"
    lines!(axs[1], b_plot, f1.(b_plot), label = "GT")
    lines!(axs[1], b_plot, pred_a.μ, color = :orange, label = "Prediction")
    band!(
        axs[1],
        b_plot,
        pred_a.μ .+ 2σ_a,
        pred_a.μ .- 2σ_a,
        color = :orange,
        alpha = 0.3
    )
    scatter!(axs[1], data.u1, f1.(data.u1), color = :red, label = "Train Points")

    axs[2].title = "RGP 2: 0.1 + 0.5 * b + 0.1 * sinpi(b * 2)"
    lines!(axs[2], b_plot, f2.(b_plot), label = "GT")
    lines!(axs[2], b_plot, pred_b.μ, color = :orange, label = "Prediction")
    band!(
        axs[2],
        b_plot,
        pred_b.μ .+ 2σ_b,
        pred_b.μ .- 2σ_b,
        color = :orange,
        alpha = 0.3
    )
    scatter!(axs[2], data.u1, f2.(data.u1), color = :red, label = "Train Points")

    axislegend.(axs, position = :rb)
    fig
end

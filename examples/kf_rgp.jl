using RecursiveGPs
using Random
using AbstractGPs
using LinearAlgebra
using ComponentArrays
using StaticArrays
using CairoMakie


#### KF-RGP####
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


# === model
begin

    b0 = collect(0:0.05:1)

    m1(x) = 0.1 + 0.5 .* x
    kernel1 = 0.01 * with_lengthscale(SEKernel(), 0.3)
    rgp1 = RGP(m1, kernel1, b0)

    kf = ExtendedKalmanFilter(rgp1)

end

# === training
begin
    for (u, y) in zip(us, ys)
        kf(u, y)
    end
end

# === plots
begin
    us_test = collect(range(0.0, 0.9, length = 100))
    ys_pred = (; μ = [], σ = [])

    for u in us_test
        y_pred = predict_kf(kf, u)
        push!(ys_pred.μ, y_pred.μ[1])
        push!(ys_pred.σ, sqrt(y_pred.Σ[1, 1]))
    end
    y_pred

    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1])
    ax.title = "RGP: 0.5 * b + 0.1 * sinpi(b * 2)"
    lines!(ax, us_test, f.(us_test), label = "GT")
    lines!(ax, us_test, ys_pred.μ, color = :orange, label = "Prediction")
    band!(
        ax,
        us_test,
        ys_pred.μ .+ 2ys_pred.σ,
        ys_pred.μ .- 2ys_pred.σ,
        color = :orange,
        alpha = 0.3,
        label = "Prediction"
    )

    scatter!(ax, us, data.y, color = :red, label = "Train Points")

    ylims!(ax, 0.05, 0.35)
    axislegend(ax, position = :rb, merge = true)
    fig
end

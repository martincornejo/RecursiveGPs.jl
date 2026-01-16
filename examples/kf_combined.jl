using .RecursiveGPs
using Random
using AbstractGPs
using LinearAlgebra
using ComponentArrays
using CairoMakie


#### KF COMBINED-RGPs-####
## === dataset
begin
    f1(b) = exp(b)
    f2(b) = 0.1 + 0.5 * b + 0.1 * sinpi(b * 2) # <- function to infer
    df = let n = 100
        rng = Xoshiro(123)
        ts = collect(range(0,100, n))
        b = 0.1 .+ rand(rng, n) / 1.5
        i = 0.2 .* randn(rng, n)
        gt = @. f1(b) + i * f2(b)
        y = gt
        DataFrame(; ts,gt, b, i, y)
    end

    ys = [SA[y] for y in df.y]
    us = [[x.b, x.i] for x in eachrow(df)]
end

## Generating RGP and Component
begin
    b0 = collect(0:0.05:1)
    kernel1 = 0.02 * with_lengthscale(SEKernel(), 0.1)
    rgp1 = RGP(kernel1, b0)


    kernel2 = 0.02 * with_lengthscale(SEKernel(), 0.1)
    rgp2 = RGP(kernel2, b0)

    components = (; a=rgp1, b=rgp2)

    dynamics(x, u, p, t) = x

    function measurement(x, u, p, t)
        (; xid) = p
        xc = ComponentVector(x, xid)
        μ1 = measurement_gp(p.a, xc.a, u[1])
        μ2 = measurement_gp(p.b, xc.b, u[1])
        μ1 + u[2] * μ2 |> SVector{1}
    end

    function R2(x, u, p, t)
        R1 = uncertainty_gp(p.a, u[1])
        R2 = uncertainty_gp(p.b, u[1])
        R1 + u[2]^2 * R2 |> SMatrix{1,1}
    end

end

## Instantiation and Training
begin
    kf = make_ekf(components, dynamics, measurement, R2)
    for (u, y) in zip(us, ys)
        kf(u,y)
    end
end

## Plot Output
begin
    ys_pred = (; μ = [], σ = [])
    for u in us
        y_pred = predict_kf!(kf, u)
        push!(ys_pred.μ, y_pred.μ[1])
        push!(ys_pred.σ, y_pred.σ[1])
    end
    
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1])
    lines!(ax, df.ts, df.gt, label = "GT")
    lines!(ax, df.ts, ys_pred.μ, color = :orange, label = "Prediction")
    band!(ax, 
        df.ts, 
        ys_pred.μ .+ 2ys_pred.σ,
        ys_pred.μ .- 2ys_pred.σ,
        color = :orange,
        alpha=0.3)

    scatter!(ax, ts, df.y, color = :red, label = "Train Points")
    axislegend(ax)
    fig
end

# Plot Components
begin
    cx = ComponentVector(kf.x, kf.p.xid)
    cσ = ComponentVector(sqrt.(diag(kf.R)), kf.p.xid)

    fig = Figure()
    axs = [CairoMakie.Axis(fig[i, 1]) for i in 1:2]
    
    axs[1].title = "RGP 1: exp(b) function"
    lines!(axs[1], b0, f1.(b0) , label = "GT")
    lines!(axs[1], b0, cx.a, color = :orange, label = "Prediction")
    band!(axs[1], 
        b0, 
        cx.a .+ 2cσ.a,
        cx.a .- 2cσ.a,
        color = :orange,
        alpha=0.3)

    scatter!(axs[1], df.b, f1.(df.b), color = :red, label = "Train Points")

    axs[2].title = "RGP 2: 0.1 + 0.5 * b + 0.1 * sinpi(b * 2)"
    lines!(axs[2],b0, f2.(b0) , label = "GT")
    lines!(axs[2],b0, cx.b, color = :orange, label = "Prediction")
    band!(axs[2], 
        b0, 
        cx.b .+ 2cσ.b,
        cx.b .- 2cσ.b,
        color = :orange,
        alpha=0.3)
    scatter!(axs[2], df.b, f2.(df.b), color = :red, label = "Train Points")

    axislegend.(axs, position = :rb)
    fig
end



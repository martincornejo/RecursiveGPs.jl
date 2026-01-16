using RecursiveGPs
using Random
using AbstractGPs
using LinearAlgebra
using ComponentArrays
using StaticArrays
using CairoMakie
using DataFrames


#### KF-RGP####
## === dataset
begin
    f(b) = 0.5 * b + 0.1 * sinpi(b * 2)# <- function to infer
    
    df = let n = 100
        b = 0.1 .+ rand(n) / 1.5
        y = f.(b) .+ 5e-3 .* randn(n)
        DataFrame(; b, y)
    end
    ys = [SA[y] for y in df.y]
    us = [b for b in df.b]
end


## === model
begin
    
    b0 = collect(0:0.05:1)

    m1(x) = 0.1 + 0.5 .* x
    kernel1 = 0.02 * with_lengthscale(SEKernel(), 0.1)
    rgp1 = RGP(m1, kernel1, b0)

    components = (; rgp1)

    ## Kf functions
    function dynamics(x,u,p,t)
        x
    end

    function measurement(x,u,p,t)
        [measurement_gp(p.rgp1, x, u)]
    end

    function R2(x,u,p,t)
        [1e-3]
    end
    kf = make_ekf(components, dynamics, measurement, R2)

end

## === training
begin
    for (u, y) in zip(us, ys)
        kf(u,y)
    end
end

## === plots
begin
    ys_pred = (; μ = [], σ = [])
    us_plt = collect(range(0.0,0.9, length=100))
    gt = f.(us_plt)

    kf_copy = deepcopy(kf)
    for u in us_plt
       # y_pred = predict_kf!(kf_copy, u)
        # Note: 
            # predict_gp assumes R2 = 0 since there is no noise
            # predict_kf!  uses de KF R2 if used
        y_pred = predict_gp(kf_copy, u, :rgp1)
        push!(ys_pred.μ, y_pred.μ)
        push!(ys_pred.σ, y_pred.σ)
    end
    
    
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1])
    ax.title = "RGP: 0.5 * b + 0.1 * sinpi(b * 2)"
    lines!(ax, us_plt, gt, label = "GT")
    lines!(ax, us_plt, ys_pred.μ, color = :orange, label = "Prediction")
    band!(ax, 
        us_plt, 
        ys_pred.μ .+ 2ys_pred.σ,
        ys_pred.μ .- 2ys_pred.σ,
        color = :orange,
        alpha=0.3)

    scatter!(ax, us, df.y, color = :red, label = "Train Points")

    ylims!(ax, 0.05, 0.35)
    axislegend(ax, position = :rb)
    fig
end



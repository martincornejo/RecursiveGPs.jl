using .RecursiveGPs
using Random
using AbstractGPs
using LinearAlgebra
using ComponentArrays
using CairoMakie
using StatsBase


#### KF COMBINED-RGP and CTE ####
begin
    # Generating data and applying Z-score normalization
    fun(x) = x/2 + 25 * x/ (1 + x^2) * cos(x)
    us = collect(-10:0.1:10)
    σ = sqrt(0.1)
    ys = fun.(us) .+ σ .* randn(length(us))
    
    dt = (; 
        u = StatsBase.fit(ZScoreTransform, us),
        y = StatsBase.fit(ZScoreTransform, ys)
    )
    
    us_tr = StatsBase.transform(dt.u, us)
    ys_tr = StatsBase.transform(dt.y, ys)
   
    # Generating RGP and Component
    n_basis = 20
    limits = extrema(us_tr)
    b0 = collect(range(limits..., length=n_basis))               
    
    gp = GP(ZeroMean(), SqExponentialKernel())
    rgp = RGP(gp, b0)
    
    ## Instantiation and Training
    kf = make_ekf(rgp)

    for (u, y) in zip(us_tr, ys_tr)
        kf(u,[y])
    end

    ## Plots
    us_test = collect(-12:0.1:12)
    gt = fun.(us_test)
    ys_pred = (; μ = [], σ = [])
    
    for u in us_test
        y_pred = predict_kf!(kf, u)
        push!(ys_pred.μ, y_pred.μ[1])
        push!(ys_pred.σ, y_pred.σ[1])
    end
  
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1])

    lines!(ax,us_test,gt, label = "GT")
    lines!(ax,us_test, ys_pred.μ,color = :orange, label = "Prediction")
    band!(ax, 
        us_test, 
        ys_pred.μ .+ 2ys_pred.σ,
        ys_pred.μ .- 2ys_pred.σ,
        color = :orange,
        alpha=0.3)
    axislegend(ax)
    fig
end
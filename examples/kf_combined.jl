using .RecursiveGPs
using Random
using AbstractGPs
using LinearAlgebra
using ComponentArrays
using CairoMakie
using StatsBase
using LowLevelParticleFilters

#### KF COMBINED-RGPs-####
## Generating data and applying Z-score normalization
begin
    fun_1(x) = sin(x)
    fun_2(x) = 25 * x / (1 + x^2) * cos(x)
    cte = 2
    fun(x,z) = z * fun_1(x) + fun_2(x) + z * x/cte
    n_points = 100
    ts = collect(range(0.0,100, length=n_points))
    xs = collect(range(-10,10,length=n_points))
    zs = collect(range(0,1,length=n_points))
    us = [(;x, z) for (x, z) in zip(xs, zs)]
    
    gt = [fun(x,z) + sqrt(0.1) * randn() for (x, z) in zip(xs, zs)]
    ys = gt .+ sqrt(0.1) * randn(n_points)
end

## Generating RGP and Component
begin
    n_basis = 20
    limits = extrema(xs_tr)
    b0 = collect(range(-12, 12, length=n_basis))               
    
    gp_1 = GP(ZeroMean(), SqExponentialKernel())
    rgp_1 = RGP(gp_1, b0)
    
    gp_2 = GP(ZeroMean(), SqExponentialKernel())
    rgp_2 = RGP(gp_2, b0)

    cte_A = (; μ0 = [1], Σ0 =[0.5], R1 = [1e-6] )
    components = (; rgp_1, rgp_2, cte_A)

    ## Kf functions
    function dynamics(x,u,p,t)
        x
    end

    function measurement(x,u,p,t)
        cx = ComponentVector(x, p.xid)
        u.z * measurement_gp(p.rgp_1, cx.rgp_1, u.x) .+ measurement_gp(p.rgp_2, cx.rgp_2, u.x) .+ u.x * u.z *cx.cte_A
    end

    function R2(x,u,p,t)
        [0.01]
    end
end

## Instantiation and Training
begin
    kf = make_comb_ekf(components, dynamics, measurement, R2)
    for (u, y) in zip(us, ys)
        kf(u,[y])
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
    lines!(ax, ts, gt, label = "GT")
    lines!(ax, ts, ys_pred.μ, color = :orange, label = "Prediction")
    band!(ax, 
        ts, 
        ys_pred.μ .+ 2ys_pred.σ,
        ys_pred.μ .- 2ys_pred.σ,
        color = :orange,
        alpha=0.3)
    axislegend(ax)
    fig
end

# Plot Components
begin
    cx = ComponentVector(kf.x, kf.p.xid)
    cσ = ComponentVector(sqrt.(diag(kf.R)), kf.p.xid)

    fig = Figure()
    axs = [CairoMakie.Axis(fig[i, 1]) for i in 1:2]
    
    lines!(axs[1],b0, fun_1.(b0) , label = "GT")
    lines!(axs[1], b0, cx.rgp_1, color = :orange, label = "Prediction")
    band!(axs[1], 
        b0, 
        cx.rgp_1 .+ 2cσ.rgp_1,
        cx.rgp_1 .- 2cσ.rgp_1,
        color = :orange,
        alpha=0.3)

    lines!(axs[2],b0, fun_2.(b0) , label = "GT")
    lines!(axs[2],b0, cx.rgp_2, color = :orange, label = "Prediction")
    band!(axs[2], 
        b0, 
        cx.rgp_2 .+ 2cσ.rgp_2,
        cx.rgp_2 .- 2cσ.rgp_2,
        color = :orange,
        alpha=0.3)
    
    axislegend.(axs)
    fig
end



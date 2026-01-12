using .RecursiveGPs
using Random
using AbstractGPs
using LinearAlgebra
using ComponentArrays
using ForwardDiff
using Optimization
using OptimizationOptimJL

#### KF COMBINED-RGP and CTE VAR ####

function merge(θ::ComponentVector, ϑ::ComponentVector)
    ComponentVector(Base.merge(NamedTuple(θ), NamedTuple(ϑ)))
end
function merge(θ::ComponentVector, ϑ::NamedTuple) 
    ComponentVector(Base.merge(NamedTuple(θ),  ϑ))
end
function merge(θ::NamedTuple,  ϑ::ComponentVector) 
    ComponentVector(Base.merge(θ, NamedTuple( ϑ)))
end

function build_kf(θ, ϑ, df, zt)
    θ´ = merge(θ, ϑ)
    N_basis = θ.n_basis
    b0 = collect(range(-12, 12, length=N_basis))               
    
    gp = GP(ZeroMean(),θ´.rgp.l * SEKernel(θ´.rgp.σ))
    rgp = RGP(gp, b0)
    
    cte_A = (; μ0 = θ´.cte_A.μ0, Σ0 = θ´.cte_A.Σ0, R1 = θ´.cte_A.R1)

    components = (; rgp, cte_A)

    ## Kf functions
    function dynamics(x,u,p,t)
        x
    end

    function measurement(x,u,p,t)
        cx = ComponentVector(x, p.xid)
        measurement_gp(p.rgp, cx.rgp, u) .+ u * cx.cte_A
    end

    function R2(x,u,p,t)
        θ´.R2
    end


end

function loss_function(u,p)
    """Squared error"""
    (;ϑ, df, zt, us,ys) = p
    cost = 0.0
    θ_cons = softplus.(u)
    kf = build_kf(θ_cons, ϑ, df, zt)
    
    for (u, y) in zip(us,ys)
        ll, e = correct!(kf, u, y, kf.p)
        predict!(kf, u)
        cost += dot(e,1,e)
    end
    cost
end

begin
    # Generating data and applying Z-score normalization
    fun(x) = x/2 + 25 * x/ (1 + x^2) * cos(x)
    us = collect(-10:0.1:10)
    σ = sqrt(0.1)
    ys = fun.(us) .+ σ .* randn(length(us))

    # Generating RGP and Component

end

begin
    adtype = AutoForwardDiff()
    f = OptimizationFunction(loss_function, adtype)
    prob = OptimizationProblem(f, u0, p)
    
    alg = LBFGS(linesearch=LineSearches.BackTracking())
    sol = solve(prob,
        alg,
        reltol=1e-4,
        show_trace = true
    ) 

    θ = softplus.(sol.u)
end 

begin

    ## Instantiation and Training
    kf = make_comb_ekf(components, dynamics, measurement, R2)
    for (u, y) in zip(us, ys)
        kf(u,[y])
    end

    ## Plots
end




#### EXTRA ####

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
    
    dim = 3
    cte_A = (; μ0 = [1], Σ0 =[0.2], R1 = [1e-6] )
    components = (; rgp, cte_A)

    ## Kf functions
    function dynamics(x,u,p,t)
        x
    end

    function measurement(x,u,p,t)
        cx = ComponentVector(x, p.xid)
        measurement_gp(p.rgp, cx.rgp, u)
    end

    function R2(x,u,p,t)
        [0.1]
    end

    ## Instantiation and Training
    kf = make_comb_ekf(components, dynamics, measurement, R2)

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

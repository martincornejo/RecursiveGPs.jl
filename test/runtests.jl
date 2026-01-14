using Test
using .RecursiveGPs
using Random
using AbstractGPs
using LinearAlgebra
using ComponentArrays
using ForwardDiff
using Optimization
using OptimizationOptimJL
using LineSearches
using LowLevelParticleFilters


@testset "RGP" begin
    rng = MersenneTwister(123)
    
    N_basis = 5
    b0 = sort(rand(rng, N_basis))  
    g_obs = rand(rng, N_basis)      
    b_query = 0.5                   
    
    k = SEKernel() 
    m = ZeroMean() 
    
    gp_obj = GP(m, k)

    @testset "Instantiation" begin
        RGP(gp_obj, b0)
        RGP(k, b0)
        RGP(m, k, b0)
    end
    
    rgp_A = RGP(gp_obj, b0)
    rgp_B = RGP(k, b0)
    rgp_C = RGP(m, k, b0)
    
    @testset "Internal State Consistency" begin
        @test rgp_A.μ0 ≈ rgp_B.μ0
        @test rgp_A.μ0 ≈ rgp_C.μ0
        
        @test rgp_A.Σ0⁻¹ ≈ rgp_B.Σ0⁻¹
        @test rgp_A.Σ0⁻¹ ≈ rgp_C.Σ0⁻¹
    end

    @testset "Output Consistency: measurement_gp" begin
        val_A = measurement_gp(rgp_A, g_obs, b_query)
        val_B = measurement_gp(rgp_B, g_obs, b_query)
        val_C = measurement_gp(rgp_C, g_obs, b_query)

        @test val_A ≈ val_B
        @test val_A ≈ val_C
        
    end

    @testset "Output Consistency: uncertainty_gp" begin
        unc_A = uncertainty_gp(rgp_A, b_query)
        unc_B = uncertainty_gp(rgp_B, b_query)
        unc_C = uncertainty_gp(rgp_C, b_query)

        @test unc_A ≈ unc_B
        @test unc_A ≈ unc_C
        
    end
end


@testset "Combined kf" begin

    ## SET-UP
    rng = MersenneTwister(123)
    
    N_basis = 5
    b0 = sort(rand(rng, N_basis))  
    g_obs = rand(rng, N_basis)      
    b_query = 0.5                   
    
    gp_obj = GP(ZeroMean(), SqExponentialKernel())
    rgp_obj = RGP(gp_obj, b0)
    
    dim_A = 3
    A = randn(rng, dim_A, dim_A)
    comp_A = (; μ0 = rand(rng, dim_A), Σ0 = A* A' + 1e-6*I, R1 = fill(1e-3,dim_A,dim_A))
    components = (; rgp_obj, comp_A)

    function dynamics(x,u,p,t)
        x
    end

    function measurement(x,u,p,t)
        cx = ComponentVector(x, p.xid)
        [measurement_gp(p.rgp_obj, cx.rgp_obj, u) .+ sum(cx.comp_A)]
    end

    function R2(x,u,p,t)
        [1e-3]
    end

    ## Testing all possible Instantiation methods
    @testset "Instantiation" begin

        @testset "Promotion: No dual" begin
            @test_nowarn make_ekf(components, dynamics, measurement, R2)
            kf = make_ekf(components, dynamics, measurement, R2)
            @test eltype(kf.d0.μ) <: Float64
            @test eltype(kf.d0.Σ) <: Float64
            @test eltype(kf.R1) <: Float64
        end

        @testset "Promotion: μ0 dual" begin
            μ0_dual = ForwardDiff.Dual.(comp_A.μ0, 1.0)

            comp_A_dual = (; μ0 = μ0_dual, Σ0 = A* A' + 1e-6*I, R1 = fill(1e-3,dim_A,dim_A))
            components_dual = (; rgp_obj, comp_A_dual)
            
            @test_nowarn make_ekf(components_dual, dynamics, measurement, R2)
            kf = make_ekf(components_dual, dynamics, measurement, R2)

            @test eltype(kf.d0.μ) <: ForwardDiff.Dual
            @test eltype(kf.d0.Σ) <: ForwardDiff.Dual
            @test eltype(kf.R1) <: ForwardDiff.Dual
        end

        @testset "Promotion: Σ0 is Dual" begin
            Σ0_dual = ForwardDiff.Dual.(comp_A.Σ0, 1.0)

            comp_A_dual = (; μ0 = rand(rng, dim_A), Σ0 = Σ0_dual, R1 = fill(1e-3,dim_A,dim_A))
            components_dual = (; rgp_obj, comp_A_dual)

            @test_nowarn make_ekf(components_dual, dynamics, measurement, R2)
            kf = make_ekf(components_dual, dynamics, measurement, R2)
            
            @test eltype(kf.d0.μ) <: ForwardDiff.Dual
            @test eltype(kf.d0.Σ) <: ForwardDiff.Dual
            @test eltype(kf.R1) <: ForwardDiff.Dual
        end

        @testset "Promotion: R1 dual" begin

            R1_dual = ForwardDiff.Dual.(comp_A.R1, 1.0)

            comp_A_dual = (; μ0 = rand(rng, dim_A), Σ0 = A*A' + 1e-6*I, R1 = R1_dual)
            components_dual = (; rgp_obj, comp_A_dual)

            @test_nowarn make_ekf(components_dual, dynamics, measurement, R2)
            
            kf = make_ekf(components_dual, dynamics, measurement, R2)
            @test eltype(kf.d0.μ) <: ForwardDiff.Dual
            @test eltype(kf.d0.Σ) <: ForwardDiff.Dual
            @test eltype(kf.R1) <: ForwardDiff.Dual
        end

    end

    kf = make_ekf(components, dynamics, measurement, R2)
    xid = kf.p.xid
    Σid = kf.p.Σid

    @testset "Mean, Cov and R1 consistency" begin
        cx = ComponentVector(kf.x, xid)
        cR = ComponentMatrix(kf.R, Σid)
        cR1 = ComponentMatrix(kf.R1, Σid)

        @test cx.rgp_obj ≈ rgp_obj.μ0
        @test cx.comp_A ≈ comp_A.μ0
        
        @test cR[:rgp_obj, :rgp_obj] ≈ rgp_obj.Σ0
        @test cR[:comp_A, :comp_A] ≈ comp_A.Σ0

        @test cR1[:rgp_obj, :rgp_obj] ≈ rgp_obj.R1
        @test cR1[:comp_A, :comp_A] ≈ comp_A.R1
    end

    ## Training data
    N_points = 2
    us_test = rand(rng, N_points)
    ys_test = sin.(us_test)

    ## Train test
    @testset "Train" begin
        for (u, y) in zip(us_test, ys_test)
            kf(u,[y])
        end
    end
    
    ## Post training state
    cx = ComponentVector(kf.x, xid)
    cR = ComponentMatrix(kf.R, Σid)

    ## Testint predict_gp functionalities
    @testset "predict_gp: Array Input" begin
        @testset "Array input" begin
            b_ar = [2.0]
            
            m1_ar = predict_gp(kf, b_ar, :rgp_obj)
            m2_ar = predict_gp(kf, b_ar, cx.rgp_obj, cR[:rgp_obj, :rgp_obj], :rgp_obj)

            @test size(m1_ar.μ) == size(b_ar)
            @test m1_ar.μ ≈ m2_ar.μ
            @test m1_ar.σ ≈ m2_ar.σ
        end

        @testset "Scalar Input" begin
            b_fl = 2.0
            m1_fl = predict_gp(kf, b_fl, :rgp_obj)
            m2_fl = predict_gp(kf, b_fl, cx.rgp_obj, cR[:rgp_obj, :rgp_obj], :rgp_obj)
            
            @test m1_fl.μ ≈ m2_fl.μ
            @test m1_fl.σ ≈ m2_fl.σ
        end
    end

    ## Testing if parameter Hypertuning works. 
    ## Hyp GP params, cmp_A cov matrix and R2
    @testset "Hyp Tunning" begin

        function build_kf(θ, ϑ)
            N_basis = 5
            b0 = sort(rand(rng, N_basis))  
              
            
            gp_obj = GP(ZeroMean(), θ.rgp.σ * with_lengthscale(SEKernel(), θ.rgp.ℓ))
            rgp_obj = RGP(gp_obj, b0)
            
            comp_A = (; μ0 = rand(rng, dim_A), Σ0 = θ.A.Σ0, R1 = fill(1e-3,dim_A,dim_A))
            components = (; rgp_obj, comp_A)

            function R2(x,u,p,t)
                [θ.R2]
            end

            make_ekf(components, dynamics, measurement, R2)
        end

        function loss_function(θ,p)
            """Squared error"""
            (;ϑ, us, ys) = p
            kf = build_kf(θ, ϑ)

            cost = 0.0
            for (u, y) in zip(us,ys)
                ll, e = correct!(kf, u, [y], kf.p)
                predict!(kf, u)
                cost += dot(e,1,e)
            end
            cost
        end
        θ0 = ComponentVector(;
            rgp = (;
                σ = 0.8,
                ℓ = 0.8
            ), 
            A = (; Σ0 = randn(rng, dim_A, dim_A) ),
            R2 = 1e-1
        )
        ϑ = (;)
        p = (;ϑ, us = us_test, ys = ys_test)
        adtype = AutoForwardDiff()
        f = OptimizationFunction(loss_function, adtype)
        prob = OptimizationProblem(f, θ0, p)
        
        alg = LBFGS(linesearch=LineSearches.BackTracking())
        @test_nowarn sol = solve(prob, alg, reltol=1e-4, show_trace = false, maxiters = 1)   
    end
end



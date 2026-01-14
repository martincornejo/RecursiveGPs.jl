using Test
using .RecursiveGPs
using Random
using AbstractGPs
using LinearAlgebra
using ComponentArrays
using ForwardDiff



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
    rng = MersenneTwister(123)
    
    N_basis = 5
    b0 = sort(rand(rng, N_basis))  
    g_obs = rand(rng, N_basis)      
    b_query = 0.5                   
    
    gp_obj = GP(ZeroMean(), SqExponentialKernel())
    rgp_obj = RGP(gp_obj, b0)
    
    dim_A = 3
    dim_B = 1
    A = randn(rng, dim_A, dim_A)
    B = randn(rng, dim_B, dim_B)
    component_A = (; μ0 = rand(rng, dim_A), Σ0 = A* A' + 1e-6*I, R1 = fill(1e-3,dim_A,dim_A))
    component_B =  (; μ0 = rand(rng, dim_B), Σ0 = B* B' + 1e-6*I, R1 = fill(1e-3,dim_B,dim_B))
    components = (; rgp_obj, component_A, component_B)

    function dynamics(x,u,p,t)
        x
    end

    function measurement(x,u,p,t)
        cx = ComponentVector(x, p.xid)
        measurement_gp(p.rgp_obj, cx.rgp_obj, u) .+ sum(cx.component_A) .+ u .* cx.component_B
    end

    function R2(x,u,p,t)
        [1e-3]
    end

    @testset "Instantiation" begin
        @test_nowarn make_comb_ekf(components, dynamics, measurement, R2)

    end

    kf = make_comb_ekf(components, dynamics, measurement, R2)
    xid = kf.p.xid
    Σid = kf.p.Σid

    @testset "Mean, Cov and R1 consistency" begin
        cx = ComponentVector(kf.x, xid)
        cR = ComponentMatrix(kf.R, Σid)
        cR1 = ComponentMatrix(kf.R1, Σid)

        @test cx.rgp_obj ≈ rgp_obj.μ0
        @test cx.component_A ≈ component_A.μ0
        @test cx.component_B ≈ component_B.μ0
        
        @test cR[:rgp_obj, :rgp_obj] ≈ rgp_obj.Σ0
        @test cR[:component_A, :component_A] ≈ component_A.Σ0
        @test cR[:component_B, :component_B] ≈ component_B.Σ0

        @test cR1[:rgp_obj, :rgp_obj] ≈ rgp_obj.R1
        @test cR1[:component_A, :component_A] ≈ component_A.R1
        @test cR1[:component_B, :component_B] ≈ component_B.R1
    end

    N_points = 2
    us_test = rand(rng, N_points)
    ys_test = sin.(us_test)

    @testset "Train" begin
        for (u, y) in zip(us_test, ys_test)
            kf(u,[y])
        end
    end
    
    cx = ComponentVector(kf.x, xid)
    cR = ComponentMatrix(kf.R, Σid)

    @testset "predict_gp: Array Input" begin
        b_ar = [2.0]
        
        m1_ar = predict_gp(kf, b_ar, :rgp_obj)
        m2_ar = predict_gp(kf, b_ar, cx.rgp_obj, cR[:rgp_obj, :rgp_obj], :rgp_obj)

        @test size(m1_ar.μ) == size(b_ar)
        @test m1_ar.μ ≈ m2_ar.μ
        @test m1_ar.σ ≈ m2_ar.σ

    end

    @testset "predict_gp: Scalar Input" begin
        b_fl = 2.0
        m1_fl = predict_gp(kf, b_fl, :rgp_obj)
        m2_fl = predict_gp(kf, b_fl, cx.rgp_obj, cR[:rgp_obj, :rgp_obj], :rgp_obj)
        
        @test m1_fl.μ ≈ m2_fl.μ
        @test m1_fl.σ ≈ m2_fl.σ
    end

    @testset "Hyp Tunning" begin
        
        function build_kf(θ, ϑ)
            N_basis = 5
            b0 = sort(rand(rng, N_basis))                     
            
            gp_obj = GP(ZeroMean(), SqExponentialKernel())
            rgp_obj = RGP(gp_obj, b0)
            
            dim_A = 3
            dim_B = 1
            A = randn(rng, dim_A, dim_A)
            B = randn(rng, dim_B, dim_B)
            component_A = (; μ0 = rand(rng, dim_A), Σ0 = A* A' + 1e-6*I, R1 = fill(1e-3,dim_A,dim_A))
            component_B =  (; μ0 = rand(rng, dim_B), Σ0 = B* B' + 1e-6*I, R1 = fill(1e-3,dim_B,dim_B))
            components = (; rgp_obj, component_A, component_B)

            function dynamics(x,u,p,t)
                x
            end

            function measurement(x,u,p,t)
                cx = ComponentVector(x, p.xid)
                measurement_gp(p.rgp_obj, cx.rgp_obj, u) .+ sum(cx.component_A) .+ u .* cx.component_B
            end

            function R2(x,u,p,t)
                [1e-3]
            end
            kf = make_comb_ekf(components, dynamics, measurement, R2)
            kf
        end

    end
end
nothing


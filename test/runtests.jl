using Test
using RecursiveGPs
using Random
using AbstractGPs
using LinearAlgebra
using ComponentArrays
using StaticArrays
using ForwardDiff
using LowLevelParticleFilters
using Optimization
using OptimizationOptimJL
using LineSearches


@testset "RGP" begin
    rng = Xoshiro(123)

    b0 = collect(0:0.05:1)  # 21 basis points

    m1(x) = 0.1 + 0.5 * x
    kernel1 = 0.02 * with_lengthscale(SEKernel(), 0.1)
    rgp1 = RGP(m1, kernel1, b0)

    components = (; rgp1)

    dynamics(x, u, p, t) = x

    function measurement(x, u, p, t)
        [measurement_gp(p.rgp1, x, u)]
    end

    function R2(x, u, p, t)
        [1.0e-3]
    end

    # Dataset: f(b) = 0.5b + 0.1*sinpi(2b), n=100, with noise
    f(b) = 0.5 * b + 0.1 * sinpi(b * 2)
    n = 100
    bs = 0.1 .+ rand(rng, n) / 1.5
    ys_raw = f.(bs) .+ 5.0e-3 .* randn(rng, n)
    ys = [SA[y] for y in ys_raw]
    us = [b for b in bs]

    @testset "Instantiation" begin
        gp_obj = GP(m1, kernel1)
        @test_nowarn RGP(gp_obj, b0)
        @test_nowarn RGP(kernel1, b0)
        @test_nowarn RGP(m1, kernel1, b0)
    end

    @testset "Constructor consistency" begin
        gp_obj = GP(m1, kernel1)
        rgp_A = RGP(gp_obj, b0)
        rgp_B = RGP(m1, kernel1, b0)

        @test rgp_A.μ0 == rgp_B.μ0
        @test rgp_A.Σ0⁻¹ == rgp_B.Σ0⁻¹
    end

    @testset "Correctness: measurement_gp" begin
        g_obs = rand(rng, length(b0))
        b_query = 0.5

        # Manual computation: m(b) + k(b, b0)' * Σ0⁻¹ * (g - μ0)
        kb_b0 = rgp1.gp.kernel.(Ref(b_query), b0)
        H = kb_b0' * rgp1.Σ0⁻¹
        expected = H * (g_obs - rgp1.μ0) + m1(b_query)

        val = measurement_gp(rgp1, g_obs, b_query)
        @test val[1] ≈ expected[1]
    end

    @testset "Correctness: uncertainty_gp" begin
        b_query = 0.5

        # Manual computation: k(b, b) - k(b, b0)' * Σ0⁻¹ * k(b0, b)
        kb_b0 = rgp1.gp.kernel.(Ref(b_query), b0)
        kb_b = rgp1.gp.kernel(b_query, b_query)
        expected = kb_b - kb_b0' * rgp1.Σ0⁻¹ * kb_b0

        unc = uncertainty_gp(rgp1, b_query)
        @test unc ≈ expected atol = 1.0e-4
        @test unc ≥ 0
    end

    @testset "make_ekf + train" begin
        kf = make_ekf(components, dynamics, measurement, R2)
        for (u, y) in zip(us, ys)
            kf(u, y)
        end
    end

    kf = make_ekf(components, dynamics, measurement, R2)
    for (u, y) in zip(us, ys)
        kf(u, y)
    end

    @testset "measure_kf" begin
        u_test = us[1]
        result = measure_kf(kf, u_test)

        # μ should match calling the measurement function directly
        expected_μ = measurement(kf.x, u_test, kf.p, kf.t)
        @test result.μ ≈ expected_μ

        # Σ should be symmetric and positive semi-definite
        @test result.Σ ≈ result.Σ'
        @test all(eigvals(Symmetric(result.Σ)) .≥ -1.0e-10)
    end

    @testset "predict_gp" begin
        # Predict within the training data range (0.1 to ~0.77)
        us_test = collect(range(0.15, 0.7, length = 50))

        for u in us_test
            pred = predict_gp(kf, u, :rgp1)
            @test pred.σ ≥ 0
        end

        # After training on 100 noisy points, predictions should be close to ground truth
        preds = [predict_gp(kf, u, :rgp1) for u in us_test]
        pred_μ = [p.μ for p in preds]
        @test pred_μ ≈ f.(us_test) atol = 0.01
    end

    @testset "state/covariance accessors" begin
        cx = ComponentVector(kf.x, kf.p.xid)
        cR = ComponentMatrix(kf.R, kf.p.Σid)

        @test LowLevelParticleFilters.state(kf, :rgp1) ≈ cx.rgp1
        @test LowLevelParticleFilters.covariance(kf, :rgp1) ≈ cR[:rgp1, :rgp1]
    end
end


@testset "Combined KF" begin
    rng = Xoshiro(123)

    b0 = collect(0:0.05:1)
    kernel1 = 0.02 * with_lengthscale(SEKernel(), 0.1)

    rgp1 = RGP(kernel1, b0)
    rgp2 = RGP(kernel1, b0)

    components = (; a = rgp1, b = rgp2)

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
        R1 + u[2]^2 * R2 |> SMatrix{1, 1}
    end

    # Dataset: y = f1(b) + i * f2(b)
    f1(b) = exp(b)
    f2(b) = 0.1 + 0.5 * b + 0.1 * sinpi(b * 2)
    n = 100
    bs = 0.1 .+ rand(rng, n) / 1.5
    is = 0.2 .* randn(rng, n)
    ys_raw = @. f1(bs) + is * f2(bs)
    ys = [SA[y] for y in ys_raw]
    us = [[b, i] for (b, i) in zip(bs, is)]

    @testset "Instantiation + type promotion" begin
        @testset "No dual" begin
            @test_nowarn make_ekf(components, dynamics, measurement, R2)
            kf = make_ekf(components, dynamics, measurement, R2)
            @test eltype(kf.d0.μ) <: Float64
            @test eltype(kf.d0.Σ) <: Float64
            @test eltype(kf.R1) <: Float64
        end
    end

    kf = make_ekf(components, dynamics, measurement, R2)
    xid = kf.p.xid
    Σid = kf.p.Σid

    @testset "Mean, Cov and R1 consistency" begin
        cx = ComponentVector(kf.x, xid)
        cR = ComponentMatrix(kf.R, Σid)
        cR1 = ComponentMatrix(kf.R1, Σid)

        @test cx.a ≈ rgp1.μ0
        @test cx.b ≈ rgp2.μ0

        @test cR[:a, :a] ≈ rgp1.Σ0
        @test cR[:b, :b] ≈ rgp2.Σ0

        @test cR1[:a, :a] ≈ rgp1.R1
        @test cR1[:b, :b] ≈ rgp2.R1
    end

    @testset "state/covariance accessors" begin
        cx = ComponentVector(kf.x, xid)
        cR = ComponentMatrix(kf.R, Σid)

        @test LowLevelParticleFilters.state(kf, :a) ≈ cx.a
        @test LowLevelParticleFilters.state(kf, :b) ≈ cx.b

        @test LowLevelParticleFilters.covariance(kf, :a) ≈ cR[:a, :a]
        @test LowLevelParticleFilters.covariance(kf, :b) ≈ cR[:b, :b]
    end

    @testset "Train" begin
        for (u, y) in zip(us, ys)
            kf(u, y)
        end
    end

    @testset "measure_kf" begin
        u_test = us[1]
        result = measure_kf(kf, u_test)

        expected_μ = measurement(kf.x, u_test, kf.p, kf.t)
        @test result.μ ≈ expected_μ

        @test result.Σ ≈ result.Σ'
        @test all(eigvals(Symmetric(result.Σ)) .≥ -1.0e-10)
    end

    @testset "predict_gp" begin
        b_test = collect(range(0.15, 0.7, length = 20))

        preds_a = [predict_gp(kf, b, :a) for b in b_test]
        preds_b = [predict_gp(kf, b, :b) for b in b_test]

        # σ should be non-negative
        @test all(p -> p.σ ≥ 0, preds_a)
        @test all(p -> p.σ ≥ 0, preds_b)

        # Predicted means should be close to ground truth
        pred_μ_a = [p.μ for p in preds_a]
        pred_μ_b = [p.μ for p in preds_b]
        @test pred_μ_a ≈ f1.(b_test) atol = 0.15
        @test pred_μ_b ≈ f2.(b_test) atol = 0.05
    end
end


@testset "Hyperparameter Tuning" begin
    rng = Xoshiro(123)

    softplus(x) = 1 / (1 + exp(-x))
    inv_softplus(x) = log(x / (1 - x))

    # Dataset (same as testset 1)
    f(b) = 0.5 * b + 0.1 * sinpi(b * 2)
    n = 100
    bs = 0.1 .+ rand(rng, n) / 1.5
    ys_raw = f.(bs) .+ 5.0e-3 .* randn(rng, n)
    ys = [SA[y] for y in ys_raw]
    us = [b for b in bs]

    function build_kf(θ, ϑ)
        b0 = collect(range(0, 1, length = ϑ.n_basis))
        gp = GP(ConstMean(ϑ.mean), θ.σ * with_lengthscale(SEKernel(), θ.ℓ))
        rgp1 = RGP(gp, b0)
        components = (; rgp1)

        dynamics(x, u, p, t) = x

        measurement(x, u, p, t) = [measurement_gp(p.rgp1, x, u)]

        R2(x, u, p, t) = [θ.R2]

        make_ekf(components, dynamics, measurement, R2)
    end

    function loss_function(θ, p)
        (; ϑ, us, ys) = p
        θ_ = softplus.(θ)
        kf = build_kf(θ_, ϑ)

        cost = 0.0
        for (u, y) in zip(us, ys)
            ll, e = correct!(kf, u, y, kf.p)
            predict!(kf, u)
            cost += dot(e, 1, e)
        end
        cost
    end

    ϑ = (; n_basis = 20, mean = 0.0)
    p = (; ϑ, us, ys)

    θ0 = ComponentVector(; σ = 0.2, ℓ = 0.8, R2 = 0.1^2)
    θ0 = inv_softplus.(θ0)

    adtype = AutoForwardDiff()
    fs = OptimizationFunction(loss_function, adtype)
    prob = OptimizationProblem(fs, θ0, p)
    alg = LBFGS(linesearch = LineSearches.BackTracking())

    @testset "Loss decreases" begin
        initial_cost = loss_function(θ0, p)
        sol = solve(prob, alg, reltol = 1.0e-4, show_trace = false, maxiters = 10)
        @test sol.objective < initial_cost
    end

    @testset "ForwardDiff compatibility" begin
        g = ForwardDiff.gradient(θ -> loss_function(θ, p), θ0)
        @test length(g) == length(θ0)
        @test all(isfinite, g)
    end
end

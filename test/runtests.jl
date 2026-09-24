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

    # Dataset: f(b) = 0.5b + 0.1*sinpi(2b), n=100, with noise
    f(b) = 0.5 * b + 0.1 * sinpi(b * 2)
    n = 100
    us = 0.1 .+ rand(rng, n) / 1.5
    ys_raw = f.(us) .+ 5.0e-3 .* randn(rng, n)
    ys = [SA[y] for y in ys_raw]

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

    @testset "KernelSum" begin
        # `cov!` has a KernelSum specialization; it must not be ambiguous with the
        # generic method, and must agree with it numerically.
        ksum = 0.02 * with_lengthscale(SEKernel(), 0.1) + 0.05 * with_lengthscale(SEKernel(), 0.5)
        rgp_s = RGP(ksum, b0)
        g_obs = rand(rng, length(b0))
        b_query = 0.42

        @test_nowarn measurement_gp(rgp_s, g_obs, b_query)
        @test_nowarn uncertainty_gp(rgp_s, b_query)

        expected = ksum.(b0, b_query)' * rgp_s.Σ0⁻¹ * (g_obs - rgp_s.μ0)
        @test measurement_gp(rgp_s, g_obs, b_query)[1] ≈ expected atol = 1.0e-8
        @test uncertainty_gp(rgp_s, b_query)[1] ≥ 0

        # must stay differentiable w.r.t. the query point (needed for non-identity dynamics)
        @test_nowarn ForwardDiff.derivative(b -> measurement_gp(rgp_s, g_obs, b)[1], b_query)
    end

    @testset "Differentiable w.r.t. kernel hyperparameters" begin
        # Σ0⁻¹ carries Duals here; the cache element type must still be concrete
        g_obs = rand(rng, length(b0))
        # larger jitter keeps Σ0 well conditioned, so finite differences are a reliable reference
        make_rgp = θ -> RGP(θ[1] * with_lengthscale(SEKernel(), θ[2]), b0, 1.0e-3)
        output = θ -> measurement_gp(make_rgp(θ), g_obs, 0.42)[1] + uncertainty_gp(make_rgp(θ), 0.42)[1]
        θ = [0.02, 0.3]

        grad = ForwardDiff.gradient(output, θ)
        δ = i -> 1.0e-4 * θ[i] .* (eachindex(θ) .== i)
        fdiff = [(output(θ .+ δ(i)) - output(θ .- δ(i))) / (2 * 1.0e-4 * θ[i]) for i in eachindex(θ)]
        @test all(isfinite, grad)
        @test grad ≈ fdiff rtol = 1.0e-6

        # hyperparameters and query point differentiated together
        slope = θ -> ForwardDiff.derivative(b -> measurement_gp(make_rgp(θ), g_obs, b)[1], 0.42)
        @test all(isfinite, ForwardDiff.gradient(slope, θ))
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

    @testset "ExtendedKalmanFilter + train" begin
        kf = ExtendedKalmanFilter(rgp1)
        for (u, y) in zip(us, ys)
            kf(u, y)
        end
    end

    kf = ExtendedKalmanFilter(rgp1)
    for (u, y) in zip(us, ys)
        kf(u, y)
    end

    @testset "predict_gp" begin
        u_test = [us[1]]
        result = predict_gp(kf, u_test)

        # μ should match calling the measurement function directly
        expected_μ = measurement_gp(rgp1, kf.x, us[1])
        @test result.μ ≈ [expected_μ]

        # Σ should be positive semidefinite
        @test result.Σ[1, 1] ≥ 0
    end

    @testset "predict_kf" begin
        # Predict within the training data range (0.1 to ~0.77)
        us_test = collect(range(0.15, 0.7, length = 50))

        # After training on 100 noisy points, predictions should be close to ground truth
        preds = [predict_kf(kf, u) for u in us_test]
        pred_μ = [first(p.μ) for p in preds]
        @test all(isapprox.(pred_μ, f.(us_test), atol = 0.005))
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

    kf = ExtendedKalmanFilter(components, dynamics, measurement, R2)
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

    for (u, y) in zip(us, ys)
        kf(u, y)
    end

    @testset "predict_gp" begin

        u_test = collect(range(0.15, 0.7, length = 20))

        pred_a = predict_gp(kf, u_test, :a)
        pred_b = predict_gp(kf, u_test, :b)

        # Σ diagonal should be non-negative (positive semidefinite)
        @test all(diag(pred_a.Σ) .≥ 0)
        @test all(diag(pred_b.Σ) .≥ 0)

        # Predicted means should be close to ground truth
        @test all(isapprox.(pred_a.μ, f1.(u_test); atol = 0.001))
        @test all(isapprox.(pred_b.μ, f2.(u_test); atol = 0.01))
    end

    @testset "predict_kf" begin
        u_test = us[1]
        result = predict_kf(kf, u_test)

        expected_μ = measurement(kf.x, u_test, kf.p, kf.t)
        @test result.μ ≈ expected_μ

        @test result.Σ ≈ result.Σ'
        @test all(eigvals(Symmetric(result.Σ)) .≥ -1.0e-10)
    end
end


@testset "Hyperparameter Tuning (single RGP)" begin
    rng = Xoshiro(123)

    softplus(x) = 1 / (1 + exp(-x))
    inv_softplus(x) = log(x / (1 - x))

    # Dataset
    f(b) = 0.5 * b + 0.1 * sinpi(b * 2)
    n = 100
    us = 0.1 .+ rand(rng, n) / 1.5
    ys_raw = f.(us) .+ 5.0e-3 .* randn(rng, n)
    ys = [SA[y] for y in ys_raw]

    function build_kf_single(θ, ϑ)
        b0 = collect(range(0, 1, length = ϑ.n_basis))
        gp = GP(ConstMean(ϑ.mean), θ.σ * with_lengthscale(SEKernel(), θ.ℓ))
        rgp = RGP(gp, b0)
        ExtendedKalmanFilter(rgp)
    end

    function loss_single(θ, p)
        (; ϑ, us, ys) = p
        θ_ = softplus.(θ)
        kf = build_kf_single(θ_, ϑ)

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

    θ0 = ComponentVector(; σ = 0.2, ℓ = 0.8)
    θ0 = inv_softplus.(θ0)

    adtype = AutoForwardDiff()
    fs = OptimizationFunction(loss_single, adtype)
    prob = OptimizationProblem(fs, θ0, p)
    alg = LBFGS(linesearch = LineSearches.BackTracking())

    @testset "Loss decreases" begin
        initial_cost = loss_single(θ0, p)
        sol = solve(prob, alg, reltol = 1.0e-4, show_trace = false, maxiters = 10)
        @test sol.objective < initial_cost
    end

    @testset "ForwardDiff compatibility" begin
        g = ForwardDiff.gradient(θ -> loss_single(θ, p), θ0)
        @test length(g) == length(θ0)
        @test all(isfinite, g)
    end

    @testset "Optimized predictions" begin
        sol = solve(prob, alg, reltol = 1.0e-4, show_trace = false)
        θ_opt = softplus.(sol.u)
        kf = build_kf_single(θ_opt, ϑ)

        for (u, y) in zip(us, ys)
            kf(u, y)
        end

        us_test = collect(range(0.15, 0.7, length = 50))
        pred = predict_gp(kf, us_test)
        @test all(isapprox.(pred.μ, f.(us_test); atol = 0.01))
    end
end


@testset "Hyperparameter Tuning (combined RGPs)" begin
    rng = Xoshiro(123)

    softplus(x) = 1 / (1 + exp(-x))
    inv_softplus(x) = log(x / (1 - x))

    # Dataset: y = f1(b) + i * f2(b)
    f1(b) = exp(b)
    f2(b) = 0.1 + 0.5 * b + 0.1 * sinpi(b * 2)
    n = 100
    bs = 0.1 .+ rand(rng, n) / 1.5
    is = 0.2 .* randn(rng, n)
    ys_raw = @. f1(bs) + is * f2(bs)
    ys = [SA[y] for y in ys_raw]
    us = [[b, i] for (b, i) in zip(bs, is)]

    function build_kf_combined(θ, ϑ)
        b0 = collect(range(0, 1, length = ϑ.n_basis))
        rgp_a = RGP(θ.σ_a * with_lengthscale(SEKernel(), θ.ℓ_a), b0)
        rgp_b = RGP(θ.σ_b * with_lengthscale(SEKernel(), θ.ℓ_b), b0)
        components = (; a = rgp_a, b = rgp_b)

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

        ExtendedKalmanFilter(components, dynamics, measurement, R2)
    end

    function loss_combined(θ, p)
        (; ϑ, us, ys) = p
        θ_ = softplus.(θ)
        kf = build_kf_combined(θ_, ϑ)

        cost = 0.0
        for (u, y) in zip(us, ys)
            ll, e = correct!(kf, u, y, kf.p)
            predict!(kf, u)
            cost += dot(e, 1, e)
        end
        cost
    end

    ϑ = (; n_basis = 20)
    p = (; ϑ, us, ys)

    θ0 = ComponentVector(; σ_a = 0.2, ℓ_a = 0.8, σ_b = 0.2, ℓ_b = 0.8)
    θ0 = inv_softplus.(θ0)

    adtype = AutoForwardDiff()
    fs = OptimizationFunction(loss_combined, adtype)
    prob = OptimizationProblem(fs, θ0, p)
    alg = LBFGS(linesearch = LineSearches.BackTracking())

    @testset "Loss decreases" begin
        initial_cost = loss_combined(θ0, p)
        sol = solve(prob, alg, reltol = 1.0e-4, show_trace = false, maxiters = 10)
        @test sol.objective < initial_cost
    end

    @testset "ForwardDiff compatibility" begin
        g = ForwardDiff.gradient(θ -> loss_combined(θ, p), θ0)
        @test length(g) == length(θ0)
        @test all(isfinite, g)
    end

    @testset "Optimized predictions" begin
        sol = solve(prob, alg, reltol = 1.0e-4, show_trace = false)
        θ_opt = softplus.(sol.u)
        kf = build_kf_combined(θ_opt, ϑ)

        for (u, y) in zip(us, ys)
            kf(u, y)
        end

        u_test = collect(range(0.15, 0.7, length = 20))
        pred_a = predict_gp(kf, u_test, :a)
        pred_b = predict_gp(kf, u_test, :b)

        @test all(isapprox.(pred_a.μ, f1.(u_test); atol = 0.01))
        @test all(isapprox.(pred_b.μ, f2.(u_test); atol = 0.01))
    end
end

# Extends AbstractGPs to evaluate `mean` and `cov` of a GP to single values (instead of `Vector`s only)
mean_value(m::ZeroMean, x::Real) = zero(x)
mean_value(m::ConstMean, x::Real) = m.c
mean_value(m::CustomMean, x::Real) = m.f(x)

Statistics.mean(gp::GP, x::Real) = mean_value(gp.mean, x)

Statistics.cov(gp::GP, x::AbstractVector, y::Real) = gp.kernel.(x, y)
Statistics.cov(gp::GP, x::Real, y::AbstractVector) = gp.kernel.(x, y)'
Statistics.cov(gp::GP, x::Real) = kernelmatrix(gp.kernel, x)

function cov!(c::AbstractVector, gp::GP, x::AbstractVector, y::Real)
    @. c = gp.kernel(x, y)
    return
end

function cov!(c::AbstractVector, gp::GP{<:AbstractGPs.MeanFunction, <:KernelSum}, x::AbstractVector, y::Real)
    fill!(c, zero(eltype(c)))
    for kernel in gp.kernel.kernels
        @. c += kernel(x, y)
    end
    return
end

"""
    struct RGP{bT, mT, BT, RT, cT}

Recursive Gaussian process [1]: a GP prior represented by its values at the basis
points `b0`. The values at `b0` are the state of a Kalman filter, with prior mean
``\\mu_0 = m(b_0)`` and covariance ``\\Sigma_0 = K_{b_0 b_0} + \\varepsilon I``. See
[Mathematical Background](@ref) for the derivation.

# Fields
- `gp`: The underlying `AbstractGPs.GP` object.
- `b0`: The basis points (input locations) defining the reference distribution.
- `μ0`: Initial mean vector at `b0`.
- `Σ0`: Initial covariance matrix at `b0`, including the jitter ``\\varepsilon``.
- `Σ0⁻¹`: Pre-computed inverse of `Σ0`, used for the interpolation weights ``H(u)``.
- `R1`: Process noise matrix (initialized to zeros, i.e. a function constant in time).
- `cache`: A `NamedTuple` containing `DiffCache` arrays (from `PreallocationTools.jl`).

# References
[1] M. F. Huber, "Recursive Gaussian process: On-line regression and learning,"
Pattern Recognition Letters, vol. 45, pp. 85-91, 2014,
doi: [10.1016/j.patrec.2014.03.004](https://doi.org/10.1016/j.patrec.2014.03.004)
"""
struct RGP{bT, mT, BT, RT, cT}
    gp::GP
    b0::bT
    μ0::mT
    Σ0::BT
    Σ0⁻¹::BT
    R1::RT
    cache::cT
end

"""
RGP(gp::GP, b0::AbstractArray)
RGP(gp::GP, b0::AbstractArray, cov_jitter=1e-6)
"""
function RGP(gp::GP, b0::T, cov_jitter = 1.0e-6) where {T <: AbstractArray}
    nb = length(b0) # 1-dim basis vector (for now)

    # pre-compute for the basis points: mean vector, covariance matrix
    # and inverse covariance matrix (adding a generic `1e-6` jitter for stability)
    μ0 = mean(gp, b0)
    Σ0 = cov(gp, b0) + cov_jitter * I
    Σ0⁻¹ = inv(Σ0)

    R1 = zeros(nb, nb)

    # initialize `DiffCache` buffers with basis vector size.
    csize = ForwardDiff.pickchunksize(length(b0) + 2)
    cache = (;
        k = DiffCache(similar(b0), csize),
        k⁻ = DiffCache(similar(b0), csize),
        H = DiffCache(similar(b0'), csize),
        Δg = DiffCache(similar(b0), csize),
    )

    return RGP(gp, b0, μ0, Σ0, Σ0⁻¹, R1, cache)
end

"""
RGP(kernel::Kernel, b0::AbstractArray)
RGP(kernel::Kernel, b0::AbstractArray, cov_jitter=1e-6)
"""
function RGP(kernel::Kernel, b0::AbstractArray, cov_jitter = 1.0e-6)
    gp = GP(kernel)
    return RGP(gp, b0, cov_jitter)
end

"""
RGP(mean, kernel::Kernel, b0::AbstractArray)
RGP(mean, kernel::Kernel, b0::AbstractArray, cov_jitter=1e-6)
"""
function RGP(mean, kernel::Kernel, b0::AbstractArray, cov_jitter = 1.0e-6)
    gp = GP(mean, kernel)
    return RGP(gp, b0, cov_jitter)
end


"""
    measurement_gp(rgp::RGP, g::AbstractArray, b::Real)

Mean of the function at input `b`, given the values `g` at the basis points:

```math
\\mathbb{E}[f(b) \\mid g] = m(b) + H(b)\\,(g - \\mu_0), \\qquad H(b) = K_{b b_0}\\,\\Sigma_0^{-1}
```

This is the measurement function of an RGP observation. See
[Function values from the basis values](@ref) in the Mathematical Background.
"""
function measurement_gp(rgp::RGP, g::AbstractArray, b::Real)
    (; gp, b0, μ0, Σ0⁻¹, cache) = rgp
    T = promote_type(eltype(Σ0⁻¹), typeof(b))
    k = get_tmp(cache.k, T)
    H = get_tmp(cache.H, T)
    Δg = get_tmp(cache.Δg, g)

    # (cov(gp, b, b0) * Σ0⁻¹) * (g - μ0) + mean(gp, b)
    #        k                    Δg
    #                H

    cov!(k, gp, b0, b) # k = cov(gp, b, b0)
    mul!(H, k', Σ0⁻¹) # H = k' * Σ0⁻¹
    Δg .= g - μ0
    # Δg .= g .- μ0
    return muladd(H, Δg, mean(gp, b)) # H * (g - μ0) + m
end


"""
    uncertainty_gp(rgp::RGP, b::Real)

Residual variance of the function at input `b`, given the values at the basis points:

```math
r(b) = k(b, b) - H(b)\\,K_{b_0 b}
```

It is zero at the basis points and grows between them. Add it to the sensor noise
variance to obtain the measurement noise ``R_2`` of an RGP observation. See
[Observation model](@ref) in the Mathematical Background.
"""
function uncertainty_gp(rgp::RGP, b::Real)
    (; gp, b0, Σ0⁻¹, cache) = rgp
    T = promote_type(eltype(Σ0⁻¹), typeof(b))
    k = get_tmp(cache.k, T)
    H = get_tmp(cache.H, T)
    k⁻ = get_tmp(cache.k⁻, T)

    cov!(k, gp, b0, b) # k = cov(gp, b, b0)
    mul!(H, k', Σ0⁻¹) # H = k' * Σ0⁻¹
    @. k⁻ = -k
    return muladd(H, k⁻, gp.kernel(b, b)) # kb - H * k
end

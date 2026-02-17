# Extends AbstractGPs to evaluate `mean` and `cov` of a GP to single values (instead of `Vector`s only)
mean_value(m::ZeroMean, x::Real) = zero(x)
mean_value(m::ConstMean, x::Real) = m.c
mean_value(m::CustomMean, x::Real) = m.f(x)

Statistics.mean(gp::GP, x::Real) = mean_value(gp.mean, x)

Statistics.cov(gp::GP, x::AbstractVector, y::Real) = gp.kernel.(x, y)
Statistics.cov(gp::GP, x::Real, y::AbstractVector) = gp.kernel.(x, y)'
Statistics.cov(gp::GP, x::Real) = kernelmatrix(gp.kernel, x)

function cov!(c::AbstractVector, gp::GP, x::AbstractVector, y::Real)
    return @. c = gp.kernel(x, y)
end

function cov!(c::AbstractVector, gp::GP{<:Any, <:KernelSum}, x::AbstractVector, y)
    fill!(c, 0.0)
    for kernel in gp.kernel.kernels
        @. c += kernel(x, y)
    end
    return
end

"""
    struct RGP{bT, mT, BT, RT, cT}

A "Recursive Gaussian Process" structure.
Available constructors are:

    RGP(gp::GP, b0::AbstractArray)
    RGP(kernel::Kernel, b0::AbstractArray)
    RGP(mean, kernel::Kernel, b0::AbstractArray)

# Fields
- `gp`: The underlying `AbstractGPs.GP` object.
- `b0`: The basis points (input locations) defining the reference distribution.
- `μ0`: Initial mean vector at `b0`.
- `Σ0`: Initial covariance matrix at `b0`.
- `Σ0⁻¹`: Pre-computed inverse of `Σ0` (used for calculating the Kalman Gain/Projection matrix).
- `R1`: Process noise matrix (initialized to zeros).
- `cache`: A `NamedTuple` containing `DiffCache` arrays (from `PreallocationTools.jl`).
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

function RGP(gp::GP, b0::T) where {T <: AbstractArray}
    nb = length(b0) # 1-dim basis vector (for now)

    # pre-compute for the basis points: mean vector, covariance matrix
    # and inverse covariance matrix (adding a generic `1e-6` jitter for stability)
    μ0 = mean(gp, b0) #|> T
    # μ0 = SVector{nb}(mean(gp, b0))
    Σ0 = cov(gp, b0) + 1.0e-6I
    Σ0⁻¹ = inv(Σ0)

    R1 = zeros(nb, nb)

    # initialize `DiffCache` buffers with basis vector size.
    csize = ForwardDiff.pickchunksize(length(b0) + 2)
    cache = (;
        k = DiffCache(similar(b0), csize),
        k⁻ = DiffCache(similar(b0), csize),
        H = DiffCache(similar(b0'), csize),
        Δg = DiffCache(similar(b0), csize), # <- use DiffCache #
    )

    return RGP(gp, b0, μ0, Σ0, Σ0⁻¹, R1, cache)
end

function RGP(kernel::Kernel, b0::AbstractArray)
    gp = GP(kernel)
    return RGP(gp, b0)
end

function RGP(mean, kernel::Kernel, b0::AbstractArray)
    gp = GP(mean, kernel)
    return RGP(gp, b0)
end


"""
    measurement_gp(rgp::RGP, g::AbstractArray, b::Real)

Calculate the conditional mean of the Gaussian Process at a new point `b`.
See [1] for a detailed description on recursive GPs.

```math
\\mu_{post} = m(b) + k(b, b_0) \\Sigma_0^{-1} (g - \\mu_0)
```

[1] - M. F. Huber, "Recursive Gaussian process: On-line regression and learning,"
Pattern Recognition Letters, 2014, doi: 10.1016/j.patrec.2014.03.004
"""
function measurement_gp(rgp::RGP, g::AbstractArray, b::Real)
    (; gp, b0, μ0, Σ0⁻¹, cache) = rgp
    # (; k, H) = cache
    # Δg = get_tmp(cache.Δg, g)
    T = eltype(Σ0⁻¹) <: ForwardDiff.Dual ? ForwardDiff.Dual : typeof(b)
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

Calculates the conditional variance (uncertainty) of the Gaussian Process at a point `b`.
This represents uncertainty at `b` conditioned on the basis points `b0`.
See [1] for a detailed description on recursive GPs.

```math
    \\sigma^2_{post} = k(b, b) - k(b, b_0) \\Sigma_0^{-1} k(b_0, b)
```

[1] - M. F. Huber, "Recursive Gaussian process: On-line regression and learning,"
Pattern Recognition Letters, 2014, doi: 10.1016/j.patrec.2014.03.004
"""
function uncertainty_gp(rgp::RGP, b::Real)
    (; gp, b0, Σ0⁻¹, cache) = rgp
    T = eltype(Σ0⁻¹) <: ForwardDiff.Dual ? ForwardDiff.Dual : typeof(b)
    k = get_tmp(cache.k, T)
    H = get_tmp(cache.H, T)
    k⁻ = get_tmp(cache.k⁻, T)

    cov!(k, gp, b0, b) # k = cov(gp, b, b0)
    mul!(H, k', Σ0⁻¹) # H = k' * Σ0⁻¹
    @. k⁻ = -k
    return muladd(H, k⁻, gp.kernel(b, b)) # kb - H * k
end

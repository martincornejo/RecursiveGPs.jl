"""
    ExtendedKalmanFilter(rgp::RGP; σn=0.0, ny=1, nu=1, p=(;), kwargs...)

Construct an `ExtendedKalmanFilter` for a single [`RGP`](@ref) model.

The state is the GP function values at the basis points `b0`. Dynamics is set to
identity and the measurement model uses [`measurement_gp`](@ref). The measurement
noise is the residual variance of the GP, [`uncertainty_gp`](@ref), plus the sensor
noise variance `σn^2`.

# Arguments
- `rgp`: The [`RGP`](@ref) model.
- `σn`: Standard deviation of the sensor noise. The default `0.0` treats the
  observations as noise-free.
- `ny`, `nu`: Output and input dimensions.
- `p`: Additional parameters merged into the filter's parameter tuple.
- `kwargs...`: Forwarded to the base `ExtendedKalmanFilter` constructor.
"""
function ExtendedKalmanFilter(rgp::RGP; σn::Real = 0.0, ny::Int = 1, nu::Int = 1, p::NamedTuple = (;), kwargs...)

    dynamics(x, u, p, t) = x # identity

    measurement(x, u, p, t) = measurement_gp(p.rgp, x, u) |> SVector{ny}

    R2(x, u, p, t) = uncertainty_gp(p.rgp, u) + σn^2 |> SMatrix{ny, ny}

    T = promote_type(eltype(rgp.μ0), eltype(rgp.Σ0), eltype(rgp.R1))
    μ0 = T.(rgp.μ0)
    R1 = T.(rgp.R1)

    d0 = LLPF.SimpleMvNormal(μ0, rgp.Σ0)
    nx = length(μ0)

    p = (; rgp, p...)

    return ExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; p, nx, nu, ny, kwargs...)
end


"""
    predict_kf(kf, u, x=state(kf), R=covariance(kf), p=kf.p, t=index(kf))

Calculate the predicted measurement and innovation covariance.

# Arguments
- `kf`: The Extended Kalman Filter.
- `u`: The control input.
- `x`: State estimate.
- `R`: Covariance matrix.
- `p`: Additional parameters passed to the filter.
- `t`: Current time index.

# Returns
A named tuple `(;μ, Σ)` where:
- `μ`: The expected measurement ``h(x^-, u, p, t)``.
- `Σ`: The innovation covariance ``C \\Sigma^- C^T + R_2``.
"""
function predict_kf(kf::LLPF.AbstractExtendedKalmanFilter, u, x = state(kf), R = covariance(kf), p = kf.p, t = index(kf))
    measurement_model = kf.measurement_model
    return predict_kf(measurement_model, u, x, R, p, t)
end

function predict_kf(measurement_model::EKFMeasurementModel{IPM}, u, x, R, p, t) where {IPM}
    (; measurement, Cjac, ny) = measurement_model
    C = Cjac(x, u, p, t)
    R2 = LLPF.get_mat(measurement_model.R2, x, u, p, t)

    if IPM
        μ = zeros(ny)
        measurement(μ, x, u, p, t)
    else
        μ = measurement(x, u, p, t)
    end

    Σ = LLPF.symmetrize(C * R * C') + R2

    return (; μ, Σ)
end


"""
    predict_gp(kf, b::AbstractVector, x = state(kf), P = covariance(kf))

Posterior of the function at the query points `b`, for a filter built with
`ExtendedKalmanFilter(rgp)`. With interpolation weights
``H^* = K_{b b_0}\\,\\Sigma_0^{-1}``, filter mean ``\\hat g = x`` and covariance ``P``:

```math
\\begin{aligned}
\\mu^* &= m(b) + H^*(\\hat g - \\mu_0) \\\\
\\Sigma^* &= H^* P H^{*\\top} + K_{bb} - H^* K_{b_0 b}
\\end{aligned}
```

The first term of ``\\Sigma^*`` is the uncertainty of the basis values, the second the
residual between basis points. ``\\Sigma^*`` excludes sensor noise. See
[Prediction at new inputs](@ref) in the Mathematical Background.

Returns a `NamedTuple` `(; μ, Σ)`.
"""
function predict_gp(kf, b::AbstractArray, x::AbstractArray = state(kf), R::AbstractMatrix = covariance(kf))
    (; gp, b0, μ0, Σ0⁻¹) = kf.p.rgp

    H = cov(gp, b, b0) * Σ0⁻¹
    m = mean(gp, b)
    μ = H * (x - μ0) + m

    R2 = cov(gp, b) - H * cov(gp, b0, b) # residual covariance between basis points
    Σ = R2 + H * R * H' # plus uncertainty of the basis values
    return (; μ, Σ)
end

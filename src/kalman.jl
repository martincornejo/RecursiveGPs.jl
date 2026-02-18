function make_ekf(rgp::RGP; ny::Int = 1, nu::Int = 1, p::NamedTuple = (;))

    dynamics(x, u, p, t) = x # identity
    # Ajac = I(nx)

    measurement(x, u, p, t) = measurement_gp(p.rgp, x, u) |> SVector{ny}

    R2(x, u, p, t) = uncertainty_gp(p.rgp, u) |> SMatrix{ny, ny}

    R1 = rgp.R1 # SA?

    d0 = LLPF.SimpleMvNormal(rgp.μ0, rgp.Σ0)
    nx = length(rgp.μ0)

    p = (; rgp, p...)

    return ExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; p, nx, nu, ny)
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
    predict_gp(kf, b::Real, x::AbstractArray, R::AbstractMatrix)

Core implementation of the GP-KF projection at a single point `b`

# Arguments
- `kf`: The Extended Kalman Filter.
- `b`: The scalar input point.
- `x`: State vector, default internal `state(kf)``.
- `R`: Sate covariance matrix, defaul internal `covariance(kf)`.

# Returns
A `NamedTuple` `(; μ, σ)` containing:
- `μ`: The projected mean.
- `σ`: The projected standard deviation.

# Mathematical Details
The prediction accounts for both the GP's intrinsic uncertainty and the filter's state uncertainty:
1. **Gain**: ``H = cov(gp, b, b_0) \\Sigma_0^{-1}``
2. **Mean**: ``\\mu = H(x' - \\mu_0) + m(b)``
3. **Covariance**: ``\\Sigma = R_2 + H R' H^T``

Where ``R_2`` is the GP conditional variance.

# References
 - M. F. Huber, "Recursive Gaussian process regression," 2013 IEEE International Conference on Acoustics, Speech and Signal Processing, Vancouver, BC, Canada, 2013, pp. 3362-3366, doi: 10.1109/ICASSP.2013.6638281.
"""
function predict_gp(kf, b::Real, x::AbstractArray = state(kf), R::AbstractMatrix = covariance(kf))
    (; gp, b0, μ0, Σ0⁻¹) = kf.p.rgp

    H = cov(gp, b, b0) * Σ0⁻¹
    m = mean(gp, b)
    μ = H * (x - μ0) + m

    R2 = gp.kernel(b, b) - H * cov(gp, b0, b) #eq.7
    Σ = R2 + H * R * H' #eq.9
    σ = sqrt(Σ)
    return (; μ, σ)
end

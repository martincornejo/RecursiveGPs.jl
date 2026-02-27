"""
    ExtendedKalmanFilter(components, dynamics, measurement, R2; Ajac=nothing, Cjac=nothing, p=(;), ny=1, nu=1, kwargs...)

Construct an `ExtendedKalmanFilter` with a structured, named state representation.

The state is a `ComponentVector` formed by concatenating the `μ0` vectors of each
component. The initial covariance and process noise are block-diagonal matrices built
from each component's `Σ0` and `R1`. Each component is stored in the filter's parameter
tuple under its key, alongside `xid` and `Σid` axes for slicing with [`state`](@ref) and
[`covariance`](@ref).

# Arguments
- `components`: A `NamedTuple` keyed by component ID. Each value must have fields
  `μ0` (initial mean vector), `Σ0` (initial covariance matrix), and `R1` (process noise
  matrix). Typical values are [`RGP`](@ref) models, but any struct with these fields works.
- `dynamics`: State transition function `f(x, u, p, t)`.
- `measurement`: Observation function `h(x, u, p, t)`.
- `R2`: Measurement noise covariance function `R2(x, u, p, t)`.
- `Ajac`, `Cjac`: Optional Jacobians (see base `ExtendedKalmanFilter`).
- `p`: Additional parameters merged into the filter's parameter tuple.
- `ny`, `nu`: Output and input dimensions.
- `kwargs...`: Forwarded to the base `ExtendedKalmanFilter` constructor.
"""
function ExtendedKalmanFilter(components::NamedTuple, dynamics, measurement::Function, R2::Function; p::NamedTuple = (;), ny::Int64 = 1, nu::Int64 = 1, kwargs...)
    ids = keys(components)

    T = mapreduce(c -> promote_type(eltype(c.μ0), eltype(c.Σ0), eltype(c.R1)), promote_type, components; init = Float64)
    x0 = ComponentVector{T}(; (id => components[id].μ0 for id in ids)...)

    Σ0 = zero(T) .* x0 * x0'
    R1 = zero(T) .* x0 * x0'

    for id in ids
        component = components[id]
        Σ0[id, id] = component.Σ0
        R1[id, id] = component.R1
    end

    d0 = LLPF.SimpleMvNormal(x0, Σ0)
    xid = getaxes(x0)
    Σid = getaxes(Σ0)
    nx = length(x0)

    p = (;
        xid,
        Σid,
        components...,
        p...,
    )

    return ExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; nx, nu, ny, p, kwargs...)
end

"""
    state(kf, id::Symbol)

Extracts the mean vector of a specific sub-component `id` from the current filter state.
"""
function LLPF.state(kf, id::Symbol)
    (; xid) = kf.p
    cx = ComponentVector(kf.x, xid)
    return cx[id]
end

"""
    covariance(kf, id::Symbol)

Extracts the covariance sub-matrix of a specific sub-component `id` from the current filter covariance.
Retrieves the diagonal block \$Σ_{id, id}\$ using the saved axes in the filter parameters.
"""
function LLPF.covariance(kf, id::Symbol)
    (; Σid) = kf.p
    cx = ComponentMatrix(kf.R, Σid)
    return cx[id, id]
end


"""
    predict_gp(kf, b::AbstractVector, x::AbstractArray, R::AbstractMatrix, id::Symbol)

GP-KF projection at a vector of query points `b` for component `id`.

# Arguments
- `kf`: The Extended Kalman Filter.
- `b`: Vector of input query points.
- `x`: Full state vector.
- `R`: The current covariance matrix.
- `id`: The component identifier symbol.

# Returns
A `NamedTuple` `(; μ, Σ)` containing:
- `μ`: The projected mean vector of component `id`.
- `Σ`: The projected covariance matrix of component `id`.

# Mathematical Details
The prediction accounts for both the GP's intrinsic uncertainty and the filter's state uncertainty:
1. **Gain**: ``H = cov(gp, b, b_0) \\Sigma_0^{-1}``
2. **Mean**: ``\\mu = H(x' - \\mu_0) + m(b)``
3. **Covariance**: ``\\Sigma = R_2 + H R' H^T``

Where ``R_2`` is the GP conditional variance.

# References
 - M. F. Huber, "Recursive Gaussian process regression," 2013 IEEE International Conference on Acoustics, Speech and Signal Processing, Vancouver, BC, Canada, 2013, pp. 3362-3366, doi: 10.1109/ICASSP.2013.6638281.
"""
function predict_gp(kf, b::AbstractVector, x::AbstractArray, R::AbstractMatrix, id::Symbol)
    (; xid, Σid) = kf.p
    (; gp, b0, μ0, Σ0⁻¹) = kf.p[id]

    cx = ComponentVector(x, xid)
    x´ = cx[id]

    cR = ComponentMatrix(R, Σid)
    R´ = cR[id, id]

    H = cov(gp, b, b0) * Σ0⁻¹
    m = mean(gp, b)
    μ = H * (x´ - μ0) + m

    R2 = cov(gp, b) - H * cov(gp, b0, b) #eq.7
    Σ = R2 + H * R´ * H' #eq.9
    return (; μ, Σ)
end

"""
    predict_gp(kf, b::AbstractVector, id::Symbol)

GP-KF projection at a vector of query points `b` for component `id`.

This wrapper extracts the full state ``x`` and covariance ``R`` from the filter and delegates to the core projection logic.

# Arguments
- `kf`: The Extended Kalman Filter.
- `b`: Vector of input query points.
- `id`: The symbol identifying the GP component in the state vector.

# Returns
A `NamedTuple` `(; μ, Σ)` containing the predicted mean vector and covariance matrix.
"""
function predict_gp(kf, b, id::Symbol)
    x = state(kf)
    R = covariance(kf)
    return predict_gp(kf, b, x, R, id)
end

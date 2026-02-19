"""
    make_ekf(components, dynamics, measurement, R2; Ajac=nothing, Cjac=nothing, p=(;), ny=1, nu=1)

Constructs an `ExtendedKalmanFilter` from `LowLevelParticleFilters.jl` using a structured state representation.

# Arguments
- `components`: A NamedTuple where each key is a state component ID and each value contains `μ0` (initial mean), `Σ0` (initial covariance), and `R1` (process noise).
- `dynamics`: The state transition function \$x_{t+1} = f(x_t, u_t, p, t)\$.
- `measurement`: The observation function \$y_t = h(x_t, u_t, p, t)\$.
- `R2`: A function returning the measurement noise covariance matrix.
- `Ajac`, `Cjac`: Optional Jacobians for the dynamics and measurement models.
- `p`: Additional parameters passed to the filter.
- `ny`: Number of outputs (measurement dimension).
- `nu`: Number of inputs (control dimension).

# Returns
- An `ExtendedKalmanFilter` initialized with a `ComponentVector` state and structured covariance matrices.
"""

function make_ekf(components::NamedTuple, dynamics, measurement::Function, R2::Function; Ajac = nothing, Cjac = nothing, p::NamedTuple = (;), ny::Int64 = 1, nu::Int64 = 1)
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

    return ExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; Ajac, Cjac, nx, nu, ny, p)
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

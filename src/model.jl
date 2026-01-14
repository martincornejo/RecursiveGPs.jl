function make_comb_ekf(components, dynamics, measurement::Function, R2::Function; Ajac=nothing, Cjac=nothing, p::NamedTuple=(;))
    ids = keys(components)
    x0 = ComponentVector(; (id => components[id].μ0 for id in ids)...)

    Σ0 = false .* x0 * x0'
    R1 = false .* x0 * x0'
    
    for id in ids
        component = components[id]
        Σ0[id, id] = component.Σ0
        R1[id, id] = component.R1
    end

    d0 = LLPF.SimpleMvNormal(x0, Σ0)
    xid = getaxes(x0)
    Σid = getaxes(Σ0)

    nx = length(x0)
    nu = 1
    ny = 1

    p = (;
        xid,
        Σid,
        components...,
        p...
    )

    ExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; Ajac, Cjac, nx, nu, ny, p) # , Ajac=fAjac, Cjac=fCjac)
end


function LLPF.state(kf, id::Symbol)
    (; xid) = kf.p
    cx = ComponentVector(kf.x, xid)
    cx[id]
end

function LLPF.covariance(kf, id::Symbol)
    (; Σid) = kf.p
    cx = ComponentMatrix(kf.R, Σid)
    cx[id, id]
end


function measurement_kf(kf::LowLevelParticleFilters.AbstractExtendedKalmanFilter{IPD}, x⁻, Σ⁻, u, p=LowLevelParticleFilters.parameters(kf), t::Real=index(kf); R2=LowLevelParticleFilters.get_mat(kf.measurement_model.R2, x⁻, u, p, t)) where IPD
    (; Cjac, measurement) = kf.measurement_model
    ny = kf.kf.ny
    if false ### TODO: False for now, IPD not working well here
        μ = zeros(ny)
        measurement(μ, x⁻, u, p, t)
    else
        μ = measurement(x⁻, u, p, t)
    end

    C = Cjac(x⁻, u, p, t)
    S = LowLevelParticleFilters.symmetrize(C * Σ⁻ * C') + R2
    (μ, S)
end

function predict_kf!(
    kf,
    u,
)  
    predict!(kf, u)
    μ, S = measurement_kf(kf, kf.x, kf.R, u)
    σ = sqrt.(S)
    (; μ = μ , σ = σ)
end


function predict_gp(kf, b, id::Symbol)
    x = state(kf, id)
    R = covariance(kf, id)
    predict_gp(kf,b,x,R,id)
end


function predict_gp(kf, bs::AbstractArray, x::AbstractArray, R::AbstractMatrix, id::Symbol)
    results = predict_gp.(Ref(kf), bs, Ref(x), Ref(R), id)
    μ = vcat([res.μ for res in results]...)
    σ = vcat([res.σ for res in results]...)
    return (; μ , σ )
end


function predict_gp(kf, b::Real, x::AbstractArray, R::AbstractMatrix, id::Symbol)
    (; xid, Σid) = kf.p
    (; gp, b0, μ0, Σ0⁻¹) = kf.p[id]

    cx = ComponentVector(x, xid)
    x´ = cx[id]

    cR = ComponentMatrix(R, Σid)
    R´ = cR[id, id]

    H = cov(gp, b, b0) * Σ0⁻¹
    m = mean(gp, b)
    μ = H * (x´ - μ0) + m

    R2 = gp.kernel(b,b) - H * cov(gp, b0, b) #eq.7 
    Σ = R2 + H * R´ * H' #eq.9
    σ = sqrt(Σ)
    (; μ = μ,σ =σ)
end


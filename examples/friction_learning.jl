# # Learning Missing Physics Online
#
# A sliding mass is driven by a known force. Its equation of motion is known
# except for the friction force, which is a nonlinear function of velocity:
#
# ```math
# m\,\dot v = F_u(t) - F_f(v), \qquad F_f \text{ unknown}
# ```
#
# An [`RGP`](@ref) represents ``F_f``. Its basis-point values are appended to the
# state vector of an Extended Kalman Filter, so one pass over the data estimates
# the velocity and the friction curve together. No optimiser is involved, and the
# posterior covariance shows which velocities the data constrains.
#
# The setup is adapted from the
# [ModelingToolkitNeuralNets friction tutorial](https://docs.sciml.ai/ModelingToolkitNeuralNets/dev/friction/),
# which learns the same friction law with a neural network.

using RecursiveGPs
using AbstractGPs
using SeeToDee
using StaticArrays
using ComponentArrays
using LinearAlgebra
using LowLevelParticleFilters
using Statistics
using Random
using CairoMakie

# ## The system
#
# The friction law to be recovered is a Stribeck curve: a break-away peak near
# ``v = \pm v_{st}`` on a Coulomb `tanh` saturation, odd in ``v``. It generates
# the data and is not available to the filter.

"""
    stribeck(; Fbrk, vbrk, Fc)

Stribeck friction law. Returns a callable `f(v)`.
"""
function stribeck(; Fbrk = 100.0, vbrk = 10.0, Fc = 80.0)
    vst, vcol = vbrk / 10, vbrk * sqrt(2)
    return v -> sqrt(2 * ℯ) * (Fbrk - Fc) * exp(-(v / vst)^2) * (v / vst) + Fc * tanh(v / vcol)
end

params = (;
    mass     = 1.0,
    dt       = 1.0e-3,     # 1 kHz
    friction = stribeck(),
    σ_v      = 0.05,       # velocity-sensor noise
)

colors = (; truth = "#0173B2", rgp = "#DE8F05", nofric = "#029E73")
rms(a, b) = sqrt(mean(abs2, a .- b))

# The plant, integrated with a zero-order hold on ``F_u``:

function simulate(Fu, tend, p; fric = p.friction, supersample = 10, x0 = SA[0.0, 0.0])
    f(x, u, _p, t) = SA[x[2], (u[1] - fric(x[2])) / p.mass]
    step = SeeToDee.Rk4(f, p.dt; supersample)
    ts = collect(0:p.dt:tend)
    xs = Vector{SVector{2, Float64}}(undef, length(ts))
    xs[1] = x0
    for i in 2:length(ts)
        xs[i] = step(xs[i - 1], SA[Fu(ts[i - 1])], nothing, ts[i - 1])
    end
    return ts, first.(xs), last.(xs)
end

# ## Excitation
#
# The applied force is a multisine, so the velocity changes sign repeatedly.
# Friction is odd in ``v``, so a one-directional experiment constrains only half
# the curve.

Fu(t) = 120.0 * sinpi(2t / 0.6) + 60.0 * sinpi(2t / 0.23 + 0.3)

ts, ss, vs = simulate(Fu, 6.0, params)

Random.seed!(1)
ys = [SA[v + params.σ_v * randn()] for v in vs]
us = [SA[Fu(t)] for t in ts];

# ## The RGP
#
# Basis points span the velocities we expect to visit. The length scale is the
# width of the narrowest feature we want to resolve, here the break-away peak at
# about 1 m/s.

kernel = 60.0^2 * with_lengthscale(SEKernel(), 1.0)
b0 = collect(range(-14.0, 16.0, length = 61))
rgp = RGP(kernel, b0, 1.0e-2)

# ## Coupling the RGP to the physics model
#
# The filter state concatenates the physical state with the GP basis values,
# ``x = [v;\, g]``. `st` is a plain named tuple: the multi-component constructor
# only needs `μ0`, `Σ0` and `R1`, so any struct with those fields works alongside
# an [`RGP`](@ref).
#
# The dynamics call [`measurement_gp`](@ref) to evaluate the learned force at the
# current velocity estimate. `R2` is the velocity-sensor noise, and `q_v` the
# process noise on the velocity state, which also covers the GP's representation
# error.

function build_filter(rgp, p; q_v = 1.0e-6, v0 = 0.0, Σv0 = 1.0)
    components = (;
        st   = (; μ0 = [v0], Σ0 = fill(Σv0, 1, 1), R1 = fill(q_v, 1, 1)),
        fric = rgp,
    )

    function continuous(x, u, q, t)
        xc = ComponentVector(x, q.xid)
        Ff = measurement_gp(q.fric, xc.fric, xc.st[1])[1]
        return vcat((u[1] - Ff) / q.mass, zero(xc.fric))   # the GP block is stationary
    end

    dynamics = SeeToDee.Rk4(continuous, p.dt)
    measurement(x, u, q, t) = SA[ComponentVector(x, q.xid).st[1]]
    R2(x, u, q, t) = @SMatrix [q.σ_v^2]

    return ExtendedKalmanFilter(components, dynamics, measurement, R2; p, nu = 1, ny = 1)
end

kf = build_filter(rgp, params)

# ## Training
#
# A single pass. Each `kf(u, y)` runs one predict-correct cycle. The posterior is
# stored at four times for the plot below.

snap_times = [0.1, 0.3, 1.0, 6.0]
snap_steps = round.(Int, snap_times ./ params.dt) .+ 1   # sample index of each snapshot
snaps = Dict{Int, Any}()

@time for (i, (u, y)) in enumerate(zip(us, ys))
    kf(u, y)
    i in snap_steps && (snaps[i] = (copy(state(kf)), copy(covariance(kf))))
end

# ## What the model knows, as data arrives
#
# The grey rug marks the velocities visited so far. Inside that range the
# posterior follows the truth with a narrow band; outside it the mean returns to
# the prior and the band widens.
#
# After 0.3 s the mass has covered the whole positive range and the fit there is
# already accurate, while negative velocities are still at the prior.

vplot = collect(range(-16.0, 18.0, length = 300))
seen = [vs[1:n] for n in snap_steps]

fig1 = Figure(size = (800, 560))
axs = [CairoMakie.Axis(fig1[cld(i, 2), mod1(i, 2)];
           title = "after $(snap_times[i]) s  ($(snap_steps[i]) samples)")
       for i in eachindex(snap_times)]

for (ax, n, visited) in zip(axs, snap_steps, seen)
    p = predict_gp(kf, vplot, snaps[n]..., :fric)
    σ = sqrt.(abs.(diag(p.Σ)))
    rug = visited[1:max(1, length(visited) ÷ 120):end]
    band!(ax, vplot, p.μ .- 2σ, p.μ .+ 2σ; color = (colors.rgp, 0.25), label = "±2σ")
    lines!(ax, vplot, params.friction.(vplot); color = colors.truth, linewidth = 2, label = "Ground truth")
    lines!(ax, vplot, p.μ; color = colors.rgp, linewidth = 2, label = "RGP posterior mean")
    scatter!(ax, rug, fill(-114.0, length(rug)); marker = '|', markersize = 7, color = (:gray, 0.45))
end

xlims!.(axs, Ref(extrema(vplot)))
ylims!.(axs, -125, 125)
axs[3].xlabel = axs[4].xlabel = "v [m/s]"
axs[1].ylabel = axs[3].ylabel = "Ff [N]"
hidexdecorations!.(axs[1:2]; grid = false)
axislegend(axs[1]; position = :lt, framevisible = false)
fig1

# ## Accuracy
#
# Measured over the velocity range the experiment covered. `coverage_2σ` is the
# fraction of the curve inside the ±2σ band, which should be ``\approx 0.95`` if
# the posterior is well calibrated.

lo, hi = extrema(vs)
vtest = collect(range(lo, hi, length = 200))
post = predict_gp(kf, vtest, :fric)
σtest = sqrt.(abs.(diag(post.Σ)))
err = post.μ .- params.friction.(vtest)

@info "Learned friction" covered_range = (lo, hi) rmse = rms(post.μ, params.friction.(vtest)) coverage_2σ = mean(abs.(err) .<= 2 .* σtest) friction_scale = maximum(abs, params.friction.(vtest))

# ## Re-simulation with the learned term
#
# The posterior mean is inserted into the physical model and the trajectory
# re-simulated. The frictionless model is the starting point the RGP was added to.

ĝ = state(kf, :fric)
f̂(v) = measurement_gp(rgp, ĝ, v)[1]

_, _, v_hybrid = simulate(Fu, 6.0, params; fric = f̂)
_, _, v_nofric = simulate(Fu, 6.0, params; fric = v -> zero(v))

@info "Re-simulation with the completed model" hybrid_rmse = rms(v_hybrid, vs) no_friction_rmse = rms(v_nofric, vs)

fig2 = Figure(size = (800, 600))
axs = [CairoMakie.Axis(fig2[i, 1]; height = 215) for i in 1:2]

lines!(axs[1], ts, first.(us); color = :black, linewidth = 1.5)
lines!(axs[2], ts, v_nofric; color = colors.nofric, linewidth = 2, linestyle = :dash, label = "physics only (no friction)")
lines!(axs[2], ts, vs; color = colors.truth, linewidth = 2, label = "ground truth")
lines!(axs[2], ts, v_hybrid; color = colors.rgp, linewidth = 2, linestyle = :dot, label = "physics + RGP friction")

axs[1].title = "Simulation with the learned friction term"
axs[1].ylabel = "Fu [N]"
axs[2].ylabel = "v [m/s]"
axs[2].xlabel = "t [s]"
xlims!.(axs, Ref(extrema(ts)))
linkxaxes!(axs...)
hidexdecorations!(axs[1]; grid = false)
Legend(fig2[3, 1], axs[2]; orientation = :horizontal, framevisible = false)
resize_to_layout!(fig2)
fig2

# !!! note "Validity range"
#     The learned term is only usable inside the velocity range the experiment
#     visited. Outside it the posterior returns to the prior and the band widens;
#     the practical guide covers using that band as a validity check.

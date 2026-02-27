module RecursiveGPs

export RGP, measurement_gp, uncertainty_gp
export predict_gp, predict_kf
export ExtendedKalmanFilter

using AbstractGPs
using ComponentArrays
using ForwardDiff
using LinearAlgebra
using PreallocationTools
using StaticArrays
using Statistics
using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF
import LowLevelParticleFilters: ExtendedKalmanFilter

include("rgp.jl")
include("kalman.jl")
include("model.jl")

end

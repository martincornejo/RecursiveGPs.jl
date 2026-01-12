cov_1(gp::GP, x::Real) = kernelmatrix(gp.kernel, x)
cov_1(gp::GP, x::AbstractArray) = kernelmatrix(gp.kernel, x)

begin
    rng = MersenneTwister(123)

    N_basis = 5
    b0 = sort(rand(rng, N_basis))  
    g_obs = rand(rng, N_basis)      
    b_query = 0.5                   

    gp = GP(ZeroMean(), SqExponentialKernel())
    cov_1(gp, [1.4])
end
include("01_dirichlet_process.jl")


# Data simulation from a DPM of normals
function tic_rdpm_normal(n::Int64, M, m, tau, s, S)
    """
    Random sample from a Dirichlet Process Mixture of Normals. Returns the component
    parameters, the number of unique components (k) and the simulated data

    n            : sample length
    M            : precision parameter
    m, tau, s, S : parameters for the Normal-Inv-Gamma distribution

    TODO: Use any model for the data
    TODO: Use any centering measure G0
    """

    G0 = NormalInverseGamma(m, tau, s, S)
    pi_vector = tic_rdp_marginal(n, M, G0)
    k = size(unique(pi_vector, dims=1))[1]
    y = map(params -> rand(Normal(params[1], sqrt(params[2]))), eachrow(pi_vector))

    return (y=y, params=pi_vector, k=k)
end

include("03_DPM_EW.jl")
include("04_DPM_Neal.jl")

# Pruebas
## Data simulation
Random.seed!(219)
n, alpha, m, tau, s, S = 20, 1, 0, 100, 50, 2
neal_data = tic_rdpm_normal(n, alpha, m, tau, s, S)
prior_par = (alpha, m, tau, s, S)

## Parameters
F_y = Normal
alpha = 1
G0 = NormalInverseGamma(m, tau, s, S)
m = 1

# Phi sampler
function phi_sampler_nig(phi, c, y, G0)
    # Sampler for the Normal-Inverse-Gamma conjugate prior
    c_unique = unique(c)
    for cluster in c_unique
        y_clust = y[c.==cluster]
        n = length(y_clust)
        sum_y = sum(y_clust)
        y_bar = sum_y / n

        # Posterior parameters
        mu_n = (G0.mu + G0.v0 * sum_y) / (1 + G0.v0 * n)
        nu_n = G0.v0 / (n + G0.v0)
        sh_n = G0.shape/2 + n / 2
        sc_n = G0.scale/2 + sum((y_clust .- y_bar) .^ 2) / 2 +
               n / (1 + n * G0.v0) * (y_bar - G0.mu)^2 / 2

        G_c = NormalInverseGamma(mu_n, nu_n, sh_n, sc_n)
        phi[cluster] = rand(G_c)
    end

    return phi
end

N = 1000
@time aux = tic_dpm_neal(neal_data.y, F_y, alpha, G0, m, phi_sampler_nig, N)
@time ew_algorithm1(neal_data.y, prior_par, N)

[1, 2, 1, 1, 1, 3, 1, 4, 1, 1]'
neal_data.params'

# N = 10000, n = 1000
## Neal: 268.8 segundos
## EW:   243.3 segundos

include("../src/00_extras.jl")
include("../src/01_DP.jl")
include("../src/02_DPM.jl")
include("../src/03_DPM_EW.jl")
include("../src/04_DPM_Neal.jl")


velocities = [9172, 9558, 10406, 18419, 18927, 19330, 19440, 19541,
    19846, 19914, 19989, 20179, 20221, 20795, 20875, 21492,
    21921, 22209, 22314, 22746, 22914, 23263, 23542, 23711,
    24289, 24990, 26995, 34279, 9350, 9775, 16084, 18552,
    19052, 19343, 19473, 19547, 19856, 19918, 20166, 20196,
    20415, 20821, 20986, 21701, 21960, 22242, 22374, 22747,
    23206, 23484, 23666, 24129, 24366, 25633, 32065, 9483,
    10227, 16170, 18600, 19070, 19349, 19529, 19663, 19863,
    19973, 20175, 20215, 20629, 20846, 21137, 21814, 22185,
    22249, 22495, 22888, 23241, 23538, 23706, 24285, 24717,
    26960, 32789] ./ 1000
histogram(velocities, bins=1:40, label="")


# 1.1 Ejemplo EW - galaxies
#region
a, b, A, w, W, s, S = 2, 4, 1000, 1, 100, 4, 2
prior_par = (a, b, A, w, W, s, S)

Random.seed!(219)
N = 10000
warmup = 2000
@time g_eta, g_alpha, g_mt, g_pi = tic_dpm_ew(velocities, prior_par, N, warmup);

## Figure 1 - Posterior predictive density p(y|D)
n = length(velocities)
function cond_dens(y)
    s1 = [
        g_alpha[i] * pdf(TDist(s), (y - g_mt[i, 1]) / sqrt(1 + g_mt[i, 2] * S / s)) /
        sqrt(1 + g_mt[i, 2] * S / s) for i in 1:(N-warmup)
    ]

    s2 = [
        sum(map(x -> pdf(Normal(x[1], sqrt(x[2])), y), eachrow(sample)))
        for sample in eachslice(g_pi, dims=1)
    ]

    dens = (s1 .+ s2) ./ (g_alpha .+ n)

    return mean(dens)
end

y_grid = range(8, 40, length=500);
@time dens_est = [cond_dens(y) for y in y_grid];

histogram(velocities, bins=1:40, label="", normalize=true)
plot!(y_grid, dens_est, label="Estimated density", linewidth=2)

## Table 6 - Posterior p(k|D)
k_sim = [size(unique(pi_v, dims=1))[1] for pi_v in eachslice(g_pi, dims = 1)];
prop(freqtable(k_sim))[1:10]
#endregion

# 1.2 Ejemplo EW - Sticky-Clusters
#region
# Gráfico de mu_1 por cada iteración
mu_1_sim = g_pi[7001:end, 1, 1]
plot(1:1000, mu_1_sim, label="")
#endregion

# 2.1 Ejemplo Neal - galaxies

# 2.2 Ejemplo Neal - arreglo sticky-clusters

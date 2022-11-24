include("05_DPM_SIM.jl")


# Pruebas univariate normal
#region
# Estimación datos galaxy: densidad y alpha
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
    26960, 32789] ./ 1000;

histogram(velocities, bins=1:40, label="")

a, b, A, w, W, s, S = 2, 4, 1000, 1, 100, 4, 2;
prior_par = (a, b, A, w, W, s, S);

Random.seed!(219);
N = 4000;
warmup = 2000;

## Escobar & West
#region
@time hyp_ew, pi_ew = _dpm_norm_ew(velocities, prior_par, N, warmup);

## Posteriori predictiva p(y|D)
n = length(velocities)
function cond_dens(y)
    g_alpha = hyp_ew[:, "alpha"].array
    g_mt = hyp_ew[:, ["m", "tau"]].array
    g_pi = pi_ew.array
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

histogram(velocities, bins=1:40, label="", normalize=true);
plot!(y_grid, dens_est, label="Densidad estimada", linewidth=2.5)

## Posteriori p(alpha|D)
function cond_dens(alpha)
    g_eta = hyp_ew[:, "eta"].array
    g_pi = pi_ew.array
    function a_dist(i, alpha)
        eta = g_eta[i]

        unique_pi = unique(g_pi[i, :, :], dims=1)
        k = size(unique_pi)[1]
        odds_w = (a + k - 1) / (n * (b - log(eta)))
        weight = odds_w / (1 + odds_w)

        weight * pdf(Gamma(a + k, 1 / (b - log(eta))), alpha) +
        (1 - weight) * pdf(Gamma(a + k - 1, 1 / (b - log(eta))), alpha)
    end

    dens = [a_dist(i, alpha) for i in 1:(N-warmup)]
    return mean(dens)
end

alpha_grid = range(0, 3, length=500);
dens_est = [cond_dens(alpha) for alpha in alpha_grid]
plot(alpha_grid, dens_est, label="Estimated density", linewidth=3)
plot!(alpha_grid, pdf(Gamma(a, 1 / b), alpha_grid), label="Prior density")

## Posteriori p(k|D)
k_sim = [size(unique(pi_v, dims=1))[1] for pi_v in eachslice(pi_ew.array, dims=1)];
prop(freqtable(k_sim))[1:10]
#endregion

### Bush & MacEachern

### Neal




#endregion

# Pruebas multivariate normal
#region
#endregion

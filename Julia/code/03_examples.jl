using Plots
using StatsPlots
using FreqTables

include("03_ew.jl")


### Recovering parameters
Random.seed!(219)
n, alpha, m, tau, s, S = 50, 1, 0, 100, 50, 2
a1_data = tic_rdpm_normal(n, alpha, m, tau, s, S)

histogram(a1_data.y, bins=30, label="Simulated data")
vline!(unique(a1_data.params[:, 1]), label="Components")

prior_par = (alpha, m, tau, s, S)
@time test_a1_rp = ew_algorithm1(a1_data.y, prior_par, 10000);

sample_means = mapslices(mean, test_a1_rp, dims=[1])

scatter(a1_data.params[:, 1], sample_means[1, :, 1], ms=3, ma=0.3, label="")
Plots.abline!(1, 0, label="")

scatter(a1_data.params[:, 2], sample_means[1, :, 2], ms=3, ma=0.3, label="")
Plots.abline!(1, 0, label="")

k_sim = map(
    x -> size(unique(x, dims=1))[1],
    eachslice(test_a1_rp, dims=1)
)
k_mean = mean(k_sim)

bar(sort(unique(k_sim)), counts(k_sim), label="Sampled values of k")
vline!([a1_data.k], label="Real value")

### Recovering density
Random.seed!(219)
mmodel = MixtureModel(
    Normal[Normal(-5, 1), Normal(-1, 1), Normal(0, 1), Normal(5, 1)],
    [0.15, 0.25, 0.3, 0.3]
)

n = 100
probs = [0.15, 0.25, 0.3, 0.3]
nmeans = [-5, -1, 0, 5]
components = sample(1:4, Weights(probs), n)
y = rand(Normal(0, 1), n) + nmeans[components]

# See https://github.com/JuliaPlots/StatsPlots.jl/issues/458
histogram(y, normalize=true, bins=30, alpha=0.3, label="samples")
plot!(mmodel, components=false, label="Real density")
vline!([-5, -1, 0, 5], c="red")

alpha, m, tau, s, S = 1, 0, 100, 4, 2
prior_par = (alpha, m, tau, s, S)
test_a1_rd = ew_algorithm1(y, prior_par, 10000)
sample_means = mapslices(mean, test_a1_rd, dims=[1])

scatter(nmeans[components], sample_means[1, :, 1], ms=3, ma=0.3, label="")
Plots.abline!(1, 0, label="")

function cond_dens(y)
    M = (1 + tau) * S / s
    s1 = alpha * pdf(TDist(s), (y - m) / sqrt(M)) / sqrt(M)
    s2 = [sum(map(x -> pdf(Normal(x[1], sqrt(x[2])), y), eachrow(sample)))
          for sample in eachslice(test_a1_rd, dims=1)]
    dens = (s1 .+ s2) ./ (alpha + n)

    return mean(dens)
end

@time dens_est = [cond_dens(y) for y in range(-8, 8, length=500)]

plot(mmodel, components=false, label="Real density")
plot!(range(-8, 8, length=500), dens_est, label="Estimated density")


## Galaxies example
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

a, b, A, w, W, s, S = 2, 4, 1000, 1, 100, 4, 2
prior_par = (a, b, A, w, W, s, S)

Random.seed!(219)
N = 10000
warmup = 2000
@time g_eta, g_alpha, g_mt, g_pi = ew_algorithm(velocities, prior_par, N, warmup);

## Posterior predictive density p(y|D)
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
plot!(y_grid, dens_est, label="Estimated density", linewidth=3)


## Posterior p(tau|D)
function cond_dens(tau)
    function ig_pdf(i, tau)
        unique_pi = unique(g_pi[i, :, :], dims=1)
        k = size(unique_pi)[1]
        K = sum((unique_pi[:, 1] .- g_mt[i, 1]) .^ 2 ./ unique_pi[:, 2])
        pdf(InverseGamma((w + k) / 2, (W + K) / 2), tau)
    end

    dens = [ig_pdf(i, tau) for i in 1:(N-warmup)]

    return mean(dens)
end

tau_grid = range(0.1, 250, length=500);
@time dens_est = [cond_dens(tau) for tau in tau_grid];
plot(tau_grid, dens_est, label="Estimated density", linewidth=3);
plot!(tau_grid, pdf(InverseGamma(w / 2, W / 2), tau_grid), label="Prior density")


## Posterior p(alpha|D)
function cond_dens(alpha)
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
@time dens_est = [cond_dens(alpha) for alpha in alpha_grid]
plot(alpha_grid, dens_est, label="Estimated density", linewidth=3)
plot!(alpha_grid, pdf(Gamma(a, 1 / b), alpha_grid), label="Prior density")

## Posterior p(k|D)
k_sim = [size(unique(pi_v, dims=1))[1] for pi_v in eachslice(g_pi, dims = 1)];
prop(freqtable(k_sim))[1:10]

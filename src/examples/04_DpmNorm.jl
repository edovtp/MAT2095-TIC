include("../helpers.jl");
include("../DpmData.jl");
include("../DpmNorm.jl");

## Algorithm 1 - Fixed hyperparameters
# Recovering parameters
#region
Random.seed!(219);
n, M, m, γ, s, S = 100, 1, 0, 100, 10, 100;
data_a1f = DataDpm(n, M, (m, γ, s, S), "Normal");
histogram(data_a1f.y, bins=30, label="Simulated data");
components_means = [x[1] for x in data_a1f.θ];
components_variances = [x[2] for x in data_a1f.θ];
vline!(unique(components_means), label="Components")

prior_par = (M, m, γ, 2 * s, 2 * S);
@time test_a1_normf = DpmNorm1f(data_a1f.y, prior_par, 10000, 8000);

## μ and V
μ_means = [mean(c) for c in eachcol(test_a1_normf.μ_samples)];
scatter(components_means, μ_means, ms=3, ma=0.3, label="")
Plots.abline!(1, 0, label="")
V_means = [mean(c) for c in eachcol(test_a1_normf.V_samples)];
scatter(components_variances, V_means, ms=3, ma=0.3, label="");
Plots.abline!(1, 0, label="")
xlims!((5, 9))
ylims!((5, 9))

## k (number of components)
k_sim = map(
    x -> size(unique(x))[1],
    eachrow(collect(zip(test_a1_normf.μ_samples, test_a1_normf.V_samples)))
);
k_mean = mean(k_sim);
bar(sort(unique(k_sim)), counts(k_sim), label="Sampled values of k");
vline!([data_a1f.k], label="Real value", linewidth=2, color="red");
vline!([k_mean], label="Sample mean", linewidth=2, color="blue")
#endregion

# Recovering density
#region
Random.seed!(219);
mmodel = MixtureModel(
    Normal[Normal(-5, 1), Normal(-1, 1), Normal(0, 1), Normal(5, 1)],
    [0.15, 0.25, 0.3, 0.3]
);

n = 200;
probs = [0.15, 0.25, 0.3, 0.3];
nmeans = [-5, -1, 0, 5];
components = sample(1:4, Weights(probs), n);
y = rand(Normal(0, 1), n) + nmeans[components];
histogram(y, normalize=true, bins=30, alpha=0.3, label="samples");
plot!(mmodel, components=false, label="Real density", color="black", linewidth=2);
vline!([-5, -1, 0, 5], c="red", line=:dash, label="Components")

M, m, γ, s, S = 1, 0, 100, 4, 2;
prior_par = (M, m, γ, s, S);
@time test_a1_normf = DpmNorm1f(y, prior_par, 100000, 80000);

## μ
μ_means = [mean(c) for c in eachcol(test_a1_normf.μ_samples)];
scatter(nmeans[components], μ_means, ms=3, ma=0.3, label="");
Plots.abline!(1, 0, label="")

## Density
function cond_dens(y)
    mean([pdf(Normal(θ[1], sqrt(θ[2])), y) for θ in test_a1_normf.θ_new])
end
@time dens = [cond_dens(y) for y in range(-8, 8, 200)];
plot(mmodel, components=false, label="Real density");
plot!(range(-8, 8, length=200), dens, label="Estimated density")
#endregion

## Algorithm 1 - Random hyperparameters
# Galaxies example
#region
#endregion

# Sticky clusters
#region
#endregion

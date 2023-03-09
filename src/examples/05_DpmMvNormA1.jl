include("../helpers.jl");
include("../DpmData.jl");
include("../DpmMvNorm.jl");

## Algorithm 1 - Fixed hyperparameters
# Recovering parameters
#region
Random.seed!(219);
n, M, m, γ, Ψ, ν = 100, 1, [0, 0], 10, [1 0; 0 1], 5;
data_a1f = DataDpm(n, M, (m, γ, Ψ, ν), "MvNormal");
components_means = unique([x[1] for x in data_a1f.θ]);
components_covs = unique([x[2] for x in data_a1f.θ]);
y1_means = [x[1] for x in components_means];
y2_means = [x[2] for x in components_means];
scatter([x[1] for x in data_a1f.y], [x[2] for x in data_a1f.y],
    label="", xlabel="y1", ylabel="y2");
scatter!(y1_means, y2_means, label = "Components", color="red")

prior_par = (M, m, γ, Ψ, ν);
@time test_a1_mvnormf = DpmMvNorm1f(data_a1f.y, prior_par, 10000, 8000);

## μ and Σ
components_means
unique(test_a1_mvnormf.μ_samples[end, :])
μ_means = [mean(c) for c in eachcol(test_a1_mvnormf.μ_samples)];
scatter([x[1] for x in μ_means], [x[2] for x in μ_means], label = "Simulated");
scatter!(y1_means, y2_means, label = "Real", color="red")
components_covs
unique(test_a1_mvnormf.Σ_samples[end, :])

## k (number of components)
k_sim = map(
    x -> size(unique(x))[1],
    eachrow(collect(zip(test_a1_mvnormf.μ_samples, test_a1_mvnormf.Σ_samples)))
);
k_mean = mean(k_sim);
bar(sort(unique(k_sim)), counts(k_sim), label="Sampled values of k");
vline!([data_a1f.k], label="Real value", linewidth=2, color="red");
vline!([k_mean], label="Sample mean", linewidth=2, color="blue")
#endregion

# Recovering density
#region

## μ

## Density
#endregion
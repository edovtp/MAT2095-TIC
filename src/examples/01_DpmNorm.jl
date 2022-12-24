include("00_helpers.jl")
include("../01_DpmNorm.jl")

## Fixed hyperparameters - Algorithm 1
#region
# Recovering parameters
Random.seed!(219);
n, M, m, λ, s, S = 200, 1, 0, 0.01, 2, 2;
data = RdpmNormal(n, M, (m, λ, s, S));
histogram(data.y, bins=30, label="Simulated data")
vline!(unique(data.θ[:, 1]), label="Components")

prior_par = (M, m, λ, 2 * s, 2 * S);
@time test_a1_normf = DpmNorm1f(data.y, prior_par, 10000);

μ_means = [mean(c) for c in eachcol(test_a1_normf.μ_samples)];
scatter(data.θ[:, 1], μ_means, ms=3, ma=0.3, label="");
Plots.abline!(1, 0, label="")
V_means = [mean(c) for c in eachcol(test_a1_normf.V_samples)];
scatter(data.θ[:, 2], V_means, ms=3, ma=0.3, label="");
Plots.abline!(1, 0, label="")

# Recovering density
Random.seed!(219);
mmodel = MixtureModel(
    Normal[Normal(-5, 1), Normal(-1, 1), Normal(0, 1), Normal(5, 1)],
    [0.15, 0.25, 0.3, 0.3]
);

n = 100;
probs = [0.15, 0.25, 0.3, 0.3];
nmeans = [-5, -1, 0, 5];
components = sample(1:4, Weights(probs), n);
y = rand(Normal(0, 1), n) + nmeans[components];



## Random hyperparameters - Algorithm 1
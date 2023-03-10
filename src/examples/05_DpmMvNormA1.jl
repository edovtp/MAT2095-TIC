using CairoMakie
using DelimitedFiles
using ColorSchemes

include("../helpers.jl");
include("../DpmData.jl");
include("../DpmMvNorm.jl");
include("../DpData.jl")


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


## A2 - Simulated data
#region
Random.seed!(219);
n, M, m, γ, Ψ, ν = 100, 0.5, [0, 0], 10, [1 0; 0 1], 5;
c1 = MvNormal([0, 0], [[0.5, 0.1] [0.1, 0.5]]);
c2 = MvNormal([-5, -2], [[0.6, 0] [0, 0.6]]);
c3 = MvNormal([0, -4], [[6, 0] [0, 0.1]]);
c4 = MvNormal([5, -5], [[1, -0.5] [-0.5, 1]]);

data_a2 = vcat(rand(c1, 25)', rand(c2, 25)', rand(c3, 25)', rand(c4, 25)');
c = repeat([1, 2, 3, 4], inner=25, outer=1);

y = [Vector(a) for a in eachrow(data_a2)];
prior_par = (M, m, γ, Ψ, ν);
@time a1_mvnorm = DpmMvNorm1f(y, prior_par, 10000, 8000);

c_sim = Matrix{Int64}(undef, 2000, 100);
for (idx, row) in enumerate(eachrow(a1_mvnorm.μ_samples))
    unique_row = unique(row)
    c_unsorted = Vector(indexin(row, unique_row))
    c_sim[idx, :] = c_unsorted
end

# writedlm("aux.csv", c_sim, ",")

c_bind = [1, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4,
4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 6, 6, 5, 5, 6, 5, 6, 6, 5, 6, 5, 7, 6, 8, 5, 6, 5, 6, 7, 6, 5, 6, 8, 6, 7, 8, 8, 8, 8, 7,
8, 8, 8, 8, 8, 7, 8, 8, 7, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8];

c_iv = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2,
2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 3, 4, 3, 4, 4, 3, 4, 3, 5, 4, 6, 3, 4, 3, 4, 5, 4, 3, 4, 6, 4, 5, 6, 6, 6, 6, 5,
6, 6, 6, 6, 6, 5, 6, 6, 5, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6];

c_omari = [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4,
4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 6, 6, 5, 5, 6, 5, 6, 6, 5, 6, 5, 7, 6, 8, 5, 6, 5, 6, 7, 6, 5, 6, 8, 6, 7, 8, 8, 8, 8, 7,
8, 8, 8, 8, 8, 7, 8, 8, 7, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8];

begin
    CairoMakie.activate!()
    fig = Figure()
    ax1 = Axis(fig[1, 1]; title="Real", xgridvisible=false, ygridvisible=false)
    sc = scatter!(data_a2[:, 1], data_a2[:, 2]; color=c, colormap=ColorSchemes.Dark2_8)

    ax_bind = Axis(fig[1, 2]; title="Pérdida Binder", xgridvisible=false, ygridvisible=false)
    sc_bind = scatter!(data_a2[:, 1], data_a2[:, 2]; color=c_bind, colormap=ColorSchemes.Dark2_8)

    ax_iv = Axis(fig[2, 1]; title="Pérdida VI", xgridvisible=false, ygridvisible=false)
    sc_iv = scatter!(data_a2[:, 1], data_a2[:, 2]; color=c_iv, colormap=ColorSchemes.Dark2_8)

    ax_omARI = Axis(fig[2, 2]; title="Pérdida omARI", xgridvisible=false, ygridvisible=false)
    sc_omARI = scatter!(data_a2[:, 1], data_a2[:, 2]; color=c_omari, colormap=ColorSchemes.Dark2_8)

    fig
end

CairoMakie.save("monography/figures/SALSO.png", fig)
#endregion

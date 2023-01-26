using CairoMakie
using SpecialFunctions
include("../helpers.jl")
include("../PyData.jl")
include("../DpData.jl")

### Marginal samples
## Sampling
Random.seed!(619);
precisions = [1, 10, 50, 100, 1000, 10000];
G0 = Gamma(6, 1);
py_data_samples = [DataPy(500, M, 0.6, (6, 1), "Gamma") for M in precisions];

## Figure
begin
    CairoMakie.activate!()
    fig = Figure(resolution=(1280, 720))
    xrange = range(0.01, 15, 100)

    ## M = 1
    ax_1 = Axis(fig[1, 1], xlabel="y", xgridvisible=false, ygridvisible=false,
        title="M=1", titlesize=30)
    h1 = hist!(py_data_samples[1], normalization=:pdf, color=:tomato, label="Muestra")
    l1 = lines!(xrange, x -> pdf(G0, x), label="G0")

    ## M = 10
    ax_2 = Axis(fig[1, 2], xlabel="y", xgridvisible=false, ygridvisible=false,
        title="M=10", titlesize=30)
    hist!(py_data_samples[2], normalization=:pdf, color=:tomato, label="Muestra")
    lines!(xrange, x -> pdf(G0, x), label="G0")

    ## M = 50
    ax_3 = Axis(fig[1, 3], xlabel="y", xgridvisible=false, ygridvisible=false,
        title="M=50", titlesize=30)
    hist!(py_data_samples[3], normalization=:pdf, color=:tomato, label="Muestra")
    lines!(xrange, x -> pdf(G0, x), label="G0")

    ## M = 100
    ax_4 = Axis(fig[2, 1], xlabel="y", xgridvisible=false, ygridvisible=false,
        title="M=100", titlesize=30)
    hist!(py_data_samples[4], normalization=:pdf, color=:tomato, label="Muestra")
    lines!(xrange, x -> pdf(G0, x), label="G0")

    ## M = 1000
    ax_5 = Axis(fig[2, 2], xlabel="y", xgridvisible=false, ygridvisible=false,
        title="M=1000", titlesize=30)
    hist!(py_data_samples[5], normalization=:pdf, color=:tomato, label="Muestra")
    lines!(xrange, x -> pdf(G0, x), label="G0")

    ## M = 10000
    ax_6 = Axis(fig[2, 3], xlabel="y", xgridvisible=false, ygridvisible=false,
        title="M=10000", titlesize=30)
    hist!(py_data_samples[6], normalization=:pdf, color=:tomato, label="Muestra")
    lines!(xrange, x -> pdf(G0, x), label="G0")

    Legend(fig[1:2, 4], [h1, l1], ["Muestra", "G0"])
    fig
end

CairoMakie.save("monography/figures/PY - Urn.png", fig)

### Comparison between PY and DP
Random.seed!(219);
n_values = 1000:2000:50000;
ek_py = Vector{Float64}(undef, length(n_values));
ek_dp = Vector{Float64}(undef, length(n_values));
M = 10;
b = 0.3;

function n_unique_dp(n, M)
    sim_data = DataDp(n, M, (6, 1), "Gamma")
    k = length(unique(sim_data))
end

function n_unique_py(n, M, b)
    sim_data = DataPy(n, M, b, (6, 1), "Gamma")
    k = length(unique(sim_data))
end

for (i, n) in enumerate(n_values)
    k_dp_sim = [n_unique_dp(n, M) for _ in 1:30]
    ek_dp[i] = mean(k_dp_sim)

    k_py_sim = [n_unique_py(n, M, b) for _ in 1:30]
    ek_py[i] = mean(k_py_sim)
end

begin
    CairoMakie.activate!()
    fig = Figure(resolution=(1440, 700))
    xrange = range(1000, 50000, length=500)

    ax = Axis(fig[1, 1], xgridvisible=false, ygridvisible=false, ylabel="k", xlabel="n")
    p1 = scatter!(n_values, ek_dp, markersize=20)
    p2 = scatter!(n_values, ek_py, markersize=20)

    Legend(fig[1, 2], [p1, p2], ["DP", "PY"])
    fig
end

CairoMakie.save("monography/figures/PY DP comparison.png", fig)

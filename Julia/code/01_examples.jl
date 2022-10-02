using Random
using Plots
using StatsPlots

include("01_dirichlet_process.jl")


function tic_rdp_example(n, M, G0::UnivariateDistribution, first, last, plot_lim)
    # TODO: tic_rdp_example - Put the centering measure on top
    # TODO: tic_rdp_example - Set opacity of samples

    dp_samples = [tic_rdp(M, G0) for i in 1:n]

    plot_dp = StatsPlots.plot(G0, func=cdf, size=(800, 700), label="Centering measure")
    xlims!(plot_dp, plot_lim)

    for sample in dp_samples
        locations = vec(sample.locations)
        ord_locations = sort(locations)
        ord_probs = sample.probs[sortperm(locations)]
        plot!(
            plot_dp,
            [first; ord_locations; last],
            [0; cumsum(ord_probs); 1],
            linetype=:steppost,
            label="",
        )
    end

    return plot_dp
end


# Dirichlet process simulation
# Example 1 - Normal centering measure
Random.seed!(219);
G0 = Distributions.Normal(0, 1)
tic_rdp_example(15, 1, G0, -10, 10, (-3, 3))
tic_rdp_example(15, 10, G0, -10, 10, (-3, 3))
tic_rdp_example(15, 50, G0, -10, 10, (-3, 3))
tic_rdp_example(15, 100, G0, -10, 10, (-3, 3))
tic_rdp_example(15, 500, G0, -10, 10, (-3, 3))   # R: 4s
tic_rdp_example(15, 1000, G0, -10, 10, (-3, 3))  # R: 14s

@time tic_rdp_example(15, 500, G0, -10, 10, (-3, 3))
@time tic_rdp_example(15, 1000, G0, -10, 10, (-3, 3))

# Example 2 - Poisson centering measure
Random.seed!(219);
G0 = Distributions.Poisson(5)
tic_rdp_example(15, 1, G0, -10, 20, (0, 15))
tic_rdp_example(15, 10, G0, -10, 20, (0, 15))
tic_rdp_example(15, 50, G0, -10, 20, (0, 15))
tic_rdp_example(15, 100, G0, -10, 20, (0, 15))
tic_rdp_example(15, 500, G0, -10, 20, (0, 15))
@time tic_rdp_example(15, 1000, G0, -10, 20, (0, 15))

# Example 3 - Gamma distribution
Random.seed!(219);
G0 = Distributions.Gamma(6, 1 / 4)
tic_rdp_example(15, 1, G0, -10, 10, (0, 4))
tic_rdp_example(15, 10, G0, -10, 10, (0, 4))
tic_rdp_example(15, 50, G0, -10, 10, (0, 4))
tic_rdp_example(15, 100, G0, -10, 10, (0, 4))
tic_rdp_example(15, 500, G0, -10, 10, (0, 4))
@time tic_rdp_example(15, 1000, G0, -10, 10, (0, 4))

# Data simulation from a Dirichlet Process
# Example 1 - Normal centering measure
Random.seed!(219);
G0 = Distributions.Normal(0, 1);

@time rdp_m_normal1 = tic_rdp_marginal(500, 1, G0);
StatsPlots.plot(G0, func=pdf, size=(800, 600), label="Centering measure");
StatsPlots.density!(rdp_m_normal1, label="Sample from a DP")

@time rdp_m_normal2 = tic_rdp_marginal(500, 10, G0);
StatsPlots.plot(G0, func=pdf, size=(800, 600), label="Centering measure");
StatsPlots.density!(rdp_m_normal2, label="Sample from a DP")

@time rdp_m_normal3 = tic_rdp_marginal(500, 50, G0);
StatsPlots.plot(G0, func=pdf, size=(800, 600), label="Centering measure");
StatsPlots.density!(rdp_m_normal3, label="Sample from a DP")

@time rdp_m_normal4 = tic_rdp_marginal(500, 100, G0);
StatsPlots.plot(G0, func=pdf, size=(800, 600), label="Centering measure");
StatsPlots.density!(rdp_m_normal4, label="Sample from a DP")

@time rdp_m_normal5 = tic_rdp_marginal(500, 1000, G0);
StatsPlots.plot(G0, func=pdf, size=(800, 600), label="Centering measure");
StatsPlots.density!(rdp_m_normal5, label="Sample from a DP")

@time rdp_m_normal6 = tic_rdp_marginal(500, 10000, G0);
StatsPlots.plot(G0, func=pdf, size=(800, 600), label="Centering measure");
StatsPlots.density!(rdp_m_normal6, label="Sample from a DP")

# Example 2 - Gamma centering measure
Random.seed!(219);
G0 = Distributions.Gamma(3, 0.1);

rdp_m_gamma1 = tic_rdp_marginal(500, 1, G0);
StatsPlots.plot(G0, func=pdf, size=(800, 600), label="Centering measure");
StatsPlots.density!(rdp_m_gamma1, label="Sample from a DP")

rdp_m_gamma2 = tic_rdp_marginal(500, 10, G0);
StatsPlots.plot(G0, func=pdf, size=(800, 600), label="Centering measure");
StatsPlots.density!(rdp_m_gamma2, label="Sample from a DP")

rdp_m_gamma3 = tic_rdp_marginal(500, 50, G0);
StatsPlots.plot(G0, func=pdf, size=(800, 600), label="Centering measure");
StatsPlots.density!(rdp_m_gamma3, label="Sample from a DP")

rdp_m_gamma4 = tic_rdp_marginal(500, 100, G0);
StatsPlots.plot(G0, func=pdf, size=(800, 600), label="Centering measure");
StatsPlots.density!(rdp_m_gamma4, label="Sample from a DP")

rdp_m_gamma5 = tic_rdp_marginal(500, 1000, G0);
StatsPlots.plot(G0, func=pdf, size=(800, 600), label="Centering measure");
StatsPlots.density!(rdp_m_gamma5, label="Sample from a DP")

@time rdp_m_gamma6 = tic_rdp_marginal(500, 10000, G0);
StatsPlots.plot(G0, func=pdf, size=(800, 600), label="Centering measure");
StatsPlots.density!(rdp_m_gamma6, label="Sample from a DP")


# Empirical results - Antoniak, Korwar & Hollander
function akh_empirical(n_values, M)
    # Expected value for k
    ek_values = Vector{Float64}(undef, length(n_values))

    function n_unique(n, M)
        sim_data = tic_rdp_marginal(n, M, Normal(0, 1))
        k = length(unique(sim_data))
    end

    for (i, n) in enumerate(n_values)
        k_sim = [n_unique(n, M) for _ in 1:20]
        ek_values[i] = mean(k_sim)
    end

    result = plot(x -> M * log(x), 1, maximum(n_values) + 100,
        label="Korwar & Hollander", size=(800, 600))
    plot!(result, x -> M * log(1 + x / M), label="Antoniak")
    scatter!(result, n_values, ek_values, label="Simulation")
end


Random.seed!(219)
n_values = 1000:2000:21000

@time akh_empirical(n_values, 1)
@time akh_empirical(n_values, 100)
@time akh_empirical(n_values, 1000)

include("01_DP.jl")


# Dirichlet process simulation
#region
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
#endregion

# Data simulation from a Dirichlet Process
#region

# Example 1 - Normal centering measure
Random.seed!(219);
G0 = Distributions.Normal(0, 1);

tic_rdp_marginal_example(500, 1, G0)
tic_rdp_marginal_example(500, 10, G0)
tic_rdp_marginal_example(500, 50, G0)
tic_rdp_marginal_example(500, 100, G0)
tic_rdp_marginal_example(500, 1000, G0)
tic_rdp_marginal_example(500, 10000, G0)

# Example 2 - Gamma centering measure
Random.seed!(219);
G0 = Distributions.Gamma(3, 0.1);

tic_rdp_marginal_example(500, 1, G0)
tic_rdp_marginal_example(500, 10, G0)
tic_rdp_marginal_example(500, 50, G0)
tic_rdp_marginal_example(500, 100, G0)
tic_rdp_marginal_example(500, 1000, G0)
tic_rdp_marginal_example(500, 10000, G0)

#endregion

# Empirical results - Antoniak, Korwar & Hollander
#region
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
        label="Korwar & Hollander", size=(800, 600),
        legend=:bottomright)
    plot!(result, x -> M * log(1 + x / M), label="Antoniak")
    scatter!(result, n_values, ek_values, label="Simulation")
end


Random.seed!(219)
n_values = 1000:2000:21000

@time akh_empirical(n_values, 1)
@time akh_empirical(n_values, 100)
@time akh_empirical(n_values, 1000)
#endregion

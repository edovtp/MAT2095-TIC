using Random
using Plots
using StatsPlots
using Distributions


# Dirichlet process simulation
function tic_rdp(M::Number, G0::Distributions.Distribution, tol=1e-6)
    # M   -> Precision parameter
    # G0  -> Centering measure

    # Components of the distribution
    probs = Vector{Float64}()
    locations = Vector{Float64}()

    # Algorithm
    ## Auxiliar values
    ups_aux = [0.0]
    ups_dist = Distributions.Beta(1, M)

    ## Simulation
    while sum(probs) < 1 - tol
        # Auxiliar upsilon ~ Beta(1, M)
        ups_sample = rand(ups_dist)

        # Probability (weights)
        append!(probs, ups_sample * prod(1 .- ups_aux))
        append!(ups_aux, ups_sample)

        # Location
        append!(locations, rand(G0))
    end

    # Return ordered values of the (discrete) distribution
    ord_probs = probs[sortperm(locations)]
    ord_locations = sort(locations)

    return (locations=ord_locations, probs=ord_probs)
end

# TODO: Put the centering measure on top
# TODO: Set opacity of samples

function tic_rdp_example(n, M, G0::Distributions.Distribution, first, last, plot_lim)
    dp_samples = [tic_rdp(M, G0) for i in 1:n];

    plot_dp = StatsPlots.plot(G0, func=cdf, size=(800, 700), label="Centering measure")
    xlims!(plot_dp, plot_lim)
    for sample in dp_samples
        plot!(
            plot_dp,
            [first; sample.locations; last],
            [0; cumsum(sample.probs); 1],
            linetype=:steppost,
            label="", 
        )
    end

    return plot_dp
end


# Example 1 - Normal centering measure
Random.seed!(219);
tic_rdp_example(15, 1, Distributions.Normal(0, 1), -10, 10, (-3, 3))
tic_rdp_example(15, 10, Distributions.Normal(0, 1), -10, 10, (-3, 3))
tic_rdp_example(15, 50, Distributions.Normal(0, 1), -10, 10, (-3, 3))
tic_rdp_example(15, 100, Distributions.Normal(0, 1), -10, 10, (-3, 3))
tic_rdp_example(15, 500, Distributions.Normal(0, 1), -10, 10, (-3, 3))
tic_rdp_example(15, 1000, Distributions.Normal(0, 1), -10, 10, (-3, 3))

# Example 2 - Poisson centering measure
Random.seed!(219);
tic_rdp_example(15, 1, Distributions.Poisson(5), -10, 20, (0, 15))
tic_rdp_example(15, 10, Distributions.Poisson(5), -10, 20, (0, 15))
tic_rdp_example(15, 50, Distributions.Poisson(5), -10, 20, (0, 15))
tic_rdp_example(15, 100, Distributions.Poisson(5), -10, 20, (0, 15))
tic_rdp_example(15, 500, Distributions.Poisson(5), -10, 20, (0, 15))
tic_rdp_example(15, 1000, Distributions.Poisson(5), -10, 20, (0, 15))

# Example 3 - Gamma distribution
Random.seed!(219);
tic_rdp_example(15, 1, Distributions.Gamma(6, 1/4), -10, 10, (0, 4))
tic_rdp_example(15, 10, Distributions.Gamma(6, 1/4), -10, 10, (0, 4))
tic_rdp_example(15, 50, Distributions.Gamma(6, 1/4), -10, 10, (0, 4))
tic_rdp_example(15, 100, Distributions.Gamma(6, 1/4), -10, 10, (0, 4))
tic_rdp_example(15, 500, Distributions.Gamma(6, 1/4), -10, 10, (0, 4))
tic_rdp_example(15, 1000, Distributions.Gamma(6, 1/4), -10, 10, (0, 4))

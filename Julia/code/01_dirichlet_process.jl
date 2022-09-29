using Random
using Plots
using StatsBase
using StatsPlots
using Distributions


# Dirichlet process simulation -----------------------------------------------
function tic_rdp(M::Number, G0::Distributions.Distribution, tol=1e-6)
    """
    Random sample of a Dirichlet process using the stick-breaking construction

    M  : precision parameter
    G0 : centering measure
    """

    probs = Vector{Float64}()
    locations = Vector{Float64}()
    
    ups_dist = Distributions.Beta(1, M)
    ups_aux = [0.0]
    while sum(probs) < 1 - tol
        ups_sample = rand(ups_dist)
        append!(probs, ups_sample * prod(1 .- ups_aux))
        append!(ups_aux, ups_sample)
        append!(locations, rand(G0))
    end

    ord_locations = sort(locations)
    ord_probs = probs[sortperm(locations)]

    return (locations=ord_locations, probs=ord_probs)
end;

function tic_rdp_example(n, M, G0::Distributions.Distribution, first, last, plot_lim)
    # TODO: tic_rdp_example - Put the centering measure on top
    # TODO: tic_rdp_example - Set opacity of samples

    dp_samples = [tic_rdp(M, G0) for i in 1:n]

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
end;

# Data simulation from a Dirichlet Process ------------------------------------------
function tic_rdp_marginal(n::Int64, M, G0::Distributions.Distribution)
    """
    Random sample from a Dirichlet process using the marginal distribution

    n  : sample length
    M  : precision parameter
    G0 : centering mesasure
    """
    # TODO: replace with the dimension of the centering measure
    marginal_sample = Vector{Float64}(undef, n)
    counter = Dict{Any, Int64}()

    for i in 1:n
        all_values = collect(keys(counter))
        candidate = rand(G0)
        append!(all_values, candidate)

        norm_term = 1/(M + i - 1)
        old_freq = collect(values(counter))
        probs = [old_freq .* norm_term ; M * norm_term]

        new_value = StatsBase.sample(all_values, Weights(probs))
        marginal_sample[i] = new_value

        freq = get(counter, new_value, 0)
        counter[new_value] = freq + 1
    end

    return marginal_sample
end;

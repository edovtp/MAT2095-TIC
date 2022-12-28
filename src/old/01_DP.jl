include("00_extras.jl")


# Dirichlet process simulation -----------------------------------------------
function tic_rdp(M, G0::Distribution, tol=1e-6)
    """
    Random sample of a (Finite) Dirichlet Process using the stick-breaking construction
    of Sethuraman (1994). Sampling is done until we have covered a total of 1 - tol
    probability

    M  : precision parameter
    G0 : centering measure
    """

    probs = ElasticArrays.ElasticVector{Float64}(undef, 0)

    ups_dist = Distributions.Beta(1, M)
    sum_l_ups = 0
    while sum(probs) < 1 - tol
        ups_sample = rand(ups_dist)
        append!(probs, exp(log(ups_sample) + sum_l_ups))
        sum_l_ups += log(1 - ups_sample)
    end

    append!(probs, 1 - sum(probs))
    # Some distributions can only be drawn from one at a time
    locations = [rand(G0) for _ in 1:length(probs)]

    return (locations=locations, probs=probs)
end

function tic_rdp_example(n, M, G0::UnivariateDistribution, first, last, plot_lim)
    plot_dp = StatsPlots.plot(size=(800, 700), legend=:bottomright)
    xlims!(plot_dp, plot_lim)

    dp_samples = [tic_rdp(M, G0) for i in 1:n]
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
            alpha=0.7,
            color="gray"
        )
    end

    plot!(
        plot_dp,
        G0,
        func=cdf,
        label="",
        color="red",
        linewidth=2.5
    )

    return plot_dp
end

# Data simulation from a Dirichlet Process ------------------------------------------
function tic_rdp_marginal(n, M, G0::Distribution)
    """
    Data simulation from a Dirichlet process using the Polya Urn representation of Blackwell
    and McQueen (1973)

    n  : sample length
    M  : precision parameter
    G0 : centering mesasure
    """

    marginal_sample = ElasticArray{Float64}(undef, length(G0), 0)
    counter = Dict{Any,Int64}()

    for i in 1:n
        all_values = collect(keys(counter))
        candidate = rand(G0)
        push!(all_values, candidate)

        norm_term = 1 / (M + i - 1)
        old_freq = collect(values(counter))
        probs = [old_freq .* norm_term; M * norm_term]

        new_value = StatsBase.sample(all_values, Weights(probs))
        append!(marginal_sample, new_value)

        freq = get(counter, new_value, 0)
        counter[new_value] = freq + 1
    end

    return marginal_sample'
end

function tic_rdp_marginal_example(n, M, G0::UnivariateDistribution)
    rdp_marginal_sample = tic_rdp_marginal(n, M, G0)
    StatsPlots.plot(G0, func=pdf, size=(800, 600), label="Centering measure")
    StatsPlots.density!(rdp_marginal_sample, label="Sample from a DP")
    StatsPlots.title!("M=$M")
end

# TODO: Dirichlet Process Model ------------------------------------------
function tic_model_dp(y, M, G0::Distribution, tol=1e-6)
    """
    Posterior inference for the model given by (2.1), i.e.
                    y_i | G ~ G
                          G ~ DP

    y  : data
    M  : precision parameter
    G0 : centering measure
    """
end

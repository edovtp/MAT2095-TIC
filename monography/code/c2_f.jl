# Data simulation from a Dirichlet Process (Blackwell-McQueen) ---------------
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
    StatsPlots.plot(G0, func=pdf, size=(800, 600), label="", linewidth=2.5, grid=false)
    StatsPlots.density!(rdp_marginal_sample, label="", linewidth=2.5)
end


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
    plot_dp = StatsPlots.plot(size=(800, 700), grid=false, legend=:bottomright)
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

# Antoniak, Korwar & Hollander -----------------------------------------------
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

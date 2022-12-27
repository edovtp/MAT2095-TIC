function DpmNorm1f(y, prior_par, iter, warmup=floor(Int64, iter / 2))
    """
    Implementation of algorithm 1 (Neal, 2000) for the Normal model with conjugate prior G0
    and fixed hyperparameters. That is,

        y_i | θ_i ~ Normal(μ_i, V_i)
        θ_i | G ~ G
        G ~ DP(M, G0)
        G0 = NIG(m, γ, s/2, S/2) ≡ N(μ|m, γσ^2)⋅IG(σ^2|s/2, S/2)

    y         : data to fit the model
    prior_par : prior parameters (M, m, γ, s, S)
    iter      : number of iterations including warmup
    """
    n = length(y)
    M, m, γ, s, S = prior_par
    total_samples = iter - warmup
    μ_samples = Array{Float64}(undef, total_samples, n)
    V_samples = Array{Float64}(undef, total_samples, n)
    θ_new = Array{Float64}(undef, total_samples, 2)

    # Posterior distributions
    m_p = @. (m + γ * y) / (γ + 1)
    γ_p = γ / (1 + γ)
    s_p = s + 1
    S_p = @. S + (y - m)^2 / (1 + γ)
    prior_d = NormalInverseGamma(m, γ, s / 2, S / 2) # G0
    post_d = [NormalInverseGamma(a[1], γ_p, s_p / 2, a[2] / 2) for a in zip(m_p, S_p)] # H_i
    scale_t = sqrt(S / s * (1 + γ))
    r_i = [M * pdf(TDist(s), (obs - m) / scale_t) / scale_t for obs in y]

    # Initial values
    prev_μ = Vector{Float64}(undef, n)
    prev_V = Vector{Float64}(undef, n)
    for i in 1:n
        prev_μ[i], prev_V[i] = rand(post_d[i])
    end

    # Start of the algorithm
    for n_sample in 1:iter
        # Update of each component of θ
        counter_θ = Dict{Tuple,Int64}()
        for i in 1:n
            weights = Vector{Float64}(undef, n)
            weights[i] = r_i[i]                                 # r_i
            id_c = 1:n .!= i                                    # index complement of i
            weights[id_c] = [                                   # q_ij
                pdf(Normal(x[1], sqrt(x[2])), y[i]) for x in zip(prev_μ[id_c], prev_V[id_c])
            ]
            idx_new = StatsBase.sample(1:n, Weights(weights))
            if idx_new == i
                prev_μ[i], prev_V[i] = rand(post_d[i])
            else
                prev_V[i] = prev_V[idx_new]
                prev_μ[i] = prev_μ[idx_new]
            end
            sample = (prev_μ[i], prev_V[i])
            freq = get(counter_θ, sample, 0)
            counter_θ[sample] = freq + 1
        end

        if n_sample > warmup
            μ_samples[n_sample-warmup, :] .= prev_μ
            V_samples[n_sample-warmup, :] .= prev_V

            # New θ value
            all_values = collect(keys(counter_θ))
            candidate = rand(prior_d)
            push!(all_values, candidate)
            freqs = collect(values(counter_θ))
            norm_term = 1 / (M + n)
            probs = [freqs .* norm_term; M * norm_term]
            θ_new[n_sample-warmup, :] .= StatsBase.sample(all_values, Weights(probs))
        end
    end

    return (μ_samples=μ_samples, V_samples=V_samples, θ_new=θ_new)
end

function DpmNorm1()

end

function DpmNorm2f()

end

function DpmNorm2()

end

function DpmNorm8f()

end

function DpmNorm8()

end

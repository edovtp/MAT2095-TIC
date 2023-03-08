function DpmNorm1f(y, prior_par, iter, warmup=floor(Int64, iter / 2))
    """
    Implementation of Algorithm 1 (Neal, 2000) for the Normal model with conjugate prior G0
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
    θ_new = Vector{Tuple}(undef, total_samples)

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
            id_c = 1:n .!= i                                    # index complement of {i}
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
            θ_new[n_sample-warmup] = StatsBase.sample(all_values, Weights(probs))
        end
    end

    return (μ_samples=μ_samples, V_samples=V_samples, θ_new=θ_new)
end

function _relabel(c, θ_unique)
    c_re = Vector{Int64}(undef, length(c))

    c_unique = unique(c)
    θ_re = θ_unique[c_unique]
    for i in eachindex(c_unique)
        c_re[c.==c_unique[i]] .= i
    end

    return c_re, θ_re
end

function DpmNorm2(y, prior_par, iter, init_c="same", warmup=floor(Int64, iter / 2))
    """
    Implementation of Algorithm 2-3 (Neal, 2000) for the Normal model with conjugate prior G0 and
    random hyperparameters. That is,

        y_i | θ_i ∼ Normal(μ_i, V_i)
        θ_i | G ∼ G
        G ∼ DP(M, G_η)
        G_η = NIG(m, γ, s/2, S/2)
        m ∼ Normal(a, A)
        γ ∼ IG(w/2, W/2)
        M ∼ Gamma(α, β)

    y         : data to fit the model
    prior_par : prior parameters (a, A, w, W, α, β, s, S)
    iter      : number of iterations including warmup
    init_c    : initial value for c, with "same" they all start in the same cluster, "diff" to put
                them all in different clusters
    """
    n = length(y)
    a, A, w, W, α, β, s, S = prior_par
    total_samples = iter - warmup
    samples = Dict(
        :μ => Array{Float64}(undef, total_samples, n),
        :V => Array{Float64}(undef, total_samples, n),
        :c => Array{Int64}(undef, total_samples, n),
        :m => Vector{Float64}(undef, total_samples),
        :γ => Vector{Float64}(undef, total_samples),
        :M => Vector{Float64}(undef, total_samples),
        :ϕ => Vector{Float64}(undef, total_samples),
        :π => Vector{Float64}(undef, total_samples),
        :θ_new => Vector{Tuple}(undef, total_samples)
    )

    # Initial values
    M = rand(Gamma(α, 1 / β))
    γ = rand(InverseGamma(w / 2, W / 2))
    m = rand(Normal(a, A))
    G_η = NormalInverseGamma(m, γ, s / 2, S / 2)
    if init_c == "same"
        c = ones(Int64, n)
        θ_unique = [rand(G_η)]
    elseif init_c == "diff"
        c = Vector{Int64}(1:n)
        θ_unique = [rand(G_η) for _ in 1:n]
    end

    # Start of the algorithm
    for n_sample in 1:iter
        # Update M
        k = length(θ_unique)
        ϕ = rand(Beta(M + 1, n))
        odds_weight = (α + k - 1) / (n * (β - log(ϕ)))
        weight = odds_weight / (1 + odds_weight)
        component = sample([0, 1], Weights([weight, 1 - weight]))
        M = rand(Gamma(α + k - 1 * component, 1 / (β - log(ϕ))))

        # Update m, γ
        μ_unique = [θ[1] for θ in θ_unique]
        V_unique = [θ[2] for θ in θ_unique]
        V_bar = 1 / sum(1 ./ V_unique)
        x = A / (A + γ * V_bar)
        m = rand(Normal((1 - x) * a + x * V_bar * sum(μ_unique ./ V_unique), sqrt(x * γ * V_bar)))
        K = sum(((μ_unique .- m) .^ 2) ./ V_unique)
        γ = rand(InverseGamma((w + k) / 2, (W + K) / 2))

        # Update c
        for i in 1:n
            current_c = c[i]
            c_ = c[1:end.!=i] # c minus i
            y_ = y[1:end.!=i]

            c_unique_ = unique(c_)
            k_ = length(c_unique_)
            weights = Vector{Float64}(undef, k_ + 1)
            for (idx, cluster) in enumerate(c_unique_)
                y_clust_ = y_[c_.==cluster]
                n_j_ = length(y_clust_)

                y_bar_ = mean(y_clust_)
                m_j_ = (m + γ * sum(y_clust_)) / (1 + γ * n_j_)
                γ_j_ = γ / (1 + n_j_ * γ)
                s_j_ = s + n_j_
                S_j_ = S + n_j_ / (1 + n_j_ * γ) * (y_bar_ - m)^2 + sum((y_clust_ .- y_bar_) .^ 2)

                scale_t = sqrt(S_j_ / s_j_ * (1 + γ_j_))
                weights[idx] = n_j_ * pdf(TDist(s_j_), (y[i] - m_j_) / scale_t) / scale_t
            end
            # New cluster
            scale_t = sqrt(S / s * (1 + γ))
            weights[k_ + 1] = M * pdf(TDist(s), (y[i] - m) / scale_t) / scale_t
            idx_cluster_new = StatsBase.sample(1:(k_ + 1), Weights(weights))
            if idx_cluster_new == (k_ + 1)
                if length(θ_unique) == k_
                    c[i] = maximum(c_unique_) + 1
                    push!(θ_unique, rand(NormalInverseGamma(m, γ, s / 2, S / 2)))
                else
                    θ_unique[current_c] = rand(NormalInverseGamma(m, γ, s / 2, S / 2))
                end
            else
                c[i] = c_unique_[idx_cluster_new]
            end

            c, θ_unique = _relabel(c, θ_unique)
        end

        # Update θ_unique
        n_j_list = Vector{Int64}(undef, length(θ_unique))
        for j in eachindex(θ_unique)
            y_clust = y[c .== j]
            n_j = length(y_clust)
            n_j_list[j] = n_j
            y_bar = mean(y_clust)

            m_j = (m + γ*sum(y_clust))/(1 + γ * n_j)
            γ_j = γ / (1 + n_j * γ)
            s_j = s + n_j
            S_j = S + n_j / (1 + n_j * γ) * (y_bar - m)^2 + sum((y_clust .- y_bar) .^ 2)

            θ_unique[j] = rand(NormalInverseGamma(m_j, γ_j, s_j/2, S_j/2))
        end

        if n_sample > warmup
            samples[:ϕ][n_sample-warmup] = ϕ
            samples[:π][n_sample-warmup] = weight
            samples[:M][n_sample-warmup] = M
            samples[:m][n_sample-warmup] = m
            samples[:γ][n_sample-warmup] = γ
            samples[:c][n_sample-warmup, :] = c
            samples[:μ][n_sample-warmup, :] = [θ[1] for θ in θ_unique[c]]
            samples[:V][n_sample-warmup, :] = [θ[2] for θ in θ_unique[c]]

            # New θ
            all_values = [θ_unique; rand(NormalInverseGamma(m, γ, s/2, S/2))]
            wprobs = pweights([n_j_list; M])
            samples[:θ_new][n_sample-warmup] = StatsBase.sample(all_values, wprobs)
        end
    end

    return samples
end

function DpmNorm8()

end

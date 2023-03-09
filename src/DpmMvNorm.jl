function DpmMvNorm1f(y::Vector, prior_par, iter, warmup=floor(Int64, iter / 2))
    """
    SOME PARTS AREN'T FIXED, DON'T USE
    Implementation of algorithm 1 (Neal, 2000) for the Multivariate Normal model with 
    conjugate prior G0 and fixed hyperparameters. That is,

        y_i | θ_i ~ MvNorm(μ_i, Σ_i)
        θ_i | G ~ G
        G ~ DP(M, G0)
        G0 = NIW(m, γ, Ψ, ν) ≡ MvNorm(μ|m, γΣ)⋅IW(Σ|Ψ, ν)

    with parameterization such that E(Σ) = Ψ/(ν - p + 1)

    y         : data to fit the model
    prior_par : prior parameters (M, m, γ, ψ, ν)
    iter      : number of iterations including warmup
    """
    n = length(y)
    p = length(y[1])
    M, m, γ, Ψ, ν = prior_par
    total_samples = iter - warmup
    μ_samples = Array{Vector}(undef, total_samples, n)
    Σ_samples = Array{Matrix}(undef, total_samples, n)
    θ_new = Vector{Tuple}(undef, total_samples)

    # Posterior parameters
    m_p = [(m + γ * yi) / (1 + γ) for yi in y]
    γ_p = γ / (1 + γ)
    ν_p = ν + 1
    Ψ_p = [Ψ + (yi - m) * (yi - m)' / (1 + γ) for yi in y]
    prior_Σ = InverseWishart(ν, Ψ)
    post_Σ = [InverseWishart(ν_p, Psi) for Psi in Ψ_p]
    scale_t = sqrt((1 + γ) / (ν - p + 1) * Ψ)
    r_i = [M * pdf(MvTDist(ν - p + 1, m, scale_t), obs) for obs in y]

    # Initial values
    prev_μ = Vector{Vector}(undef, n)
    prev_Σ = Vector{Matrix}(undef, n)
    for i in 1:n
        prev_Σ[i] = rand(post_Σ[i])
        prev_μ[i] = rand(MvNormal(m_p[i], γ_p * prev_Σ[i]))
    end

    # Start of the algorithm
    for n_sample in 1:iter
        # Update of each component of θ
        counter_θ = Dict{Tuple,Int64}()
        for i in 1:n
            weights = Vector{Float64}(undef, n)
            weights[i] = r_i[i]                               # r_i
            id_c = 1:n .!= i                                  # index complement of {i}
            weights[id_c] = [                                 # q_ij
                pdf(MvNormal(x[1], x[2]), y[i]) for x in zip(prev_μ[id_c], prev_Σ[id_c])
            ]
            idx_new = StatsBase.sample(1:n, Weights(weights))
            if idx_new == i
                prev_Σ[i] = rand(post_Σ[i])
                prev_μ[i] = rand(MvNormal(m_p[i], γ * prev_Σ[i]))
            else
                prev_Σ[i] = prev_Σ[idx_new]
                prev_μ[i] = prev_μ[idx_new]
            end
            sample = (prev_μ[i], prev_Σ[i])
            freq = get(counter_θ, sample, 0)
            counter_θ[sample] = freq + 1
        end

        if n_sample > warmup
            μ_samples[n_sample-warmup, :] .= prev_μ
            Σ_samples[n_sample-warmup, :] .= prev_Σ

            # New θ value
            all_values = collect(keys(counter_θ))
            candidate_Σ = rand(prior_Σ)
            candidate_μ = rand(MvNormal(m, γ * candidate_Σ))
            candidate = (candidate_μ, candidate_Σ)
            push!(all_values, candidate)
            freqs = collect(values(counter_θ))
            norm_term = 1 / (M + n)
            probs = [freqs .* norm_term; M * norm_term]
            θ_new[n_sample-warmup] = StatsBase.sample(all_values, Weights(probs))
        end
    end

    return (μ_samples=μ_samples, Σ_samples=Σ_samples, θ_new=θ_new)
end

function _rand_nig(m, γ, ν, Ψ)
    Σ_samp = rand(InverseWishart(ν, Ψ))
    μ_samp = rand(MvNormal(m, γ * Σ_samp))
    return (μ_samp, Σ_samp)
end

function DpmMvNorm2(y, prior_par, iter, warmup=floor(Int64, iter / 2))
    """
    Implementation of Algorithm 2-3 (Neal, 2000) for the Multivariate-Normal model with conjugate
    prior G_η and random hyperparameters. That is,

        y_i | θ_i ∼ Normal_p(μ_i, Σ_i)
        θ_i | G ∼ G
        G ∼ DP(M, G_η)
        G_η = NIW(m, γ, ν, Ψ)
        m ∼ Normal_p(a, A)
        γ ∼ IG(w/2, W/2)
        M ∼ Gamma(α, β)

    y         : data to fit the model
    prior_par : prior parameters (a, A, w, W, α, β, ν, Ψ)
    iter      : number of iterations including warmup
    init_c    : initial value for c, with "same" they all start in the same cluster, "diff" to put
                them all in different clusters
    """
    n = length(y)
    p = length(y[1])
    a, A, w, W, α, β, ν, Ψ = prior_par
    total_samples = iter - warmup
    samples = Dict(
        :μ => Array{Vector}(undef, total_samples, n),
        :Σ => Array{Matrix}(undef, total_samples, n),
        :c => Array{Int64}(undef, total_samples, n),
        :m => Vector{Vector}(undef, total_samples),
        :γ => Vector{Float64}(undef, total_samples),
        :M => Vector{Float64}(undef, total_samples),
        :ϕ => Vector{Float64}(undef, total_samples),
        :π => Vector{Float64}(undef, total_samples),
        :θ_new => Vector{Tuple}(undef, total_samples)
    )

    # Initial values
    M = rand(Gamma(α, 1 / β))
    γ = rand(InverseGamma(w / 2, W / 2))
    m = rand(MvNormal(a, A))
    if init_c == "same"
        c = ones(Int64, n)
        θ_unique = [_rand_nig(m, γ, ν, Ψ)]
    elseif init_c == "diff"
        c = Vector{Int64}(1:n)
        θ_unique = [_rand_nig(m, γ, ν, Ψ) for _ in 1:n]
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
        Σ_unique = [θ[2] for θ in θ_unique]
        Λ_m = inv(inv(A) + 1/γ * sum(map(inv, Σ_unique)))
        aux = inv(A) * a + 1/γ * sum([inv(θ[2]) * θ[1] for θ in θ_unique])
        m = rand(MvNormal(Λ_m * aux, Λ_m))
        K = sum([transpose(m - θ[1]) * inv(θ[2]) * (m - θ[1]) for θ in θ_unique])
        γ = rand(InverseGamma((w + k * p) / 2, (W + K) / 2))

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
                ν_j_ = ν + n_j_
                Ψ_j_ = Ψ + n_j_ / (1 + n_j_ * γ) * transpose(y_bar_ - m) * (y_bar_ - m) +
                    sum([(y - y_bar_) * transpose(y - y_bar_) for y in y_clust_])

                scale_t = (1 + γ_j_) / (ν_j_ - p + 1) * Ψ_j_
                weights[idx] = n_j_ * pdf(MvTDist(ν_j_ - p + 1, m_j_, scale_t), y[i])
            end
            # New cluster
            scale_t = (1 + γ) / (ν - p + 1) * Ψ
            weights[k_ + 1] = M * pdf(MvTDist(ν - p + 1, m, scale_t), y[i])
            idx_cluster_new = StatsBase.sample(1:(k_ + 1), Weights(weights))
            if idx_cluster_new == (k_ + 1)
                if length(θ_unique) == k_
                    c[i] = maximum(c_unique_) + 1
                    push!(θ_unique, _rand_nig(m, γ, ν, Ψ))
                else
                    θ_unique[current_c] = _rand_nig(m, γ, ν, Ψ)
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

            m_j = (m + γ * sum(y_clust)) / (1 + γ * n_j)
            γ_j = γ / (1 + n_j * γ)
            ν_j = ν + n_j
            Ψ_j = Ψ + n_j / (1 + n_j * γ) * transpose(y_bar - m) * (y_bar - m) +
                    sum([(y - y_bar) * transpose(y - y_bar) for y in y_clust])

            θ_unique[j] = _rand_nig(m_j, γ_j, ν_j, Ψ_j)
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
            all_values = [θ_unique; _rand_nig(m, γ, ν, Ψ)]
            wprobs = pweights([n_j_list; M])
            samples[:θ_new][n_sample-warmup] = StatsBase.sample(all_values, wprobs)
        end
    end

    return samples
end

function DpmMvNorm8()

end

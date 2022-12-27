function DpmMvNorm1f(y::Vector, prior_par, iter, warmup=floor(Int64, iter/2))
    """
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
    M, m, γ, Ψ, S = prior_par
    total_samples = iter - warmup
    μ_samples = Array{Vector}(undef, total_samples, n)
    Σ_samples = Array{Matrix}(undef, total_samples, n)
    θ_new = Vector{Tuple}(undef, total_samples)

    # Posterior parameters
    m_p = [@. (m + γ * yi)/(1 + γ) for yi in y]
    γ_p = γ / (1 + γ)
    ν_p = ν + 1
    Ψ_p = [@. Ψ + (yi - m) ⋅ (yi - m)' / (1 + γ) for yi in y]
    scale_t = sqrt((1 + γ)/(ν - p + 1) * Ψ) # for r_i

    # Initial values
    prev_μ = Vector{Vector}(undef, n)
    prev_Σ = Vector{Matrix}(undef, n)
    for i in 1:n
        prev_Σ[i] = rand(InverseWishart(ν_p, Ψ_p[i]))
        prev_μ[i] = rand(MvNormal(m_p[i], γ * prev_Σ[i]))
    end

    # Start of the algorithm
    for n_sample in 1:iter
        # Update of each component of θ
        counter_θ = Dict{Tuple, Int64}()
        for i in 1:n
            weights = Vector{Float64}(undef, n)
            weights[i] = M * pdf(MvTDist(ν - p + 1, m, scale_t))
            weights[1:end.!i] = map(
                x -> pdf(MvNormal(x[1], x[2]), y[i]),
                zip(prev_μ[1:end.!=i], prev_Σ[1:end.!=i])
            )
            idx_new = StatsBase.sample(1:n, Weights(weights))
            if idx_new == i
                prev_Σ[i] = rand(InverseWishart(ν_p, Ψ_p[i]))
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
            candidate_Σ = rand(InverseWishart(ν, Ψ))
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

function DpmMvNorm1(y, prior_par, iter, warmup=floor(Int64, iter/2))
end

function DpmNorm1f(y, prior_par, iter, warmup=floor(Int64, iter / 2))
    """
    Implementation of algorithm 1 (Neal, 2000) for the normal model with conjugate prior G0
    and fixed hyperparameters. That is,

        y_i | θ_i ~ Normal(μ_i, V_i)
        θ_i | G ~ G
        G ~ DP(M, G0)
        G0 = NIG(m, λ, s/2, S/2)

    y         : data to fit the model
    prior_par : prior parameters (M, m, λ, s, S)
    """
    n = length(y)
    M, m, λ, s, S = prior_par
    total_samples = iter - warmup
    μ_samples = Array{Float64}(undef, total_samples, n)
    V_samples = Array{Float64}(undef, total_samples, n)

    # Posterior parameters
    m_p = @. (λ * m + y) / (λ + 1)
    λ_p = λ + 1
    s_p = s + 1
    S_p = @. S + λ / (λ + 1) * (y - m)^2
    scale_t = sqrt(S / s * (λ + 1) / λ) # for r_i

    # Initial values
    prev_μ = Vector{Float64}(undef, n)
    prev_V = Vector{Float64}(undef, n)
    for i in 1:n
        prev_V[i] = rand(InverseGamma(s_p / 2, S_p[i] / 2))
        prev_μ[i] = rand(Normal(m_p[i], sqrt(prev_V[i] / λ_p)))
    end

    # Start of the algorithm
    for n_sample in 1:iter
        # Update of each component of θ
        for i in 1:n
            weights = Vector{Float64}(undef, n)
            weights[i] = M * pdf(TDist(s), (y[i] - m) / scale_t) / scale_t # r_i
            weights[1:end.!=i] = map(                                      # q_ij
                x -> pdf(Normal(x[1], sqrt(x[2])), y[i]),
                zip(prev_μ[1:end.!=i, :], prev_V[1:end.!=i, :])
            )
            idx_new = StatsBase.sample(1:n, Weights(weights))
            if idx_new == i
                prev_V[i] = rand(InverseGamma(s_p / 2, S_p[i] / 2))
                prev_μ[i] = rand(Normal(m_p[i], sqrt(prev_V[i] / λ_p)))
            else
                prev_V[i] = prev_V[idx_new]
                prev_μ[i] = prev_μ[idx_new]
            end
        end

        if n_sample > warmup
            μ_samples[n_sample-warmup, :] .= prev_μ
            V_samples[n_sample-warmup, :] .= prev_V
        end
    end

    return (μ_samples=μ_samples, V_samples=V_samples)
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

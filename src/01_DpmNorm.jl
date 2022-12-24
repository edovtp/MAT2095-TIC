using Random
using Distributions

function DpmNorm1f(y, prior_par, iter, warmup=floor(Int64, iter / 2))
    """
    Implementation of algorithm 1 (Neal, 2000) for the normal model with conjugate prior G0
    and fixed hyperparameters. That is,

        y_i | θ_i ~ Normal(μ_i, V_i)
        θ_i | G ~ G
        G ~ DP
        G_0 = NGI(m, λ, s/2, S/2)

    y         : data to fit the model
    prior_par : prior parameters (alpha, m, λ, s, S)
    """
    n = length(y)
    α, m, λ, s, S = prior_par
    total_samples = iter - warmup
    μ_samples = Array{Float64}(undef, total_samples, n)
    V_samples = Array{Float64}(undef, total_samples, n)

    # Posterior parameters
    @. m_p = (λ*m + y) / (λ + 1)
    λ_p = λ + 1
    s_p = s + 1
    @. S_p = S + λ / (λ + 1) * (y[i] - m)^2
    scale_t = sqrt(S/s * (λ + 1)/λ) # for r_i

    # Initial values
    prev_μ = Vector{Float64}(undef, n)
    prev_V = Vector{Float64}(undef, n)
    for i in 1:n
        post_1 = InverseGamma(s_p/2, S_p[i]/2)
        prev_V[i] = rand(post_1)
        post2 = Normal(m[i], sqrt(prev_V / λ_p))
        prev_μ[i] = rand(post_2)
    end

    # Start of the algorithm
    for n_sample in 1:iter
        # Update of each component of θ
        for i in 1:n
            weights = Vector{Float64}(undef, n)
            # r_i
            weights[i] = α * pdf(TDist(s), (y[i] - m)/scale_t) / scale_t
            # q_ij
            weights[1:end.!=i] = map(
                x -> pdf(Normal(x[1], sqrt(x[2])), y[i]),
                eachrow(zip(prev_μ[1:end.!=i, :], prev_V[1:end.!=i, :]))
            )
    end
end

function DpmNorm1()

end

function DpmNorm2f()

end

function DpmNorm2()
end


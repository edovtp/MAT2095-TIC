include("00_extras.jl")
include("01_DP.jl")
include("02_DPM.jl")


# Fixed hyperparameters
function tic_dpm_ew_fixed(y, prior_par, iter, warmup=floor(Int64, iter / 2))
    """
    Implementation of the first algorithm given in Escobar & West (1995)

    y         : data to fit the model
    prior_par : prior parameters (alpha, m, tau, s and S)
    """
    n = length(y)
    alpha, m, tau, s, S = prior_par
    samples = Array{Float64,3}(undef, (iter + 1, n, 2))

    # Initial values
    for i in 1:n
        xi = (m + tau * y[i]) / (1 + tau)
        X = tau / (1 + tau)
        Si = S + (y[i] - m)^2 / (1 + tau)

        post = NormalInverseGamma(xi, X, (1 + s) / 2, Si / 2)
        samples[1, i, :] .= rand(post)
    end

    # Start of the algorithm
    prev_sample = samples[1, :, :]
    for n_sample in 2:(iter+1)
        # Update of each component
        for i in 1:n
            xi = (m + tau * y[i]) / (1 + tau)
            X = tau / (1 + tau)
            Si = S + (y[i] - m)^2 / (1 + tau)
            M = (1 + tau) * S / s
            q_weights = Vector{Float64}(undef, n)

            q_weights[i] = alpha * pdf(TDist(s), (y[i] - m) / sqrt(M)) / sqrt(M)
            q_weights[1:end.!=i] = map(
                x -> pdf(Normal(x[1], sqrt(x[2])), y[i]),
                eachrow(prev_sample[1:end.!=i, :])
            )

            idx_new = StatsBase.sample(1:n, Weights(q_weights))
            if idx_new == i
                prev_sample[i, :] .= rand(NormalInverseGamma(xi, X, (1 + s) / 2, Si / 2))
            else
                prev_sample[i, :] .= prev_sample[idx_new, :]
            end
        end

        samples[n_sample, :, :] .= prev_sample
    end

    return samples[(warmup+2):(iter+1), :, :]
end

function tic_dpm_ew(y, prior_par, iter, warmup=floor(Int64, iter / 2))
    """
    Implementation of the final algorithm given in Escobar & West (1995) to sample from
    the posterior distribution of a DPM model of normals.

    y         : data to fit the model
    prior_par : prior parameters (a, b, A, w, W, s, S)
    """
    n = length(y)
    a, b, A, w, W, s, S = prior_par
    et_samples = Vector{Float64}(undef, iter)
    al_samples = Vector{Float64}(undef, iter + 1)
    mt_samples = Array{Float64,2}(undef, (iter + 1, 2))
    pi_samples = Array{Float64,3}(undef, (iter + 1, n, 2))

    # Initial values
    ## alpha
    alpha = rand(Gamma(a, 1 / b))
    al_samples[1] = alpha

    ## m and tau
    tau = rand(InverseGamma(w / 2, W / 2))
    m = rand(Normal(0, 1))
    mt_samples[1, :] = [m, tau]

    ## pi
    for i in 1:n
        xi = (m + tau * y[i]) / (1 + tau)
        X = tau / (1 + tau)
        Si = S + (y[i] - m)^2 / (1 + tau)

        post = NormalInverseGamma(xi, X, (1 + s) / 2, Si / 2)
        pi_samples[1, i, :] .= rand(post)
    end

    # Start of the algorithm
    prev_pi = pi_samples[1, :, :]
    for n_sample in 2:(iter+1)
        unique_pi = unique(prev_pi, dims=1)
        k = size(unique_pi)[1]

        ## Update alpha
        eta = rand(Beta(alpha + 1, n))
        odds_weight = (a + k - 1) / (n * (b - log(eta)))
        weight = odds_weight / (1 + odds_weight)
        component = sample([0, 1], Weights([weight, 1 - weight]))
        alpha = rand(Gamma(a + k - 1 * component, 1 / (b - log(eta))))

        ## Update m and tau
        Vbar = 1 / sum(1 ./ unique_pi[:, 2])
        x = A / (A + tau * Vbar)
        m = rand(
            Normal(x * Vbar * sum(unique_pi[:, 1] ./ unique_pi[:, 2]), sqrt(x * tau * Vbar))
        )
        K = sum((unique_pi[:, 1] .- m) .^ 2 ./ unique_pi[:, 2])
        tau = rand(InverseGamma((w + k) / 2, (W + K) / 2))

        ## Update pi for each observation
        xi = (m .+ tau .* y) ./ (1 + tau)
        X = tau / (1 + tau)
        Si = S .+ (y .- m) .^ 2 ./ (1 + tau)
        M = (1 + tau) * S / s
        for i in 1:n
            q_weights = Vector{Float64}(undef, n)
            q_weights[i] = alpha * pdf(TDist(s), (y[i] - m) / sqrt(M)) / sqrt(M)
            q_weights[1:end.!=i] = map(
                x -> pdf(Normal(x[1], sqrt(x[2])), y[i]),
                eachrow(prev_pi[1:end.!=i, :])
            )

            idx_new = StatsBase.sample(1:n, Weights(q_weights))
            if idx_new == i
                prev_pi[i, :] .= rand(
                    NormalInverseGamma(xi[i], X, (1 + s) / 2, Si[i] / 2)
                )
            else
                prev_pi[i, :] .= prev_pi[idx_new, :]
            end
        end

        # Save the values
        et_samples[n_sample-1] = eta
        al_samples[n_sample] = alpha
        mt_samples[n_sample, :] .= [m, tau]
        pi_samples[n_sample, :, :] .= prev_pi
    end

    return (et_samples[(warmup+1):end],
        al_samples[(warmup+2):end],
        mt_samples[(warmup+2):end, :],
        pi_samples[(warmup+2):end, :, :])
end

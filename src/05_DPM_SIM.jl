include("00_extras.jl")
include("01_DP.jl")
include("02_DPM.jl")


# Univariate normal with conjugate G0
#region
function dpm_normal(y, prior_par, iter, warmup, fixed=false)
    if fixed == true
        return _dpm_norm_neal_fixed(y, prior_par, iter, warmup)
    else
        return _dpm_norm_neal(y, prior_par, iter, warmup)
    end
end

function _dpm_norm_ew_fix(y, prior_par, iter, warmup=floor(Int64, iter / 2))
    """
    Implementation of the first algorithm given in Escobar & West (1995) for the normal
    model with conjugate G0 and fixed hyperparameters

    y         : data to fit the model
    prior_par : prior parameters (alpha, m, tau, s and S)
    """
    n = length(y)
    alpha, m, tau, s, S = prior_par
    samples = Array{Float64,3}(undef, (iter - warmup, n, 2))

    # Initial values
    for i in 1:n
        xi = (m + tau * y[i]) / (1 + tau)
        X = tau / (1 + tau)
        Si = S + (y[i] - m)^2 / (1 + tau)

        post = NormalInverseGamma(xi, X, (1 + s) / 2, Si / 2)
        prev_sample .= rand(post)
    end

    # Start of the algorithm
    for n_sample in 1:iter
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

        if n_sample > warmup
            samples[n_sample - warmup, :, :] .= prev_sample
        end
    end

    return samples
end

function _dpm_norm_ew(y, prior_par, iter, warmup=floor(Int64, iter / 2))
    """
    Implementation of the last algorithm given in Escobar & West (1995) for the normal
    model with conjugate G0 and random hyperparameters. In particular,

        y_i | π_i ∼ N(μ_i, V_i)
        π_i | G   ∼ DP(α, G_0)
              G_0 = N-Inv-Gamma(m, 1/τ, s/2, S/2)
              τ   ∼ Inv-Gamma(w/2, W/2)
              m   ∼ N(0, A)
              α   ∼ Gamma(a, b)

    y         : data to fit the model
    prior_par : prior parameters (a, b, A, w, W, s, S)
    """
    n = length(y)
    a, b, A, w, W, s, S = prior_par
    total_samples = iter - warmup
    hyp_samples = NamedArray{Float64}((total_samples, 4))
    setnames!(hyp_samples, ["eta", "alpha", "m", "tau"], 2)
    pi_samples = NamedArray{Float64}((total_samples, n, 2))
    setnames!(pi_samples, ["mu", "V"], 3)

    # Initial values
    ## alpha
    alpha = rand(Gamma(a, 1 / b))

    ## m and tau
    tau = rand(InverseGamma(w / 2, W / 2))
    m = rand(Normal(0, 1))

    ## pi
    prev_pi = Array{Float64, 2}(undef, (n, 2))
    for i in 1:n
        xi = (m + tau * y[i]) / (1 + tau)
        X = tau / (1 + tau)
        Si = S + (y[i] - m)^2 / (1 + tau)

        post = NormalInverseGamma(xi, X, (1 + s) / 2, Si / 2)
        prev_pi[i, :] .= rand(post)
    end

    # Start of the algorithm
    for n_sample in 1:iter
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
        if n_sample > warmup
            hyp_samples[n_sample - warmup, :] .= (eta, alpha, m, tau)
            pi_samples[n_sample - warmup, :, :] .= prev_pi
        end
    end

    return (hyp_samples, pi_samples)
end

function _dpm_norm_neal_fixed(y, prior_par, iter, warmup=floor(Int64, iter/2))
    """
    Implementation of Algorithm 8 given in Neal (2000) for the normal model with conjugate
    G0. In particular,

    y_i | π_i ∼ N(μ_i, V_i)
    π_i | G   ∼ DP(α, G_0)
          G_0 = N-Inv-Gamma(m, 1/τ, s/2, S/2)

    y         : data to fit the model
    prior_par : prior parameters (alpha, m, tau, s and S)
    """
    # Samples
    n = length(y)
    alpha, m, tau, s, S = prior_par
    c_samples = Array{Float64, 2}(undef, (iter + 1, 2))
    phi_samples = Array{Array{Float64}}(undef)
    theta_samples = Array{Float64, 3}(undef, (iter + 1, n, 2))
end

function _dpm_norm_neal(y, prior_par, iter, warmup=floor(Int64, iter/2))
end

## ONLY FOR ILLUSTRATION PURPOSES
function _dpm_norm_bm()
end
#endregion

# Multivariate normal with conjugate G0
#region
function dpm_mvnorm(y, prior_par, iter, warmup, algorithm="neal", fixed=false)
end

function _dpm_mvnorm_neal_fixed()
end

function _dpm_mvnorm_neal()
end
#endregion

#region
function dpm_neal_normal1()
end


function tic_dpm_neal(
    y::Array,
    F_y::UnionAll,
    alpha::Real,
    G0::Distribution,
    m::Int64,
    phi_sampler::Function,
    iter::Int64,
    warmup=floor(Int64, iter / 2)
)
    """
    Implementation of Algorithm 8 given in Neal (2000)
    # TODO: Array to save and return data

    y           : data to fit the model
    F_y         : sampling model
    G0          : centering measure of the DP
    alpha       : precision parameter
    phi_sampler : custom sampler for the posterior of phi, receives phi, c and y
    """
    # Sample arrays
    c_samples = Array{Float64, 2}(undef, (iter + 1, 2))
    theta_samples = Array{Float64}

    # Starting values
    c = ones(Int64, n)
    phi = [rand(G0)]

    for _ in 1:iter
        # Update cluster memberships
        c, phi = _n8_update_clusters(c, phi, y, F_y, alpha, G0, m)

        # Update phi
        phi = phi_sampler(phi, c, y, G0)

        # Print
        # println(c)
        end
end

function tic_dpm_neal(y, prior_par, iter, warmup=floor(Int64, iter/2))
end
#endregion

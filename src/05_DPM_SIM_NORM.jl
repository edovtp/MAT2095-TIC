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

function _dpm_norm_ew_fixed(y, prior_par, iter, warmup=floor(Int64, iter / 2))
    """
    Implementation of the first algorithm given in Escobar & West (1995) for the
    normal model with conjugate G0. In particular,

        y_i | π_i ∼ N(μ_i, V_i)
          π_i | G ∼ G
                G ∼ DP
              G_0 = N-Inv-Gamma(m, τ, s/2, S/2)

        i.e. G_0 ≡ N(μ | m, τV) ⋅ Inv-Gamma(V | s/2, S/2) (scale parameterization)

    y         : data to fit the model
    prior_par : prior parameters (alpha, m, tau, s and S)
    """
    n = length(y)
    alpha, m, tau, s, S = prior_par
    total_samples = iter - warmup
    pi_samples = NamedArray{Float64}((total_samples, n, 2))
    setnames!(pi_samples, ["mu", "V"], 3)

    # Initial values
    prev_pi = Array{Float64,2}(undef, (n, 2))
    for i in 1:n
        xi = (m + tau * y[i]) / (1 + tau)
        X = tau / (1 + tau)
        Si = S + (y[i] - m)^2 / (1 + tau)

        post = NormalInverseGamma(xi, X, (1 + s) / 2, Si / 2)
        prev_pi[i, :] .= rand(post)
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
                eachrow(prev_pi[1:end.!=i, :])
            )

            idx_new = StatsBase.sample(1:n, Weights(q_weights))
            if idx_new == i
                prev_pi[i, :] .= rand(NormalInverseGamma(xi, X, (1 + s) / 2, Si / 2))
            else
                prev_pi[i, :] .= prev_pi[idx_new, :]
            end
        end

        if n_sample > warmup
            pi_samples[n_sample-warmup, :, :] .= prev_pi
        end
    end

    return pi_samples
end

function _dpm_norm_ew(y, prior_par, iter, warmup=floor(Int64, iter / 2))
    """
    Implementation of the last algorithm given in Escobar & West (1995) for the normal
    model with conjugate G0 and random hyperparameters. In particular,

        y_i | π_i ∼ N(μ_i, V_i)
          π_i | G ∼ G
                G ∼ DP(α, G_0)
              G_0 = N-Inv-Gamma(m, τ, s/2, S/2)
                τ ∼ Inv-Gamma(w/2, W/2)
                m ∼ N(0, A)
                α ∼ Gamma(a, b)

    that is, G_0 ≡ N(μ | m, τV) ⋅ Inv-Gamma(V | s/2, S/2) (scale parameterization)

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
    m = rand(Normal(0, A))

    ## pi
    prev_pi = Array{Float64,2}(undef, (n, 2))
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
            hyp_samples[n_sample-warmup, :] .= (eta, alpha, m, tau)
            pi_samples[n_sample-warmup, :, :] .= prev_pi
        end
    end

    return (hyp_samples, pi_samples)
end

function _dpm_norm_neal_fixed(y, prior_par, m_extra, iter, warmup=floor(Int64, iter / 2))
    """
    Implementation of Algorithm 8 given in Neal (2000) for the normal model with conjugate
    G0. In particular,

    y_i | π_i ∼ N(μ_i, V_i)
    π_i | G   ∼ DP(α, G_0)
          G_0 = N-Inv-Gamma(m, 1/τ, s/2, S/2)

    y         : data to fit the model
    prior_par : prior parameters (alpha, m, tau, s and S)
    """
    n = length(y)
    alpha, m, tau, s, S = prior_par
    G0 = NormalInverseGamma(m, tau, s/2, S/2)

    # Samples
    total_samples = iter - warmup
    c_samples = Array{Int64,2}(undef, (total_samples, n))
    pi_samples = NamedArray{Float64}((total_samples, n, 2))
    setnames!(pi_samples, ["mu", "V"], 3)

    # Initial values
    c = ones(Int64, n)
    phi = [rand(G0)]

    # Auxiliar function to relabel
    function _relabel(c, phi)
        c_re = Vector{Int64}(undef, length(c))

        c_unique = unique(c)
        phi_re = phi[c_unique]
        for i in 1:length(c_unique)
            c_re[c.==c_unique[i]] .= i
        end

        return c_re, phi_re
    end

    # Start of the algorithm
    for n_sample in 1:iter
        # Update cluster memberships
        for i in 1:n
            current_c = c[i]
            c_i = c[1:end.!=i]
            k_ = length(unique(c_i))
            h = k_ + m_extra

            if current_c in c_i
                phi_aug = [phi; [rand(G0) for _ in 1:(h-k_)]]
            else
                # If the current cluster is the last, do nothing
                max_c_i = maximum(c_i)
                if current_c < max_c_i
                    # Cambiar también en c
                    # Swap cluster labels and phi
                    c_i[c_i.==max_c_i] .= current_c
                    phi_tmp = phi[current_c]
                    phi[current_c] = phi[max_c_i]

                    current_c = max_c_i
                    phi[max_c_i] = phi_tmp
                end

                # Augmentation of phi
                phi_aug = [phi; [rand(G0) for _ in 1:(h-k_-1)]]
            end

            # Weights
            c_unique = unique(c_i)
            weights = Vector{Float64}(undef, h)
            for cluster in c_unique
                # Contar desde antes
                n_clust = count(==(cluster), c_i)
                mu_clust = phi_aug[cluster][1]
                sigma_clust = sqrt(phi_aug[cluster][2])
                weights[cluster] = n_clust * pdf(Normal(mu_clust, sigma_clust), y[i])
            end
            for j in (k_+1):h
                mu_aug = phi_aug[j][1]
                sigma_aug = sqrt(phi_aug[j][2])
                weights[j] = alpha / m_extra * pdf(Normal(mu_aug, sigma_aug), y[i])
            end

            # Change of state
            new_clust = sample(1:h, Weights(weights))
            if new_clust <= k_
                # Revisar el caso que se elimina
                c[i] = new_clust
            else
                c[i] = k_ + 1
                push!(phi, phi_aug[new_clust])
            end

            # Relabel and drop phi values that are not associated with any observation
            c, phi = _relabel(c, phi)
            phi = phi[sort(unique(c))]
        end

        # Update cluster parameters
        c_unique = unique(c)
        for cluster in c_unique
            y_clust = y[c.==cluster]
            n = length(y_clust)
            sum_y = sum(y_clust)
            y_bar = sum_y / n

            # Posterior parameters
            mu_n = (G0.mu + G0.v0 * sum_y) / (1 + G0.v0 * n)
            nu_n = G0.v0 / (n + G0.v0)
            sh_n = G0.shape / 2 + n / 2
            sc_n = G0.scale / 2 + sum((y_clust .- y_bar) .^ 2) / 2 +
                   n / (1 + n * G0.v0) * (y_bar - G0.mu)^2 / 2

            G_c = NormalInverseGamma(mu_n, nu_n, sh_n, sc_n)
            phi[cluster] = rand(G_c)
        end

        if n_sample > warmup
            c_samples[n_sample-warmup, :] .= c
            pi_samples[n_sample-warmup, :, :] .=
                reshape(reinterpret(Float64, phi[c]), (2, :))'
        end
    end

    return c_samples, pi_samples
end

function _dpm_norm_neal(y, prior_par, m_extra, iter, warmup=floor(Int64, iter / 2))
    """
    Implementation of Algorithm 8 given in Neal (2000) for the normal model with conjugate
    G0 and random hyperparameters. In particular,

        y_i | π_i ∼ N(μ_i, V_i)
        π_i | G ∼ G
            G ∼ DP(α, G_0)
            G_0 = N-Inv-Gamma(m, τ, s/2, S/2)
            τ ∼ Inv-Gamma(w/2, W/2)
            m ∼ N(0, A)
            α ∼ Gamma(a, b)

    i.e. G_0 ≡ N(μ | m, τV) ⋅ Inv-Gamma(V | s/2, S/2) (scale parameterization)

    y         : data to fit the model
    prior_par : prior parameters (a, b, A, w, W, s, S)
    """
    n = length(y)
    a, b, A, w, W, s, S = prior_par

    # Samples
    total_samples = iter - warmup
    c_samples = Array{Int64,2}(undef, (total_samples, n))
    hyp_samples = NamedArray{Float64}((total_samples, 4))
    setnames!(hyp_samples, ["eta", "alpha", "m", "tau"], 2)
    pi_samples = NamedArray{Float64}((total_samples, n, 2))
    setnames!(pi_samples, ["mu", "V"], 3)

    # Initial values
    ## alpha
    alpha = rand(Gamma(a, 1/b))

    ## m and tau
    tau = rand(InverseGamma(w/2, W/2))
    m = rand(Normal(0, A))

    ## Cluster memberships
    c = ones(Int64, n)

    ## phi
    G0 = NormalInverseGamma(m, tau, s/2, S/2)
    phi = [rand(G0)]

    # Auxiliar function to relabel
    function _relabel(c, phi)
        c_re = Vector{Int64}(undef, length(c))

        c_unique = unique(c)
        phi_re = phi[c_unique]
        for i in 1:length(c_unique)
            c_re[c.==c_unique[i]] .= i
        end

        return c_re, phi_re
    end

    # Start of the algorithm
    for n_sample in 1:iter
        # Update alpha
        k = length(unique(c))
        eta = rand(Beta(alpha + 1, n))
        odds_weight = (a + k - 1) / (n * (b - log(eta)))
        weight = odds_weight / (1 + odds_weight)
        component = sample([0, 1], Weights([weight, 1 - weight]))
        alpha = rand(Gamma(a + k - 1 * component, 1 / (b - log(eta))))

        # Update m and tau
        unique_pi = reshape(reinterpret(Float64, phi[c]), (2, :))'
        Vbar = 1 / sum(1 ./ unique_pi[:, 2])
        x = A / (A + tau * Vbar)
        m = rand(
            Normal(x * Vbar * sum(unique_pi[:, 1] ./ unique_pi[:, 2]), sqrt(x * tau * Vbar))
        )
        K = sum((unique_pi[:, 1] .- m) .^ 2 ./ unique_pi[:, 2])
        tau = rand(InverseGamma((w + k) / 2, (W + K) / 2))

        # Update cluster memberships
        for i in 1:n
            current_c = c[i]
            c_i = c[1:end.!=i]
            k_ = length(unique(c_i))
            h = k_ + m_extra

            if current_c in c_i
                phi_aug = [phi; [rand(G0) for _ in 1:(h-k_)]]
            else
                # If the current cluster is the last, do nothing
                max_c_i = maximum(c_i)
                if current_c < max_c_i
                    # Cambiar también en c
                    # Swap cluster labels and phi
                    c_i[c_i.==max_c_i] .= current_c
                    phi_tmp = phi[current_c]
                    phi[current_c] = phi[max_c_i]

                    current_c = max_c_i
                    phi[max_c_i] = phi_tmp
                end

                # Augmentation of phi
                phi_aug = [phi; [rand(G0) for _ in 1:(h-k_-1)]]
            end

            # Weights
            c_unique = unique(c_i)
            weights = Vector{Float64}(undef, h)
            for cluster in c_unique
                # Contar desde antes
                n_clust = count(==(cluster), c_i)
                mu_clust = phi_aug[cluster][1]
                sigma_clust = sqrt(phi_aug[cluster][2])
                weights[cluster] = n_clust * pdf(Normal(mu_clust, sigma_clust), y[i])
            end
            for j in (k_+1):h
                mu_aug = phi_aug[j][1]
                sigma_aug = sqrt(phi_aug[j][2])
                weights[j] = alpha / m_extra * pdf(Normal(mu_aug, sigma_aug), y[i])
            end

            # Change of state
            new_clust = sample(1:h, Weights(weights))
            if new_clust <= k_
                # Revisar el caso que se elimina
                c[i] = new_clust
            else
                c[i] = k_ + 1
                push!(phi, phi_aug[new_clust])
            end

            # Relabel and drop phi values that are not associated with any observation
            c, phi = _relabel(c, phi)
            phi = phi[sort(unique(c))]
        end

        # Update cluster parameters
        c_unique = unique(c)
        for cluster in c_unique
            y_clust = y[c.==cluster]
            n = length(y_clust)
            sum_y = sum(y_clust)
            y_bar = sum_y / n

            # Posterior parameters
            mu_n = (G0.mu + G0.v0 * sum_y) / (1 + G0.v0 * n)
            nu_n = G0.v0 / (n + G0.v0)
            sh_n = G0.shape / 2 + n / 2
            sc_n = G0.scale / 2 + sum((y_clust .- y_bar) .^ 2) / 2 +
                   n / (1 + n * G0.v0) * (y_bar - G0.mu)^2 / 2

            G_c = NormalInverseGamma(mu_n, nu_n, sh_n, sc_n)
            phi[cluster] = rand(G_c)
        end

        if n_sample > warmup
            c_samples[n_sample-warmup, :] .= c
            pi_samples[n_sample-warmup, :, :] .=
                reshape(reinterpret(Float64, phi[c]), (2, :))'
        end
    end

    return c_samples, pi_samples
end

## ONLY FOR ILLUSTRATION PURPOSES
function _dpm_norm_bm()
end
#endregion

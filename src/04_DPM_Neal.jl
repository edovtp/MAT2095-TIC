include("00_extras.jl")
include("01_DP.jl")
include("02_DPM.jl")


function _relabel(c, phi)
    c_re = Vector{Int64}(undef, length(c))

    c_unique = unique(c)
    phi_re = phi[c_unique]
    for i in 1:length(c_unique)
        c_re[c.==c_unique[i]] .= i
    end

    return c_re, phi_re
end

function _n8_update_clusters(c, phi, y, F_y, alpha, G0, m)
    n = length(y)
    for i in 1:n
        current_c = c[i]
        c_i = c[1:end.!=i]
        k_ = length(unique(c_i))
        h = k_ + m

        if current_c in c_i
            phi_aug = [phi; [rand(G0) for _ in 1:(h-k_)]]
        else
            # If the current cluster is the last, do nothing
            max_c_i = maximum(c_i)
            if current_c < max_c_i
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
            n_clust = count(==(cluster), c_i)
            mu_clust = phi_aug[cluster][1]
            sigma_clust = sqrt(phi_aug[cluster][2])
            weights[cluster] = n_clust * pdf(F_y(mu_clust, sigma_clust), y[i])
        end
        for j in (k_+1):h
            mu_aug = phi_aug[j][1]
            sigma_aug = sqrt(phi_aug[j][2])
            weights[j] = alpha / m * pdf(F_y(mu_aug, sigma_aug), y[i])
        end

        # Change of state
        new_clust = sample(1:h, Weights(weights))
        if new_clust <= k_
            c[i] = new_clust
        else
            c[i] = k_ + 1
            push!(phi, phi_aug[new_clust])
        end

        # Relabel and drop phi values that are not associated with any observation
        c, phi = _relabel(c, phi)
        phi = phi[sort(unique(c))]
    end

    return c, phi
end

function _n8_update_hyper()

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
    # TODO: Update hyperparameters

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

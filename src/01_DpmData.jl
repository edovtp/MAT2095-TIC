function DataDp(n::Int64, M, par, dist)
    if dist == "Nig"
        return DataDpNig(n, M, par)
    elseif dist == "Niw"
        return DataDpNiw(n, M, par)
    end
end

function DataDpm(n::Int64, M, par, dist)
    if dist == "Normal"
        return DataDpmNormal(n, M, par)
    elseif dist == "MvNormal"
        return DataDpmMvNormal(n, M, par)
    end
end

# Specific functions
function DataDpNig(n::Int64, M, par)
    """
    Data simulation from a Dirichlet process with Normal-Inv-Gamma base measure using the
    Pólya Urn representation of Blackwell and McQueen (1973).
    
    The parameterization of the base measure is such that γ scales σ. That is,
    G0 = Normal(μ|m, γσ^2)⋅Inv-Gamma(σ^2|s, S)

    n   : sample length
    M   : precision parameter
    par : NIG parameters (m, γ, s, S)
    """
    m, γ, s, S = par
    marginal_sample = ElasticVector{Tuple}(undef, 0)
    counter = Dict{Tuple,Int64}()

    for i in 1:n
        all_values = collect(keys(counter))
        candidate = rand(NormalInverseGamma(m, γ, s, S))
        push!(all_values, candidate)

        norm_term = 1 / (M + i - 1)
        old_freq = collect(values(counter))
        probs = [old_freq .* norm_term; M * norm_term]

        new_value = StatsBase.sample(all_values, Weights(probs))
        push!(marginal_sample, new_value)

        freq = get(counter, new_value, 0)
        counter[new_value] = freq + 1
    end

    return marginal_sample
end

function DataDpmNormal(n::Int64, M, par)
    """
    Random sample from a DPM of Normals with Normal-Inv-Gamma base measure. Returns the
    component parameters, the number of unique components (k) and the simulated data.

    The parameterization of the base measure is such that γ scales σ. That is,
    G0 = Normal(μ|m, γσ^2)⋅Inv-Gamma(σ^2|s, S)

    n   : sample length
    M   : precision parameter
    par : NIG parameters (m, γ, s, S)
    """
    θ = DataDpNig(n, M, par)
    k = size(unique(θ))[1]
    y = [rand(Normal(par[1], sqrt(par[2]))) for par in θ]

    return (y=y, θ=θ, k=k)
end

function DataDpNiw(n::Int64, M, par)
    """
    Data simulation from a Dirichlet process with Normal-Inv-Wishart base measure using 
    the Pólya Urn representation of Blackwell and McQueen (1973).

    The parameterization of the base measure is such that γ scales Σ. That is,
    G0 = MvNormal(μ|m, γΣ)⋅Inv-Wishart(Σ|Ψ, ν), and E(Σ) = Ψ/(ν - p + 1) with p the
    dimension of the data.

    n   : sample length
    M   : precision parameter
    par : NIW parameters (m, γ, Ψ, ν)
    """
    m, γ, Ψ, ν = par
    marginal_sample = ElasticVector{Tuple}(undef, 0)
    counter = Dict{Tuple,Int64}()

    for i in 1:n
        all_values = collect(keys(counter))
        candidate_Σ = rand(InverseWishart(ν, Ψ))
        candidate_μ = rand(MvNormal(m, γ * candidate_Σ))
        candidate = (candidate_μ, candidate_Σ)
        push!(all_values, candidate)

        norm_term = 1 / (M + i - 1)
        old_freq = collect(values(counter))
        probs = [old_freq .* norm_term; M * norm_term]

        new_value = StatsBase.sample(all_values, Weights(probs))
        push!(marginal_sample, new_value)

        freq = get(counter, new_value, 0)
        counter[new_value] = freq + 1
    end

    return marginal_sample
end

function DataDpmMvNormal(n::Int64, M, par)
    """
    Random sample from a DPM of Multivariate Normals with Normal-Inv-Wishart base measure.
    Returns the component parameters, the number of unique components (k) and the simulated
    data.

    The parameterization of the base measure is such that γ scales Σ. That is,
    G0 = MvNormal(μ|m, γΣ)⋅Inv-Wishart(Σ|Ψ, ν), and E(Σ) = Ψ/(ν - p + 1) with p the
    dimension of the data.

    n   : sample length
    M   : precision parameters
    par : NIW parameters (m, γ, Ψ, ν)
    """
    θ = DataDpNiw(n, M, par)
    k = size(unique(θ))[1]
    y = [rand(MvNormal(par[1], par[2])) for par in θ]

    return(y=y, θ=θ, k=k)
end

function DataDpm(n::Int64, M, par, dist)
    if dist == "Normal"
        return DataDpmNormal(n, M, par)
    elseif dist == "MvNormal"
        return DataDpmMvNormal(n, M, par)
    end
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

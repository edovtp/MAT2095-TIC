include("00_extras.jl")
include("01_DP.jl")
include("02_DPM.jl")


# Multivariate normal with conjugate G0
#region
function dpm_mvnorm(y, prior_par, iter, warmup, algorithm="neal", fixed=false)
end

function _dpm_mvnorm_ew_fixed()
    """
    Implementation of the first algorithm given in Escobar & West (1995) for the
    multivariate normal model with conjugate G0. In particular,

        y_i | π_i ∼ MVNorm(μ_i, Σ_i)
          π_i | G ∼ G
                G ∼ DP(α, G_0)
              G_0 = MVNorm-Inv-Wishart(m, τ, Ψ, ν)

        i.e. G_0 ≡ N(μ|m, τΣ) ⋅ Inv-Wishart(Σ|Ψ, ν)
    """
end

function _dpm_mvnorm_ew()
    """
    Implementation of the last algorithm given in Escobar & West (1995) for the
    multivariate normal model with conjugate G0 and random hyperparameters. In particular,

        y_i | π_i ∼ MVNorm(μ_i, Σ_i)
          π_i | G ∼ G
                G ∼ DP(α, G_0)
              G_0 = MVNorm-Inv-Wishart(m, τ, Ψ, ν)
                τ ∼ Inv-Gamma(w/2, W/2)
                m ∼ N(0, A*I)
                α ∼ Gamma(a, b)

        i.e. G_0 ≡ N(μ|m, τΣ) ⋅ Inv-Wishart(Σ|Ψ, ν)
    """
end

function _dpm_mvnorm_neal_fixed()
end

function _dpm_mvnorm_neal()
end
#endregion

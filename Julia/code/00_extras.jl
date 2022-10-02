using ConjugatePriors


"""
Extensions of the Normal-Inverse-Gamma distribution

Code taken from 
https://github.com/JuliaStats/ConjugatePriors.jl/blob/master/src/normalinversegamma.jl
"""

Base.length(d::NormalInverseGamma) = 2

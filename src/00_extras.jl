using Random
using StatsBase
using Distributions
using ElasticArrays
using ConjugatePriors
using Plots
using StatsPlots
using FreqTables

"""
Extensions of the Normal-Inverse-Gamma distribution

Code taken from
https://github.com/JuliaStats/ConjugatePriors.jl/blob/master/src/normalinversegamma.jl
"""

Base.length(d::NormalInverseGamma) = 2

"""
Location-Scale t-distribution

See https://github.com/JuliaStats/Distributions.jl/pull/1482#discussion_r786065465
"""

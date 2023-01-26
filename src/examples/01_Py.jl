using CairoMakie
include("../helpers.jl")
include("../Py.jl")

## Sampling
Random.seed!(619);
precisions = [1, 25, 250];
G0 = Gamma(6, 1);
py_samples_1 = [Py(1, 0.6, G0) for _ in 1:10];
py_samples_50 = [Py(50, 0.6, G0) for _ in 1:10];
py_samples_250 = [Py(250, 0.6, G0) for _ in 1:10];

## Figure
begin
    CairoMakie.activate!()
    fig = Figure(resolution=(1440, 500))
    xrange = range(0, 15, length=100)

    # M = 1
    ax_1 = Axis(fig[1, 1], title="M=1", xgridvisible=false, ygridvisible=false,
        ylabel="Probabilidad")
    lines!(xrange, x -> cdf(G0, x), color=:red)
    for py_1 in py_samples_1
        ecdfplot!(py_1.locations, weights=py_1.weights, color=(:black, 0.2))
    end

    # M = 50
    ax_50 = Axis(fig[1, 2], title="M=50", xgridvisible=false, ygridvisible=false,
        ylabel="Probabilidad")
    lines!(xrange, x -> cdf(G0, x), color=:red)
    for py_50 in py_samples_50
        ecdfplot!(py_50.locations, weights=py_50.weights, color=(:black, 0.2))
    end

    # M = 250
    ax_250 = Axis(fig[1, 3], title="M=250", xgridvisible=false, ygridvisible=false,
        ylabel="Probabilidad")
    lines!(xrange, x -> cdf(G0, x), color=:red)
    for py_250 in py_samples_250
        ecdfplot!(py_250.locations, weights=py_250.weights, color=(:black, 0.2))
    end

    fig
end

using CairoMakie
include("../helpers.jl")
include("../Dp.jl")

## Sampling
Random.seed!(619);
precisions = [1, 25, 250];
G0 = Gamma(6, 1);
dp_samples_1 = [Dp(1, G0) for _ in 1:10];
dp_samples_50 = [Dp(50, G0) for _ in 1:10];
dp_samples_250 = [Dp(250, G0) for _ in 1:10];

## Figure
begin
    CairoMakie.activate!()
    fig = Figure(resolution=(1440, 500))
    xrange = range(0, 15, length=100)

    # M = 1
    ax_1 = Axis(fig[1, 1], title="M=1", xgridvisible=false, ygridvisible=false,
        ylabel="Probabilidad")
    lines!(xrange, x -> cdf(G0, x), color=:red)
    for dp_1 in dp_samples_1
        ecdfplot!(dp_1.locations, weights=dp_1.weights, color=(:black, 0.2))
    end

    # M = 50
    ax_50 = Axis(fig[1, 2], title="M=50", xgridvisible=false, ygridvisible=false,
        ylabel="Probabilidad")
    lines!(xrange, x -> cdf(G0, x), color=:red)
    for dp_50 in dp_samples_50
        ecdfplot!(dp_50.locations, weights=dp_50.weights, color=(:black, 0.2))
    end

    # M = 250
    ax_250 = Axis(fig[1, 3], title="M=250", xgridvisible=false, ygridvisible=false,
        ylabel="Probabilidad")
    lines!(xrange, x -> cdf(G0, x), color=:red)
    for dp_250 in dp_samples_250
        ecdfplot!(dp_250.locations, weights=dp_250.weights, color=(:black, 0.2))
    end

    fig
end

# CairoMakie.save("monography/figures/DP - Stick Breaking eg.png", fig)

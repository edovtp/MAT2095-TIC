using CairoMakie

include("../helpers.jl");
include("../DpmData.jl");
include("../DpmMvNorm.jl");
include("../DpData.jl")


## A2 - Simulated data
Random.seed!(219);
n, a, A, w, W, α, β, ν, Ψ = 100, [0, 0], [1 0; 0 1], 4, 4, 2, 4, 5, [1 0; 0 1];
c1 = MvNormal([0, 0], [[0.5, 0.1] [0.1, 0.5]]);
c2 = MvNormal([-5, -2], [[0.6, 0] [0, 0.6]]);
c3 = MvNormal([0, -4], [[6, 0] [0, 0.1]]);
c4 = MvNormal([5, -5], [[1, -0.5] [-0.5, 1]]);

data_a2 = vcat(rand(c1, 25)', rand(c2, 25)', rand(c3, 25)', rand(c4, 25)');
c = repeat([1, 2, 3, 4], inner=25, outer=1);

# a, A, w, W, α, β, ν, Ψ
prior_par = ([0, 0], [[10, 0] [0, 10]], 1, 100, 2, 4, 5, [[1, 0] [0, 1]]);
y = [Vector(a) for a in eachrow(data_a2)];
N = 1000;
warmup=500;
a2_mvnorm = DpmMvNorm2(y, prior_par, N, "same", warmup);


begin
    CairoMakie.activate!()
    fig = Figure()
    ax1 = Axis(fig[1, 1]; xgridvisible=false, ygridvisible=false)
    sc = scatter!(data_a2[:, 1], data_a2[:, 2]; color=c)

    fig
end
#endregion
using CairoMakie
using KernelDensity
using MixFit
using FreqTables


include("../helpers.jl");
include("../DpmData.jl");
include("../DpmNorm.jl");

# A2 - Random hyperparameters, galaxies example
#region
velocities = [9172, 9558, 10406, 18419, 18927, 19330, 19440, 19541,
    19846, 19914, 19989, 20179, 20221, 20795, 20875, 21492,
    21921, 22209, 22314, 22746, 22914, 23263, 23542, 23711,
    24289, 24990, 26995, 34279, 9350, 9775, 16084, 18552,
    19052, 19343, 19473, 19547, 19856, 19918, 20166, 20196,
    20415, 20821, 20986, 21701, 21960, 22242, 22374, 22747,
    23206, 23484, 23666, 24129, 24366, 25633, 32065, 9483,
    10227, 16170, 18600, 19070, 19349, 19529, 19663, 19863,
    19973, 20175, 20215, 20629, 20846, 21137, 21814, 22185,
    22249, 22495, 22888, 23241, 23538, 23706, 24285, 24717,
    26960, 32789] ./ 1000;

prior_par_random = (0, 1000, 1, 100, 2, 4, 4, 2); # a, A, w, W, α, β, s, S
Random.seed!(219);
N = 10000;
warmup = 8000;
n = length(velocities);
@time a2_samples = DpmNorm2(velocities, prior_par_random, N, "same", warmup);

# Density estimation
function cond_dens(y)
    mean([pdf(Normal(θ[1], sqrt(θ[2])), y) for θ in a2_samples[:θ_new]])
end;
dpm_est = [cond_dens(y) for y in range(2, 40, length=500)];

# Inference about M
k = map(x -> length(unique(x)), eachrow(a2_samples[:μ]));
ϕ_π_k = zip(a2_samples[:ϕ], a2_samples[:π], k);
function cond_dens_M(alpha)
    aux = [p[2] * pdf(Gamma(2 + p[3], 1 / (4 - log(p[1]))), alpha) +
           (1 - p[2]) * pdf(Gamma(2 + p[3] - 1, 1 / (4 - log(p[1]))), alpha) for p in ϕ_π_k]
    mean(aux)
end
dens_M = [cond_dens_M(M) for M in range(0, 3; length=500)];

begin
    CairoMakie.activate!()
    fig = Figure(resolution=(1280, 720))
    xrange1 = range(2, 40; length=500)
    xrangeM = range(0, 3; length=500)

    ax1 = Axis(fig[1, 1]; xlabel="velocidad", xgridvisible=false, ygridvisible=false)
    hist!(velocities; normalization=:pdf, color=:steelblue1, bins=1:40)
    d_dpm = lines!(xrange1, dpm_est; color=:indianred, linewidth=4)

    ax2 = Axis(fig[1, 2]; xlabel="M", xgridvisible=false, ygridvisible=false)
    d_prior = lines!(xrangeM, x -> pdf(Gamma(2, 1 / 4), x); color=:royalblue4, linewidth=4, label="Priori", linestyle=:dash)
    d_post = lines!(xrangeM, dens_M; color=:coral2, linewidth=4, label="Posteriori")
    # leg = Legend(fig[1, 3], ax2)
    axislegend(ax2; labelsize=25)
    fig
end

# CairoMakie.save("monography/figures/DPM - galaxies example.png", fig)
#endregion

# Datos A1
#region
Random.seed!(219);
mmodel = MixtureModel(
    Normal[Normal(-5, 1), Normal(-1, 1), Normal(0, 1), Normal(5, 1)],
    [0.15, 0.25, 0.3, 0.3]
);

n = 50;
probs = [0.15, 0.25, 0.3, 0.3];
nmeans = [-5, -1, 0, 5];
components = sample(1:4, Weights(probs), n);
y = rand(Normal(0, 1), n) + nmeans[components];

prior_par_random = (0, 1000, 1, 100, 2, 4, 4, 2); # a, A, w, W, α, β, s, S
@time aux = DpmNorm2(y, prior_par_random, N, "same", warmup);

function cond_dens(y)
    mean([pdf(Normal(θ[1], sqrt(θ[2])), y) for θ in aux[:θ_new]])
end;
dpm_est = [cond_dens(y) for y in range(-8, 8, length=200)];

begin
    CairoMakie.activate!()
    fig = Figure()
    xrange = range(-8, 8; length=200)

    # Simulated data
    ax_2 = Axis(fig[1, 1]; title="(b)", xgridvisible=false, ygridvisible=false)
    h_real = hist!(y; normalization=:pdf, bins=30, color=(:blue, 0.15))
    d_dpm = lines!(xrange, dpm_est; color=:red, linewidth=3)
    fig
end

mean(map(maximum, eachrow(aux[:c])))
#endregion

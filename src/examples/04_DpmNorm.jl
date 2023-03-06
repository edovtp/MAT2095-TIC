using CairoMakie
using KernelDensity
using MixFit
using FreqTables


include("../helpers.jl");
include("../DpmData.jl");
include("../DpmNorm.jl");

# A1 - Density estimation
#region
## Sampling
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

M, m, γ, s, S = 1, 0, 100, 4, 2;
prior_par = (M, m, γ, s, S);
a1_normf = DpmNorm1f(y, prior_par, 10000, 8000);

function cond_dens(y)
    mean([pdf(Normal(θ[1], sqrt(θ[2])), y) for θ in a1_normf.θ_new])
end
dpm_est = [cond_dens(y) for y in range(-8, 8, 200)];
kde_est = kde(y; boundary=(-8, 8));
fmm_est = densfit(y; criterion=AIC);
fmm_est_model = MixtureModel(
    Normal[Normal(a[1], a[2]) for a in zip(fmm_est.μ, fmm_est.σ)],
    fmm_est.α
);

## Figure - Density estimation
begin
    CairoMakie.activate!()
    fig = Figure()
    xrange = range(-8, 8; length=200)

    # Simulated data
    ax_1 = Axis(fig[1, 1]; title="(a)", xgridvisible=false, ygridvisible=false)
    h_real = hist!(y; normalization=:pdf, bins=30, color=(:blue, 0.15))
    d_real = lines!(xrange, x -> pdf(mmodel, x), color=:black, linewidth=4)
    l_real = vlines!([-5, -1, 0, 5]; color=:red, linestyle=:dash, linewidth=3)

    ax_2 = Axis(fig[1, 2]; title="(b)", xgridvisible=false, ygridvisible=false)
    h_real = hist!(y; normalization=:pdf, bins=30, color=(:blue, 0.15))
    d_real = lines!(xrange, x -> pdf(mmodel, x); color=:black, linewidth=3)
    d_dpm = lines!(xrange, dpm_est; color=:red, linewidth=3)

    ax_3 = Axis(fig[2, 1]; title="(c)", xgridvisible=false, ygridvisible=false)
    h_real = hist!(y; normalization=:pdf, bins=30, color=(:blue, 0.15))
    d_real = lines!(xrange, x -> pdf(mmodel, x); color=:black, linewidth=3)
    d_kde = lines!(kde_est.x, kde_est.density; color=:green, linewidth=3)

    ax_4 = Axis(fig[2, 2]; title="(d)", xgridvisible=false, ygridvisible=false)
    h_real = hist!(y; normalization=:pdf, bins=30, color=(:blue, 0.15))
    d_real = lines!(xrange, x -> pdf(mmodel, x); color=:black, linewidth=3)
    d_fmm = lines!(xrange, x -> pdf(fmm_est_model, x); color=:blue, linewidth=3)

    Legend(fig[:, 3], [h_real, d_real, d_dpm, d_kde, d_fmm],
        ["Muestras", "Real", "DPM", "KDE", "FMM"])
    fig
end

# CairoMakie.save("monography/figures/DPM - density estimation.png", fig)

# Figure - Inference about k
k_sim = [length(unique(row)) for row in eachrow(a1_normf.μ_samples)];
freq_k = freqtable(k_sim);
k_sim_mean = mean(k_sim);

begin
    fig = Figure(resolution=(1280, 720))

    ax = Axis(fig[1, 1]; xticks=3:11, xgridvisible=false, ygridvisible=false)
    hist = barplot!(names(freq_k)[1], freq_k.array ./ 2000; bins=30,
        color=ifelse.(3:11 .== 4, :red, :blue))
    k_sim = vlines!(k_sim_mean; color=:black, linewidth=10)
    ylims!(0, 0.27)

    fig
end

CairoMakie.save("monography/figures/DPM - inference on k.png", fig)

# Figure - Sticky clusters
begin
    fig = Figure()

    ax = Axis(fig[1, 1]; xgridvisible=false, ygridvisible=false)
    lines = lines!(1751:2000, a1_normf.μ_samples[1751:2000, 3]; linewidth=2)
    ax.xlabel = "t"
    ax.ylabel = "μ"

    fig
end

CairoMakie.save("monography/figures/DPM - sticky clusters.png", fig)
#endregion

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

begin
    CairoMakie.activate!()
    fig = Figure()
    xrange = range(2, 40; length=500)

    ax = Axis(fig[1, 1]; xgridvisible=false, ygridvisible=false)
    hist!(velocities; normalization=:pdf, bins=1:40)
    d_dpm = lines!(xrange, dpm_est; color=:red, linewidth=3)

    fig
end

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
    fig = Figure()
    xrange = range(0, 3; length=500)

    ax = Axis(fig[1, 1]; xgridvisible=false, ygridvisible=false)
    d_prior = lines!(xrange, x -> pdf(Gamma(2, 1 / 4), x))
    d_post = lines!(xrange, dens_M; color=:red)
    fig
end

#endregion

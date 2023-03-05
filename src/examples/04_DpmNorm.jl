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
    
    ax = Axis(fig[1, 1]; xticks = 3:11, xgridvisible=false, ygridvisible=false)
    hist = barplot!(names(freq_k)[1], freq_k.array./2000; bins=30,
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

# A1 - Random hyperparameters, galaxies example
#region
#endregion

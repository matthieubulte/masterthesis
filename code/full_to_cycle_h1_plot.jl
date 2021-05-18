using Plots, StatsPlots, JLD, Distributions, HypothesisTests, Dates, ProgressMeter, LaTeXStrings

include("beta_product.jl")
include("plotutils.jl")

ps = [5; 10; 20; 50; 75; 90]; sims=250; repls=100; n=100; test_level = 0.05;

pvalues = zeros(Float32, size(ps)[1], repls);
pvalues_w = zeros(Float32, size(ps)[1], repls);

l = 1; for p = ps
    Cs = cyclebetaparams(p)
    beta_params = [((n - C - 1)/2, 1/2) for C = Cs]
    logbeta_sum_cdf = construct_cdf_log(beta_params, 100_000);

    results = load("./202105082246_output/results_$(p).jld")["data"];

    pvalues[l, :] = [
        mean(logbeta_sum_cdf.(-results[i,:]./n) .< test_level)
        for i = 1:repls
    ];

    pvalues_w[l, :] = [
        mean(cdf(Chisq(p*(p-3)/2), results[i,:]) .< test_level)
        for i = 1:repls
    ]

    l += 1
end

plot_hist(pvalues_w, ps, L"\textrm{Empirical power at level } \alpha = 0.05 \textrm{ based on } \chi^2_d", "power_complete_to_cycle_chisq"; ylabel=L"\textrm{Empirical power}")

plot_hist(pvalues, ps, L"\textrm{Empirical power at level } \alpha = 0.05 \textrm{ based on } Beta \textrm{-product}", "power_complete_to_cycle_beta"; ylabel=L"\textrm{Empirical power}")
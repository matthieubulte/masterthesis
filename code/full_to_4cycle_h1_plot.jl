

using Plots, StatsPlots, JLD, Distributions, HypothesisTests, Dates, ProgressMeter, LaTeXStrings

include("beta_product.jl")
include("plotutils.jl")

ps = [5; 10; 20; 50; 75; 90]; sims=250; repls=100; n=100; test_level = 0.05;

pvalues = zeros(Float32, size(ps)[1], repls);
pvalues_w = zeros(Float32, size(ps)[1], repls);

l = 1; for p = ps
    beta_params = ( (n - (p - 2) - 1)/2, 1/2 )
    beta_prod_cdf = construct_cdf([beta_params, beta_params], 100_000);

    results = load("./202105031741_output/results_$(p).jld")["data"];

    pvalues[l, :] = [
        mean(beta_prod_cdf.(results[i,:].^(2/n)) .< test_level)
        for i = 1:repls
    ];
    
    pvalues_w[l, :] = [
        mean(cdf(Chisq(2), -2*log.(results[i,:])) .< test_level)
        for i = 1:repls
    ]

    l += 1
end

plot_hist(pvalues_w, ps, L"\textrm{Empirical power at level } \alpha = 0.05 \textrm{ based on } \chi^2_d", "power_complete_to_chordless4cycle_chisq"; ylabel=L"\textrm{Empirical power}")

plot_hist(pvalues, ps, L"\textrm{Empirical power at level } \alpha = 0.05 \textrm{ based on } Beta \textrm{-product}", "power_complete_to_chordless4cycle_beta"; ylabel=L"\textrm{Empirical power}")
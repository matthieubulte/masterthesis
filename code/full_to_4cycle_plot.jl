using Plots, StatsPlots, JLD, Distributions, HypothesisTests, Dates, ProgressMeter, LaTeXStrings

include("beta_product.jl")
include("plotutils.jl")

ps = [5; 10; 20; 50; 75; 90]; sims=250; repls=100; n=100

pvalues = zeros(Float32, size(ps)[1], repls);
pvalues_w = zeros(Float32, size(ps)[1], repls);

l = 1; for p = ps
    beta_params = ( (n - (p - 2) - 1)/2, 1/2 )
    beta_prod_cdf = construct_cdf([beta_params, beta_params], 100_000);

    results = load("./202103112336_output/results_$(p).jld")["data"];

    pvalues[l, :] = [
        pvalue(ExactOneSampleKSTest(
            beta_prod_cdf.(results[i,:].^(2/n)),
            Uniform()))
        for i = 1:repls
    ];

    
    pvalues_w[l, :] = [
        pvalue(ExactOneSampleKSTest(
            -2*log.(results[i,:]),
            Chisq(2)))
        for i = 1:repls
    ]

    l += 1
end

plot_hist(pvalues_w, ps, L"\textrm{Distribution of p-values based on } \chi^2_d", "complete_to_chordless4cycle_chisq"; ylabel=L"\textrm{p-value}")

plot_hist(pvalues, ps, L"\textrm{Distribution of p-values based on } Beta \textrm{-product}", "complete_to_chordless4cycle_beta"; ylabel=L"\textrm{p-value}")
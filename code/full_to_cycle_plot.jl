

using Plots, StatsPlots, JLD, Distributions, HypothesisTests, Dates, ProgressMeter

include("beta_product.jl")
include("plotutils.jl")

ps = [5; 10; 20; 50; 75; 90]; sims=250; repls=100; n=100;

pvalues = zeros(Float32, size(ps)[1], repls);
pvalues_w = zeros(Float32, size(ps)[1], repls);

l = 1; for p = ps
    Cs = cyclebetaparams(p)
    beta_params = [((n - C - 1)/2, 1/2) for C = Cs]
    beta_prod_cdf = construct_cdf(beta_params, 100_000);

    results = load("./202105072217_output/results_$(p).jld")["data"];
    pvalues[l, :] = [
        pvalue(ExactOneSampleKSTest(
            beta_prod_cdf.(results[i,:].^(2/n)),
            Uniform()))
        for i = 1:repls
    ];
    pvalues_w[l, :] = [
        pvalue(ExactOneSampleKSTest(
            -2*log.(results[i,:]),
            Chisq(p*(p-3)/2)))
        for i = 1:repls
    ]
    l += 1
end

plot_hist(pvalues_w, ps, L"\textrm{Distribution of p-values based on } \chi^2_d", "complete_to_pcycle_chisq"; ylabel=L"\textrm{p-value}")

plot_hist(pvalues, ps, L"\textrm{Distribution of p-values based on } Beta \textrm{-product}", "complete_to_pcycle_beta"; ylabel=L"\textrm{p-value}")

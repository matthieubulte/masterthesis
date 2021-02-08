using Plots
using JLD
using Distributions
using HypothesisTests
using ProgressMeter

w_idx = 1
r_idx = 2
rstar_idx = 3

pvalues = zeros(Float32, 3, 7, 500)

@showprogress for u = 1:500
    stats = load("./output/stats_$(u).jld")["data"]
    for i in 1:7
        pvalues[w_idx, i, u] = pvalue(ExactOneSampleKSTest(stats[w_idx,i,:], Chisq(1)))
        pvalues[r_idx, i, u] = pvalue(ExactOneSampleKSTest(stats[r_idx,i,:], Normal()))
        pvalues[rstar_idx, i, u] = pvalue(ExactOneSampleKSTest(stats[rstar_idx,i,:], Normal()))
    end
end

function plots(pvals, idx, name)
    labels = ["$(α)" 
        for i = 1:7
        for α = round((i+2)/12, digits=2)]

    p = boxplot(pvalues[idx, 1, :], legend=false)
    for i = 2:7
        p = boxplot!(pvals[idx, i, :], legend=false)
    end
    savefig(p, "./plots/$(name)")
    title!("$(name) statistic KS p-value distribution")
    xticks!(1:7, labels)
    xlabel!("Scaling α (p = O(n^α))")
    p
end

plots(pvalues, rstar_idx, "rstar")
plots(pvalues, r_idx, "r")
plots(pvalues, w_idx, "w")
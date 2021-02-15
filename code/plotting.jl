using Plots
using StatsPlots
using JLD
using Distributions
using HypothesisTests
using Dates
using ProgressMeter

w_idx = 1
r_idx = 2
rstar_idx = 3

pvalues = zeros(Float32, 3, 5, 500);

@showprogress for u = 1:500
    stats = load("./output/stats_$(u).jld")["data"]
    for i in 1:7
        pvalues[w_idx, i, u] = pvalue(ExactOneSampleKSTest(stats[w_idx,i,:], Chisq(1)))
        pvalues[r_idx, i, u] = pvalue(ExactOneSampleKSTest(stats[r_idx,i,:], Normal()))
        pvalues[rstar_idx, i, u] = pvalue(ExactOneSampleKSTest(stats[rstar_idx,i,:], Normal()))
    end
end


@showprogress for u = 1:500
    stats = load("./output2/stats_$(u).jld")["data"]
    for i in 1:5
        pvalues[w_idx, i, u] = pvalue(ExactOneSampleKSTest(stats[w_idx,i,:], Chisq(1)))
        pvalues[r_idx, i, u] = pvalue(ExactOneSampleKSTest(stats[r_idx,i,:], Normal()))
        # pvalues[rstar_idx, i, u] = pvalue(ExactOneSampleKSTest(stats[rstar_idx,i,:], Normal()))
    end
end


function plots(pvals, idx, name)
    labels = ["$(d)" for d = [10; 50; 100; 250; 490]]
    
    p = boxplot(pvalues[idx, 1, :], legend=false)
    for i = 2:5
        p = boxplot!(pvals[idx, i, :], legend=false)
    end
    title!("$(name) statistic KS p-value distribution (n=500)")
    xticks!(1:5, labels)
    xlabel!("Dimension d")
    
    now = Dates.format(Dates.now(), "yyyymmddHHMM")
    savefig(p, "./plots/$(now)_$(name)")
    
    p
end

# plots(pvalues, rstar_idx, "rstar")
plots(pvalues, r_idx, "r")
plots(pvalues, w_idx, "w")
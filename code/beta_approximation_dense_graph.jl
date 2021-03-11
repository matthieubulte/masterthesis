using Plots
using StatsPlots
using JLD
using Distributions
using HypothesisTests
using Dates
using ProgressMeter

ps = [4, 8, 16, 32, 64, 96]

n = 100
sims = 100
pvalues = zeros(Float32, 2, size(ps)[1], sims);

χ_idx = 1
β_idx = 2

@showprogress for u = 1:sims
    stats = load("./202103032326_output/stats_$(u).jld")["data"]
    for i in 1:size(ps)[1]
        p = ps[i]
        f = p - 2
        
        
        wvals = stats[i,:]
        qvals = exp.(-wvals./2)
        q2nvals = qvals .^ (2/n)

        pvalues[χ_idx, i, u] = pvalue(ExactOneSampleKSTest(wvals, Chisq(1)))
        pvalues[β_idx, i, u] = pvalue(ExactOneSampleKSTest(q2nvals, Beta((n - f - 1)/2, 1/2)))
    end
end


function plots(pvals, idx, name, ps)
    labels = ["$(d)" for d = ps]
    
    p = boxplot(pvalues[idx, 1, :], legend=false)
    for i = 2:size(ps)[1]
        p = boxplot!(pvals[idx, i, :], legend=false)
    end
    title!("$(name) statistic KS p-value distribution (n=100)")
    xticks!(1:size(ps)[1], labels)
    xlabel!("Dimension d")
    
    now = Dates.format(Dates.now(), "yyyymmddHHMM")
    savefig(p, "./plots/$(now)_$(name)")
    
    p
end


plots(pvalues, χ_idx, "χ²", ps)
plots(pvalues, β_idx, "β", ps)

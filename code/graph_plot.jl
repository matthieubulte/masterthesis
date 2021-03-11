using Plots
using StatsPlots
using JLD
using Distributions
using HypothesisTests
using Dates
using ProgressMeter


results = load("./202102232228_output/results_$(5).jld")["data"];

n = 100;
q = LinRange(0, 0.99, 100);
f = 0;

qvals = quantile(results[:, 2], q);
q2nvals = qvals.^(2/n);
wvals = -2*log.(qvals);

qχ² = 1 .- cdf(Chisq(1), wvals);
qβ = cdf(Beta((n - f - 1)/2, 1/2), q2nvals);

plot(q,q,color="black",label=nothing,legend=:topleft)
plot!(q, qχ², label="χ²(1)")
plot!(q, qβ, label="Beta")

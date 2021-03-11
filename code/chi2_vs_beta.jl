using Plots
using StatsPlots
using Distributions

import Pluto
Pluto.run()

f = 1
qvals = LinRange(0, 0.99, 100);
q2nvals = qvals.^(2/n);
wvals = -2*log.(qvals);

qχ² = 1.0 .- cdf(Chisq(1), wvals);
qβ = cdf(Beta((n - f - 1)/2, 1/2), q2nvals);

plot(q,q,color="black",label=nothing,legend=:topleft)
plot!(q, qχ², label="χ²(1)")
plot!(q, qβ, label="Beta")

using Distributions
using LinearAlgebra
using PDMats
using Plots
using StatsPlots
using ProgressMeter

include("propscaling.jl")

# H0:
# 1 - 2 - 3 - 4
# 
# Cliques: {1,2} {2,3} {3,4}
c0 = [[1; 2], [2; 3], [3; 4]]

# H1:
#    2
#  /   \
# 1     3
#  \   /
#    4
#
# Cliques: {1,2} {1,4} {2,3} {3,4}
c1 = [[1; 2], [1; 4], [2; 3], [3; 4]]

# Here f(a) = |bd(1) ∩ bd(3)| = 2\0
f = 0

K_asym = [
    .5 .0 .0 .0;
    .1 .5 .0 .0;
    .0 .1 .5 .0;
    .0 .0 .1 .5
]
K = PDMat(K_asym + K_asym')
Σ = inv(K)

ns = [5, 10, 15];
results = zeros(size(ns)[1], sims);
sims = 10_000

ins = 1
for n = ns
    @showprogress for i = 1:sims
        X = rand(MvNormal(Σ), n)
        ssd = Symmetric(X * X')
        Σ̂ = ssd / n

        K̂ₚ = itpropscaling(c0, Σ̂)
        K̂ = itpropscaling(c1, Σ̂)

        w = n*(logdet(K̂) - logdet(K̂ₚ))
        results[ins, i] = exp(-w/2)
    end
    ins += 1
end

q = LinRange(0, 0.99, 100)
p = plot(layout=(1,3), size=(750,350))
ins = 1
for n = ns
    qvals = quantile(results[ins, :], q);
    q2nvals = qvals.^(2/n);
    wvals = -2*log.(qvals);

    qχ² = 1 .- cdf(Chisq(1), wvals);
    qβ = cdf(Beta((n - f - 1)/2, 1/2), q2nvals);

    
    plot!(subplot=ins,q,q,color="black",label=nothing,legend=:topleft)
    plot!(subplot=ins,q, qχ², label="χ²(1)")
    plot!(subplot=ins,q, qβ, label="Beta")
    xlabel!(subplot=ins,"empirical prob.",fontsize=1)
    title!(subplot=ins,"sample size = $(n)")
    
    ins += 1
end
ylabel!(subplot=1,"approx. prob.")
savefig(p, "output3/square_vs_line")
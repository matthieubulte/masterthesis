using Distributions
using LinearAlgebra
using PDMats
using ProgressMeter
using Dates
using JLD

include("propscaling.jl")
include("random_graphs.jl")


using Plots
using StatsPlots


# function log(s)
#     now = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM")
#     println("$(now): $(s)")
# end

function experiment(K₀, C₀, C₁, n)
    X = rand(MvNormal(inv(K₀)), n)
    ssd = Symmetric(X * X')
    Σ̂ = ssd / n

    K̂₀ = itpropscaling(C₀, Σ̂)
    K̂₁ = itpropscaling(C₁, Σ̂)

    w = n*(logdet(K̂₁) - logdet(K̂₀))
    exp(-w/2)
end

function run()
    sims = 500
    n = 100
    pvals = [5; 10; 20; 50; 75; 90; 99]
    log("Parameters: sims=$(sims), n=$(n), p=$(pvals)")

    now = Dates.format(Dates.now(), "yyyymmddHHMM")
    outDir = "./$(now)_output"
    mkdir(outDir)
    log("Created $(outDir)")

    log("Starting...")
    for p = pvals
        results = zeros(Float64, sims, 2)

        @showprogress "Simulation level: " for i = 1:sims
            chainCliques, chainK = chain(p)
            cycleCliques, cycleK = cycle(p)
            diamondCliques, _ = cycleWithOneChord(p)

            results[i, 1] = experiment(cycleK, cycleCliques, diamondCliques, n)
            results[i, 2] = experiment(chainK, chainCliques, cycleCliques, n)
        end

        save("$(outDir)/results_$(p).jld", "data", results)
    end

    log("Done")
end

run()


sims = 500
f = 0
results = zeros(Float64, sims, 2);
q = LinRange(0, 1, 100);

p = 100
n = 10

@showprogress "Simulation level: " for i = 1:sims
    chainCliques, chainK = chain(p)
    cycleCliques, cycleK = cycle(p)
    diamondCliques, _ = cycleWithOneChordAndNoTriade(p)

    results[i, 1] = experiment(cycleK, cycleCliques, diamondCliques, n)
    results[i, 2] = experiment(chainK, chainCliques, cycleCliques, n)
end

qvals = quantile(results[:, 2], q); q2nvals = qvals.^(2/n); wvals = -2*log.(qvals);

qχ² = 1.0 .- cdf(Chisq(1), wvals); qβ = cdf(Beta((n - f - 1)/2, 1/2), q2nvals);

plot(q,q,color="black",label=nothing,legend=:topleft); plot!(q, qχ², label="χ²(1)"); plot!(q, qβ, label="Beta")

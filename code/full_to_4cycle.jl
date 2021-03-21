using Distributions
using LinearAlgebra
using PDMats
using ProgressMeter
using Dates
using JLD

include("propscaling.jl")
include("parameters_wrangling.jl")
include("beta_product.jl")

function denseWith4Cycle(p)
    @assert p > 4

    C = Cholesky(W(p), :L, 0)
    removeEdge!(C, 1, 3)
    removeEdge!(C, 2, 4)

    cycle_edges = [ [1,2], [1,4], [2,3], [3,4] ]
    dense_nodes = 5:p
    
    cliques = [ vcat(e, dense_nodes) for e = cycle_edges ]
    
    cliques, PDMat(C)
end

function experiment(K₀, C₀, C₁, n)
    X = rand(MvNormal(inv(K₀)), n)
    ssd = Symmetric(X * X')
    Σ̂ = ssd / n

    K̂₀ = itpropscaling(C₀, Σ̂)
    K̂₁ = itpropscaling(C₁, Σ̂)

    w = n*(logdet(K̂₁) - logdet(K̂₀))
    exp(-w/2)
end


function _log(s)
    now = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM")
    println("$(now): $(s)")
end

function run()
    repls = 100
    sims = 250
    n = 100
    pvals = [5; 10; 20; 50; 75; 90]
    _log("Parameters: repls=$(repls) sims=$(sims), n=$(n), p=$(pvals)")

    now = Dates.format(Dates.now(), "yyyymmddHHMM")
    outDir = "./$(now)_output"
    mkdir(outDir)
    _log("Created $(outDir)")

    open("$(outDir)/config.txt", "w") do io
        write(io, "repls=$(repls) sims=$(sims), n=$(n), p=$(pvals)")
    end

    _log("Starting...")
    for p = pvals
        results = zeros(Float64, repls, sims)

        @showprogress "Replication level: " for j = 1:repls
            for i = 1:sims
                denseWith4CycleCliques, denseWith4CycleK = denseWith4Cycle(p)
                denseCliques = [collect(1:p)]
                results[j, i] = experiment(denseWith4CycleK, denseWith4CycleCliques, denseCliques, n)
            end
        end

        save("$(outDir)/results_$(p).jld", "data", results)
    end

    _log("Done")
end

run()

# sims = 500
# q = LinRange(0, 1, 100);
# results = zeros(Float64, sims);


# p = 90
# n = 100

# beta_params = ( (n - (p - 2) - 1)/2, 1/2 )

# beta_prod_cdf = construct_cdf([beta_params, beta_params], 10_000);

# @showprogress "Simulation level: " for i = 1:sims
#     denseWith4CycleCliques, denseWith4CycleK = denseWith4Cycle(p)
#     denseCliques = [collect(1:p)]

#     results[i] = experiment(denseWith4CycleK, denseWith4CycleCliques, denseCliques, n)
# end

# qvals = quantile(results, q); q2nvals = qvals.^(2/n); wvals = -2*log.(qvals);

# qχ² = 1.0 .- cdf(Chisq(1), wvals); qβ = beta_prod_cdf.(q2nvals);

# plot(q,q,color="black",label=nothing,legend=:topleft); plot!(q, qχ², label="χ²"); plot!(q, qβ, label="Beta product")

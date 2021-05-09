using Distributions
using LinearAlgebra
using PDMats
using ProgressMeter
using Dates
using JLD

include("propscaling.jl")
include("parameters_wrangling.jl")
include("beta_product.jl")
include("log.jl")

function cyclecliques(p)
    @assert p > 4
    cliques = [ [i, i+1] for i = 1:(p-1) ]
    push!(cliques, [1, p])
    cliques
end

function experiment(K₀, C₀, C₁, n)
    X = rand(MvNormal(inv(K₀)), n)
    ssd = Symmetric(X * X')
    Σ̂ = ssd / n

    K̂₀ = itpropscaling(C₀, Σ̂)
    K̂₁ = itpropscaling(C₁, Σ̂)

    n*(logdet(K̂₁) - logdet(K̂₀))
end

function run()
    repls = 100
    sims = 250
    n = 100
    pvals = [5; 10; 20; 50; 75; 90]

    now = Dates.format(Dates.now(), "yyyymmddHHMM")
    outDir = "./$(now)_output"
    mkdir(outDir)
    io = open("$(outDir)/logs.txt", "w")

    _log(io, "Created $(outDir)")
    _log(io, "Parameters: Cycle H1 repls=$(repls) sims=$(sims), n=$(n), p=$(pvals)")

    _log(io, "Starting...")
    for p = pvals
        fails = 0
        results = zeros(Float64, repls, sims)

        @showprogress "Replication level: " for j = 1:repls
            for i = 1:sims
                cycleCliques, cycleK = cyclecliques(p), Cholesky(W(p), :L, 0)
                denseCliques = [collect(1:p)]
                try
                    results[j, i] = experiment(cycleK, cycleCliques, denseCliques, n)
                catch
                    fails += 1
                    _log(io, "$p: $fails / $((j-1)*repls + i)")
                end
                
            end
        end

        save("$(outDir)/results_$(p).jld", "data", results)
    end

    _log(io, "Done")
    close(io)
end

run()
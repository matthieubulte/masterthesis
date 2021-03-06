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

function cycle(p)
    @assert p > 4

    C = Cholesky(Wc(p), :L, 0)
    removeEdge!(C, 2, p)
    
    cliques = [ [i, i+1] for i = 1:(p-1) ]
    push!(cliques, [1, p])
    
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
    _log(io, "Parameters: Cycle H0 repls=$(repls) sims=$(sims), n=$(n), p=$(pvals)")

    
    _log(io, "Starting...")
    for p = pvals
        results = zeros(Float64, repls, sims)

        @showprogress "Replication level: " for j = 1:repls
            for i = 1:sims
                cycleCliques, cycleK = cycle(p)
                denseCliques = [collect(1:p)]
                results[j, i] = experiment(cycleK, cycleCliques, denseCliques, n)
            end
        end

        save("$(outDir)/results_$(p).jld", "data", results)
    end

    _log(io, "Done")
    close(io)
end

run()

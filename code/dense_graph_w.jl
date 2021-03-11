using Distributions
using LinearAlgebra
using ProgressMeter
using JLD
using Dates
using PDMats

include("matrix_extra.jl")
include("parameters_wrangling.jl")


function run(n, ps, sims)
    now = Dates.format(Dates.now(), "yyyymmddHHMM")
    outDir = "./$(now)_output"
    println("Creating $(outDir)...")
    mkdir(outDir)
    
    open("$(outDir)/config.txt", "w") do io
        write(io, "n=$(n), p=$(ps), sims=$(sims)")
    end
    
    a = 1; b = 2;

    println("Starting...")
    @showprogress "Top-level" for u = 1:sims
        stats = zeros(Float64, size(ps)[1], sims);
        i = 0

        for d = ps
            compK̂ = mkRestMLE(d, a, b);
            i += 1
            for j = 1:sims
                K, Σ = sampleKΣ(d, a, b);
                X = rand(MvNormal(Σ), n);
                XX = Symmetric(X*transpose(X));
            
                Σ̂ = XX/n; # unconstrained MLE
                K̂ₚ = Symmetric(compK̂(Σ̂)); # profile MLE 

                stats[i, j] = -n * (logabsdet(Σ̂)[1] + logabsdet(K̂ₚ)[1])
            end
        end
        save("$(outDir)/stats_$(u).jld", "data", stats)
    end
end

run(100, [4; 8; 16; 32; 64; 96], 100)
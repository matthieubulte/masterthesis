using Distributions, LinearAlgebra, PDMats, ProgressMeter, Dates,JLD

include("propscaling.jl")
include("parameters_wrangling.jl")
include("beta_product.jl")

function denseWith4CycleCliques(p)
    @assert p > 4
    cycle_edges = [ [1,2], [1,4], [2,3], [3,4] ]
    dense_nodes = 5:p
    [ vcat(e, dense_nodes) for e = cycle_edges ]
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
    _log("Parameters: hypothesis=h1 repls=$(repls) sims=$(sims), n=$(n), p=$(pvals)")

    now = Dates.format(Dates.now(), "yyyymmddHHMM")
    outDir = "./$(now)_output"
    mkdir(outDir)
    _log("Created $(outDir)")

    open("$(outDir)/config.txt", "w") do io
        write(io, "repls=$(repls) sims=$(sims), n=$(n), p=$(pvals)")
    end

    

    _log("Starting...")
    for p = pvals
        fails = 0
        results = zeros(Float64, repls, sims)

        @showprogress "Replication level: " for j = 1:repls
            for i = 1:sims
                cliques, K = denseWith4CycleCliques(p), Cholesky(W(p), :L, 0)
                denseCliques = [collect(1:p)]
                try
                    results[j, i] = experiment(K, cliques, denseCliques, n)
                catch
                    println(p, " ", fails)
                    fails += 1
                end
            end
        end

        save("$(outDir)/results_$(p).jld", "data", results)
    end

    _log("Done")
end

run()
 
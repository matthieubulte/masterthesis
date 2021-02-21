using Distributions
using LinearAlgebra
using Plots
using ProgressMeter
using JLD
using PDMats
using BenchmarkTools
using Dates
using DiffResults

include("matrix_extra.jl")
include("parameters_wrangling.jl")
include("derivatives.jl")

function ℓ(ψ, λ, ψ̂, λ̂, n)
    K = thetaToK(ψ, λ)
    K̂ = thetaToK(ψ̂, λ̂)
    return ℓ(K, K̂, n)
end

function ℓ(K, K̂, n)
    n/2*[logabsdet(K)[1] - tr(K̂\K)]
end

n = 500; d = 100; a = 1; b = 2; compK̂ = mkRestMLE(d, a, b);
K, Σ = sampleKΣ(d, a, b); X = rand(MvNormal(Σ), n); XX = Symmetric(X*transpose(X));
Σ̂ = XX/n; K̂ = inv(Σ̂); ψ̂, λ̂ = KToTheta(K̂); K̂ₚ = Symmetric(compK̂(Σ̂)); ψₚ, λₚ = KToTheta(K̂ₚ);

seedx = rand(eltype(x), size(x))
tp = ReverseDiff.HessianTape(f, seedx)

lsize = size(λ̂)[1]
ℓ_t = (ψₚ, λₚ, ψ̂, λ̂) -> ℓ(ψₚ[1], λₚ, ψ̂[1], λ̂, n)
ℓ_th = (x) -> ℓ_t(x[1], x[2:2+lsize-1], x[2+lsize], x[2+lsize+1:end])

seedx = rand(Float64, 2*lsize + 2);
ℓ_tape = ReverseDiff.HessianTape(ℓ_th, seedx)
ℓ_compiled_tape = ReverseDiff.compile(ℓ_tape)

results = DiffResults.HessianResult(seedx);

@time ReverseDiff.hessian!(results, ℓ_compiled_tape, vcat(ψₚ, λₚ, ψ̂, λ̂))


ℓ_partial2 = (ψ, λ) -> ℓ(ψ[1], λ, ψ̂, λ̂, n)
ℓ_tape2 = ReverseDiff.GradientTape(ℓ_partial2, (rand(1), rand(lsize)))
ℓ_compiled_tape2 = ReverseDiff.compile(ℓ_tape2)

results = (similar([ψ̂], 1), similar(λ̂, lsize))
@benchmark ReverseDiff.gradient!(results, ℓ_compiled_tape2, ([ψₚ], λₚ))




# _dθ̂ = dθ̂(ℓ)
# @benchmark _dθ̂(ψₚ, λₚ, ψ̂, λ̂, n)

# _dθ̂2 = dθ̂2(ℓ)
# @benchmark _dθ̂2(ψₚ, λₚ, ψ̂, λ̂, n)








# function u(ψₚ, λₚ, ψ̂, λ̂, n)
#     _dθ̂ = dθ̂(ℓ)
#     hess = dθ(dθ(ℓ))

#     ddθ̂ = _dθ̂(ψ̂, λ̂, ψ̂, λ̂, n) - _dθ̂(ψₚ, λₚ, ψ̂, λ̂, n)
#     dλθ̂ = transpose(dλ(_dθ̂)(ψₚ, λₚ, ψ̂, λ̂, n))
#     ltop = det([ddθ̂ dλθ̂])

#     dθθ̂ = dθ(_dθ̂)(ψ̂, λ̂, ψ̂, λ̂, n)
#     l = ltop / det(dθθ̂)
    
#     ĵ = -hess(ψ̂, λ̂, ψ̂, λ̂, n)
#     j̃λλ = -hess(ψₚ, λₚ, ψ̂, λ̂, n)[2:end,2:end]
#     r = sqrt(det(ĵ) / det(j̃λλ))

#     l * r
# end

# function w(ψₚ, λₚ, ψ̂, λ̂, n)
#     2 * (ℓ(ψ̂, λ̂, ψ̂, λ̂, n) - ℓ(ψₚ, λₚ, ψ̂, λ̂, n))[1]
# end

# function r(ψₚ, λₚ, ψ̂, λ̂, n)
#     sign(ψ̂ - ψₚ) * sqrt(w(ψₚ, λₚ, ψ̂, λ̂, n))
# end

# function rstar(ψₚ, λₚ, ψ̂, λ̂, n)
#     _r = r(ψₚ, λₚ, ψ̂, λ̂, n) 
#     _u = u(ψₚ, λₚ, ψ̂, λ̂, n) 
#     _r + log(_u / _r) / _r
# end

# function compStats(ψₚ, λₚ, ψ̂, λ̂, n)
#     _w = max(w(ψₚ, λₚ, ψ̂, λ̂, n), 1e-5)

#     _r = sign(ψ̂ - ψₚ) * sqrt(_w)

#     _u = u(ψₚ, λₚ, ψ̂, λ̂, n) 
    
#     _w, _r, _r + log(_u / _r) / _r
# end

# # NEW 
# # BenchmarkTools.Trial: 
# #   memory estimate:  262.45 MiB
# #   allocs estimate:  5676
# #   --------------
# #   minimum time:     223.764 ms (8.20% GC)
# #   median time:      250.205 ms (11.05% GC)
# #   mean time:        252.565 ms (12.21% GC)
# #   maximum time:     282.641 ms (20.16% GC)
# #   --------------
# #   samples:          20
# #   evals/sample:     1

# # OLD
# # BenchmarkTools.Trial: 
# #   memory estimate:  269.01 MiB
# #   allocs estimate:  5679
# #   --------------
# #   minimum time:     310.596 ms (11.76% GC)
# #   median time:      1.005 s (20.38% GC)
# #   mean time:        1.378 s (18.59% GC)
# #   maximum time:     5.767 s (15.74% GC)
# #   --------------
# #   samples:          45
# #   evals/sample:     1


# # @benchmark begin
# #     @time K, Σ = sampleKΣnew(d, a, b);
# #     @time X = rand(MvNormal(Σ), n);
# #     @time XX = Symmetric(X*transpose(X));

# #     @time Σ̂ = XX/n;
    
# #     @time K̂ = inv(Σ̂).data; # unconstrained MLE
# #     @time K̂ₚ = compK̂(Σ̂); # profile MLE 

# #     ψₚ, λₚ = KToTheta(K̂ₚ);
# #     ψ̂, λ̂ = KToTheta(K̂);
# # end



# # TODO: 
# #   - compute the MLE in a more generic way to handle phi of higher dim 
# #   - compute rstar by hand to hopefully improve performance
# #

# function run()
#     now = Dates.format(Dates.now(), "yyyymmddHHMM")
#     outDir = "./$(now)_output"
#     println("Creating $(outDir)...")
#     mkdir(outDir)
    
#     w_idx = 1
#     r_idx = 2
#     rstar_idx = 3
    
#     n = 500;
#     sims = 500;
    
#     a = 1; b = 2;
#     dvals = [10; 50; 100; 250; 490]

#     println("Starting...")
#     @showprogress "Top-level" for u = 13:500
#         stats = zeros(Float64, 3, size(dvals)[1], sims);
#         i = 0
#         num_negs = 0

#         for d = dvals
#             compK̂ = mkRestMLE(d, a, b);
#             i += 1

#             println("\nTop-level = $(u), Dimension = $(d)")

#             @showprogress "Simulation-level" for j = 1:sims
#                 K, Σ = sampleKΣ(d, a, b);
#                 X = rand(MvNormal(Σ), n);
#                 XX = Symmetric(X*transpose(X));
            
#                 Σ̂ = XX/n;
#                 K̂ = inv(Σ̂); ψ̂, λ̂ = KToTheta(K̂); # unconstrained MLE
#                 K̂ₚ = Symmetric(compK̂(Σ̂)); ψₚ, λₚ = KToTheta(K̂ₚ); # profile MLE 

#                 sk = Σ̂*K̂ₚ
#                 _w = n * (-logabsdet(sk)[1] + tr(sk) - d)

#                 stats[w_idx, i, j] = _w

#                 if _w >= 0
#                     _r = sign(ψ̂ - ψₚ) * sqrt(_w)
#                     stats[r_idx, i, j] = _r
#                 else
#                     num_negs += 1
#                 end
#                 # _w, _r, _rstar = compStats(ψₚ, λₚ, ψ̂, λ̂, n)
#                 # stats[r_idx, αi-2, j] = _r
#                 # stats[rstar_idx, αi-2, j] = _rstar
#                 # stats[w_idx, αi-2, j] = _w
#             end

#             if num_negs > 0
#                 println("u=$(u), d=$(d), num_negs=$(num_negs)")
#             end
#         end
#         save("$(outDir)/stats_$(u).jld", "data", stats)
#     end
# ends
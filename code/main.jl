using Distributions
using LinearAlgebra
# using Plots
using ProgressMeter
using JLD

include("matrix_extra.jl")
include("parameters_wrangling.jl")
include("derivatives.jl")

function T(K, c, Σ̂)
    nK = copy(K)
    p = size(K)[1]
    notc = compl(1:p, c)
    
    nK[c, c] = inv(Σ̂[c,c]) + K[c,notc] * (K[notc,notc]\K[notc,c])
    return nK
end

function ℓ(ψ, λ, ψ̂, λ̂, n)
    K = thetaToK(ψ, λ)
    K̂ = thetaToK(ψ̂, λ̂)
    return ℓ(K, K̂, n)
end

function ℓ(K, K̂, n)
    p = size(K)[1]
    return [log(det(K))*n/2 - n*tr(K̂\K)/2]
end


function u(ψₚ, λₚ, ψ̂, λ̂, n)
    _dθ̂ = dθ̂(ℓ)
    hess = dθ(dθ(ℓ))

    ddθ̂ = _dθ̂(ψ̂, λ̂, ψ̂, λ̂, n) - _dθ̂(ψₚ, λₚ, ψ̂, λ̂, n)
    dλθ̂ = transpose(dλ(_dθ̂)(ψₚ, λₚ, ψ̂, λ̂, n))
    ltop = det([ddθ̂ dλθ̂])

    dθθ̂ = dθ(_dθ̂)(ψ̂, λ̂, ψ̂, λ̂, n)
    l = ltop / det(dθθ̂)
    
    ĵ = -hess(ψ̂, λ̂, ψ̂, λ̂, n)
    j̃λλ = -hess(ψₚ, λₚ, ψ̂, λ̂, n)[2:end,2:end]
    r = sqrt(det(ĵ) / det(j̃λλ))

    l * r
end

function w(ψₚ, λₚ, ψ̂, λ̂, n)
    2 * (ℓ(ψ̂, λ̂, ψ̂, λ̂, n) - ℓ(ψₚ, λₚ, ψ̂, λ̂, n))[1]
end

function r(ψₚ, λₚ, ψ̂, λ̂, n)
    sign(ψ̂ - ψₚ) * sqrt(w(ψₚ, λₚ, ψ̂, λ̂, n))
end

function rstar(ψₚ, λₚ, ψ̂, λ̂, n)
    _r = r(ψₚ, λₚ, ψ̂, λ̂, n) 
    _u = u(ψₚ, λₚ, ψ̂, λ̂, n) 
    _r + log(_u / _r) / _r
end

function compStats(ψₚ, λₚ, ψ̂, λ̂, n)
    _w = max(w(ψₚ, λₚ, ψ̂, λ̂, n), 1e-5)

    _r = sign(ψ̂ - ψₚ) * sqrt(_w)

    _u = u(ψₚ, λₚ, ψ̂, λ̂, n) 
    
    _w, _r, _r + log(_u / _r) / _r
end

function mkRestMLE(p, a, b)
    Γ = collect(3:p)
    V = collect(1:p)
    Ca = vcat([a], Γ)
    Cb = vcat([b], Γ)

    return (Σ̂) -> T(T(eye(p), Ca, Σ̂), Cb, Σ̂)
end

n = 1_000
sims = 1_000
alphas = 3:9

a = 1
b = 2

w_idx = 1
r_idx = 2
rstar_idx = 3

println("Starting...")

@showprogress for u = 1:1000
    stats = zeros(Float64, 3, size(alphas)[1], sims);

    for αi = alphas
        # NOTE: using p=n^α here is wrong because in the model we are looking at, p is the dimensionality
        # of the data while the dimensionality of the parameter space is then p^2 = n^2α. To be in the framework
        # p = O(n^α) we would then need to have p = sqrt(n^α) 
        α = αi/12
        d = Int(floor(n^(α/2)))
        compK̂ = mkRestMLE(d, a, b);

        for j = 1:sims
            K, Σ = sampleKΣ(d, a, b);
            X = rand(MvNormal(Σ), n);
            XX = Symmetric(X*transpose(X));
        
            Σ̂ = XX/n;
            K̂ = inv(Σ̂).data; # unconstrained MLE
            K̂ₚ = compK̂(Σ̂); # profile MLE 
        
            ψₚ, λₚ = KToTheta(K̂ₚ);
            ψ̂, λ̂ = KToTheta(K̂);
        
            _w, _r, _rstar = compStats(ψₚ, λₚ, ψ̂, λ̂, n)

            stats[r_idx, αi-2, j] = _r
            stats[rstar_idx, αi-2, j] = _rstar
            stats[w_idx, αi-2, j] = _w
        end
    end
    save("./output/stats_$(u).jld", "data", stats)
end

# TODO: use reverse diff
#

# histogram(
#     cdf(Normal(), stats[r_idx, 2, :]), 
#     bins=20
# )

# p,runtime(s)
# 4,5
# 5,11
# 6,34
# 7,45
# 8,61
# 10,184
# 12,540

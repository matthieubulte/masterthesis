using LinearAlgebra
include("matrix_extra.jl")

function Tₖ(K, c, Σ̂)
    nK = copy(K)
    p = size(K)[1]
    notc = [i for i = 1:p if !(i in c)]
    
    nK[c, c] = inv(Σ̂[c,c]) + K[c,notc] * (K[notc,notc]\K[notc,c])
    return nK
end

function T(C, Σ̂; K₀ = eye(Σ̂))
    K̂ = K₀
    for c = C
        K̂ = Tₖ(K̂, c, Σ̂)
    end
    K̂
end

function itpropscaling(C, Σ̂; tol=1e-8)
    K̂ = eye(Σ̂)
    Ko = K̂ .+ (tol + 1)
    while norm(K̂ - Ko) > tol
        K̂, Ko = T(C, Σ̂; K₀=K̂), K̂
    end
    Symmetric(K̂)
end

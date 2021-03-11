using LinearAlgebra
include("matrix_extra.jl")

function T!(C, Σ̂, K̂)
    p = size(K̂)[1]
    for c = C
        notc = [i for i = 1:p if !(i in c)]
        @inbounds K̂[c, c] =  inv(Σ̂[c,c]) + K̂[c,notc] * (K̂[notc,notc]\K̂[notc,c])
    end
    K̂
end

function itpropscaling(C, Σ̂; tol=1e-8)
    K̂ = eye(Σ̂)
    Ko = K̂ .+ (tol + 1)
    while norm(K̂ - Ko) > tol
        Ko .= K̂
        T!(C, Σ̂, K̂)
    end
    Symmetric(K̂)
end

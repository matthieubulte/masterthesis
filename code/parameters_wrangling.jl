using Distributions

include("matrix_extra.jl")

function sampleKΣ(p, a, b)
    L = W(p);
    C = Cholesky(L, :L, 0);
    removeEdge!(C, a, b)
    K = PDMat(C)
    Σ = inv(K)
    K, inv(K)
end

function removeEdge!(C, a, b)
    e = zeros(Float64, size(C)[1]);
    @inbounds e[a] = - C.L[a, :]' * C.L[b, :]
    @inbounds e[b] = 1
    lowrankupdate!(C, e)
end


function remedge(C, a, b)
    L = Matrix(C.L)
    La = L[a, :]
    Lb = L[b, :]
    L[a, :] -= dot(La, Lb) * Lb / dot(Lb, Lb)
    L
end


function W(p)
    L = zeros(Float64, p, p)
    for i = 1:p
        @inbounds L[i,i] = rand(Chi(p - i + 1.0))
    end
    for j in 1:p-1, i in j+1:p
        @inbounds L[i,j] = randn()
    end
    L
end

function Wc(p)
    C = Cholesky(W(p), :L, 0)
    for i = 1:p-1
        for j = i+2 : p
            if i == 1 && j == p
                continue
            end
            removeEdge!(C, i, j)
        end
    end
    C
end

function compl(V, c)
    return [i for i = V if !(i in c)]
end

function thetaToK(ψ, λ)
    vec = [λ[1]; ψ; λ[2:end]]
    return vecToSymm(vec)
end

function KToTheta(K)
    vec = symmToVec(K)
    λ = [vec[1]; vec[3:end]]    
    ψ = vec[2]
    return ψ, λ
end

function T(K, c, Σ̂)
    nK = copy(K)
    p = size(K)[1]
    notc = compl(1:p, c)
    
    nK[c, c] = inv(Σ̂[c,c]) + K[c,notc] * (K[notc,notc]\K[notc,c])
    return nK
end

function mkRestMLE(p, a, b)
    Γ = collect(3:p)
    V = collect(1:p)
    Ca = vcat([a], Γ)
    Cb = vcat([b], Γ)

    return (Σ̂) -> T(T(eye(p), Ca, Σ̂), Cb, Σ̂)
end


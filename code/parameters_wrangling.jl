using Distributions

function sampleKΣ(p, a, b)
    e = zeros(Float64, p);
    L = W(p);
    C = Cholesky(L, :L, 0);
    @inbounds e[a] = - L[a, :]' * L[b, :]
    @inbounds e[b] = 1
    lowrankupdate!(C, e)
    K = PDMat(C)
    Σ = inv(K)
    K, inv(K)
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

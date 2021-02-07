using Distributions

function sampleKΣ(p, a, b)
    while true
        K = rand(Wishart(p, eye(p)))
        K[a,b] = 0
        K[b,a] = 0
        Σ = Symmetric(inv(K))
    
        if isposdef(Σ)
            return K, Σ
        end
    end
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

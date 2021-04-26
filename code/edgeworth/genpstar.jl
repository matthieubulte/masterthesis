include("./vendor_extra.jl")
include("./cgflib.jl")

function pstar(ℓ, θ₀, n)
    function density(θ̂)
        j = ∇²(θ -> ℓ(θ, θ̂))
        sqrt(n) * sqrt(abs(j(θ̂)) / n) * exp(ℓ(θ₀, θ̂) - ℓ(θ̂, θ̂)) / sqrt(2pi)
    end
end
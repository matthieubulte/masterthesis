using Combinatorics
using Symbolics: map_subscripts
include("vendor_extra.jl")

function cumulants(cgf, order; T=Number)
    t = Taylor1(T, order)
    scaled_cumulants = cgf(t).coeffs
    scaling = exp(t).coeffs
    return (scaled_cumulants ./ scaling)[2:end]
end

iidsum(cgf, n) = (t) -> n * cgf(t)
affine(cgf, loc, scale) = (t) -> cgf(t * scale) + t * loc * scale
tilted(cgf, γ) = (t) -> cgf(t + γ) - cgf(γ)

function irwinhall(n)
    function (x)
        d = 0
        m1 = 1
        
        for k = 0:n
            d += m1 * binomial(n, k) * (x-k)^(n-1) * sign(x - k)
            m1 *= -1
        end

        d / factorial(n - 1) / 2
    end
end

daffine(d, loc, scale) = (x) -> d((x-loc)*scale)*scale
dscale(d, scale) = daffine(d, 0, scale)

_pdf(d) = (x) -> pdf(d, x)

gamma(α, θ) = (t) -> -α*log(θ)-α*log(1/θ-t)
_uniform = (t) -> t == 0 ? 1 : log((exp(t/2) - exp(-t/2))/t)
uniform(a, b) = affine(_uniform, 0.5, b - a)

symcgf(order::Int) = osymcgf(symbols.(string.("κ", map_subscripts.(1:order))))

osymcgf(cums::Vector) = let
    order = length(cums)
    function(t)
        o = zero(t); f = one(t);
        for i=1:order
            f *= i
            o += cums[i] * t^i / f
        end
        o
    end
end



symmv(dim, order) = let 
    function(t; T=eltype(t))
        o = zero(T);
        for i=1:order
            for s=combinations(1:dim, i)
                oo = one(T)
                for j=1:i
                    oo *= symbols("κ_$(map_subscripts(s[j]))", real=true) * t[s[j]]
                end
                o += oo
            end
        end
        o
    end
end
include("vendor_extra.jl")

function cumulants(cgf, order; T=Number)
    t = Taylor1(T, order)
    scaled_cumulants = cgf(t).coeffs
    scaling = exp(t).coeffs
    return (scaled_cumulants ./ scaling)[2:end]
end

iidsum(cgf, n) = (t) -> n * cgf(t)
affine(cgf, loc, scale) = (t) -> cgf(t * scale) + t * loc * scale

gamma(α, θ) = (t) -> -α*log(θ)-α*log(1/θ-t)
sym(order) = let 
    syms = symbols(join(["κ$i" for i=1:order], " "))
    function(t)
        o = 0; f = 1;
        for i=1:order
          f *= i
          o += symbols("κ$i") * t^i / f
        end
        o
    end
end
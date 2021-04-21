# ---------------------------------------------------------------

using TaylorSeries, SpecialPolynomials, SymPy

include("vendor_extra.jl")
include("cgflib.jl")

function edgeworth_coefficients(cgf, order; T=Float64)
    z = Taylor1(T, order)
    poly_exp = exp(z)
    
    cum_order = order * 2 + 1
    x = Taylor1(T, cum_order)

    poly_D = cgf(x)
    poly_N = x^2/2 # this is the cumulant generating function of N(0,1)
        
    # in this polynomial, the powers is taken on the differential operator applied to
    # the normal distribution. this means that we can simply replace the monomials
    # in the expansion by Hermite polynomials
    exp(poly_D - poly_N).coeffs
end

function edgeworth_sum(cgf, nsum, order; T=Float64)
    final_type = promote_rule(T, typeof(nsum))
    n = Pos(gensym("n"))
    
    mean, var = cumulants(cgf, 2; T=T)
    sumcgf = affine(cgf, -mean, 1/sqrt(var*n)) |> 
             cgf -> iidsum(cgf, n)

    # compute coefficients of the edgeworth expansion
    expansion_coeffs = edgeworth_coefficients(sumcgf, order; T=Sym)

    # prepare the coefficients by...
    preparecoeff = c -> truncate_order(c, n, -(order-1)/2) |> # 1. removing terms of higher order
                   c -> subs(c, n, nsum) |>                   # 2. substitute back symbolic n with it's value
                   c -> convert(final_type, c)                # 3. convert coeffs from Sym back to intended type
    coeffs = preparecoeff.(expansion_coeffs)

    # construct the polynomial based on the hermite basis and computed coefficients
    polynomial = sum([ coeffs[i] * basis(ChebyshevHermite, i-1)
                        for i=1:findlast(!iszero, coeffs) ])

    function density(z)
        κ₁ = sqrt(nsum)*mean
        x = (z - κ₁) / sqrt(var)
        return exp(-x^2/2)/sqrt(var*2pi) * polynomial(x)
    end
end

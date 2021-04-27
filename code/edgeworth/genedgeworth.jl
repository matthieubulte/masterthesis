# ---------------------------------------------------------------

using SpecialPolynomials, SymPy

include("vendor_extra.jl")
include("cgflib.jl")

function edgeworth(cgf, nsum, order; T=Float64)
    final_type = promote_rule(T, typeof(nsum))
    @vars t n::(positive, integer)

    # start by constructing the cgf of ∑(Xᵢ - μ)/σ√n
    mean, var = cumulants(cgf, 2; T=T)
    sumcgf = affine(cgf, -mean, 1/sqrt(var*n)) |> 
                cgf -> iidsum(cgf, n)

    # use the new cgf to construct the expansion of the ratio of characteristic functions
    taylororder = 3*order+1
    expansion = exp(sumcgf(t) - t^2/2).series(t, n=taylororder).removeO()

    prepare = s -> collect(expand(s), n) |> #  0. making sure terms are ready to be manipulated
              s -> truncate_order(s, n, -(order-1)/2) |> # 1. removing terms of higher order
              s -> subs(expand(s), n, nsum) |> # 2. substitute back symbolic n with it's value
              s -> collect(s, t).coeff.(t.^(0:taylororder)) # 4. collect coefficients of t^k that will be replaced with the k-th hermite polynomial

    hermitecoeffs = convert.(final_type, prepare(expansion))

    # construct the polynomial based on the hermite basis and computed coefficients
    polynomial = sum([ hermitecoeffs[i] * basis(ChebyshevHermite, i-1)
                            for i=1:findlast(!iszero, hermitecoeffs) ])
    function density(z)
        κ₁ = sqrt(nsum)*mean
        x = (z - κ₁) / sqrt(var)
        return exp(-x^2/2)/sqrt(var*2pi) * polynomial(x)
    end
end
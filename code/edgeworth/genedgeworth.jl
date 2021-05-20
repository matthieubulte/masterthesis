# ---------------------------------------------------------------

using SpecialPolynomials, SymPy
using Symbolics: map_subscripts

include("vendor_extra.jl")
include("cgflib.jl")

function H(i, x)
    if i == 0
        1
    else
        SymFunction("H$(map_subscripts(i))")(x)
    end
end

function edgeworth(cgf, nsum, order; T=Float64)
    final_type = promote_rule(T, typeof(nsum))
    symbolic = T == Sym
    @syms t n

    # start by constructing the cgf of ∑(Xᵢ - μ)/σ√n
    mean, var = cumulants(cgf, 2; T=T)
    sumcgf = affine(cgf, -mean, 1/sqrt(var*n)) |> 
                cgf -> iidsum(cgf, n)

    # use the new cgf to construct the expansion of the ratio of characteristic functions
    taylororder = 3*order+1
    charfunsdiff = exp(sumcgf(t) - t^2/2)
    expansion = charfunsdiff.series(t, n=taylororder).removeO()

    expansion = collect(expand(expansion), n) #  0. making sure terms are ready to be manipulated
    expansion = truncate_order(expansion, n, (1-order)/2) # 1. removing terms of higher order
    expansion = subs(expand(expansion), n, nsum) # 2. substitute back symbolic n with it's value
    hermitecoeffs = collect(expansion, t).coeff.(t.^(0:taylororder)) # 4. collect coefficients of t^k that will be replaced with the k-th hermite polynomial

    hermitecoeffs = convert.(final_type, hermitecoeffs)

    # construct the polynomial based on the hermite basis and computed coefficients
    polynomial = if symbolic
        (x) -> sum([ hermitecoeffs[i] * H(i-1, x)
                    for i=1:findlast(!iszero, hermitecoeffs) ])
    else
        sum([ hermitecoeffs[i] * basis(ChebyshevHermite, i-1)
            for i=1:findlast(!iszero, hermitecoeffs) ])
    end

    function density(z)
        κ₁ = sqrt(nsum)*mean
        x = (z - κ₁) / sqrt(var)
        return exp(-x^2/2)/sqrt(var*2π) * polynomial(x)
    end
end
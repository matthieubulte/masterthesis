# ---------------------------------------------------------------

using TaylorSeries, SpecialPolynomials, SymPy

include("vendor_extra.jl")
include("expfam.jl")
include("distributions.jl")

function H(n)
#   return basis(ChebyshevHermite, n)
  return symbols("H$n")
end

function remove_asympt(expr, term, opower)
    infty = SymPy.sympy.oo
    bigo = SymPy.sympy.O(term^opower, (term, infty))
    return (expr + bigo).removeO()
end

function edgeworth(d::CDistribution{T}, order) where {T<:Number}
    z = Taylor1(T, order)
    poly_exp = exp(z)
    
    cum_order = order * 3
    x = Taylor1(T, cum_order)

    poly_D = cumulant_gen_fn(d, x)
    poly_N = x^2/2 # this is the cumulant generating function of N(0,1)
        
    # in this polynomial, the powers is taken on the differential operator applied to
    # the normal distribution. this means that we can simply replace the monomials
    # in the expansion by Hermite polynomials
    expansion = exp(poly_D - poly_N)

    hermites = [ H(i) for i=1:cum_order ]
    
    return 1 + sum(hermites .* expansion.coeffs[2:end])
end


function edgeworth_sum(d, nterms, order)
    n = Pos(string("n", abs(rand(Int8)))) # just generate a random symbol
    symd = convert(Sym, d)

    mean, var = cumulants(symd, 2)
    scaled_d = AffineTransformed(symd, -mean, 1/sqrt(var*n))
    std_sum = IIDSum(scaled_d, n)
    
    expansion = edgeworth(std_sum, order)
    expansion = collect(expand(expansion), n)
    expansion = remove_asympt(expansion, n, -(order + 1)/2)
    return expansion.subs(n, nterms)
end



# base distribution
# d = EFNormal(3)
d = SymbolicDist(10)
# d = EFExponential(2)
order = 2
nterms = Pos("n")

edgeworth_sum(d, nterms, order)

using Debugger
Debugger.@enter edgeworth_sum(d, nterms, order)

# ---------------------------------------------------------------


# function mean_model(fam::MMExponentialFamily{M}, n_sum_terms)
#     distrib = to_distribution(fam)
#     @model function mod()
#         x ~ filldist(distrib, n_sum_terms)
#         centered_mean = sum(x) - n_sum_terms * d_mean
#         y ~ Dirac(centered_mean)
#     end
#     return mod()
# end

# function sample_noparams(model)
#     chain = sample(model, Prior(), mcmc_samples)
#     return chain[:y]
# end

# mcmc_samples = 1_000
# model = mean_model(EFExponential(1.0), 1000)

# @time density(sample_noparams(model))
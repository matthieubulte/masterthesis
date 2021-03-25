# ---------------------------------------------------------------

using TaylorSeries, SpecialPolynomials, SymPy, Distributions, Plots, ReverseDiff

include("edgeworth/vendor_extra.jl")
include("edgeworth/distributions.jl")
include("edgeworth/expfam.jl")

∇²(f) = x -> ReverseDiff.hessian(t -> f(t[1]), [x])[1]

function sample_sum(d, nterms, samplesize)
    result = zeros(samplesize)
    for i=1:nterms
        result .+= rand(d, samplesize)
    end
    result
end

function truncate_order(expr, term, opower)
    infty = SymPy.sympy.oo
    bigo = SymPy.sympy.O(term^opower, (term, infty))
    return (expr + bigo).removeO()
end

function edgeworth_coefficients(d::CDistribution{T}, order) where {T<:Number}
    z = Taylor1(T, order)
    poly_exp = exp(z)
    
    cum_order = order * 3
    x = Taylor1(T, cum_order)

    poly_D = cumulant_gen_fn(d, x)
    poly_N = x^2/2 # this is the cumulant generating function of N(0,1)
        
    # in this polynomial, the powers is taken on the differential operator applied to
    # the normal distribution. this means that we can simply replace the monomials
    # in the expansion by Hermite polynomials
    exp(poly_D - poly_N).coeffs
end

function edgeworth_sum(d::CDistribution{T}, nsum, order) where {T}
    n = Pos(string("n", abs(rand(Int8)))) # just generate a random symbol
    symd = convert(Sym, d)

    mean, var = cumulants(d, 2)
    scaled_d = AffineTransformed(symd, -mean, 1/sqrt(var*n))
    std_sum = IIDSum(scaled_d, n)

    # compute coefficients of the edgeworth expansion
    expansion_coeffs = edgeworth_coefficients(std_sum, order)

    # construct a symbolic polynomial (in terms of hermite basis) using the coefficients
    n_coefficients = size(expansion_coeffs)[1]
    symbolic_hermite_basis = [symbols("H$(i-1)") for i=1:n_coefficients]
    symbolic_expansion = sum(expansion_coeffs .* symbolic_hermite_basis)

    # remove terms of order higher than O(n^{-(order+1)/2})
    symbolic_expansion = collect(expand(symbolic_expansion), n)
    
    symbolic_expansion = truncate_order(symbolic_expansion, n, -(order-1)/2)
    # construct the proper polynomial based on user input (proper n, and actual polynomial instead of symbolic basis)
    symbolic_expansion = expand(symbolic_expansion.subs(n, nsum))
    sym_coefficients = [
        symbolic_expansion.coeff(Hᵢ) for Hᵢ=symbolic_hermite_basis
    ]

    final_type = promote_rule(T, typeof(nsum))
    final_coefficients = convert.(final_type, sym_coefficients)

    hermite_basis = [basis(ChebyshevHermite, i-1) for i=1:n_coefficients]
    polynomial = sum(hermite_basis .* final_coefficients)

    function density(z)
        κ₁ = sqrt(nsum)*mean

        x = (z - κ₁) / sqrt(var)
        return exp(-x^2/2)/sqrt(2pi) * polynomial(x)
    end

    return density
end

function saddlepoint_expfam(K, compθ̂, θ, nsum)
    function density(s)
        θ̂ = compθ̂(s)
        λ̂ = θ̂ - θ

        w = exp(nsum*K(λ̂) - s*λ̂)
        d = sqrt(2*pi*nsum*∇²(K)(λ̂))

        w/d
    end
end


function plot_empirical(d, nterms)
    sample = sample_sum(d, nterms, 100_000);
    p = histogram(sample, normalize=true, color="grey", alpha=0.3, label="Empirical")
    q = LinRange(minimum(sample), maximum(sample), 1000);
    p, q
end

function plot_approximations(
    nterms,
    distrib,
    cgf,
    mle,
    real_param
    )

    f = saddlepoint_expfam(cgf, mle, real_param, nterms)    
    d = FromCGF{Float64}(cgf)
    p, q = plot_empirical(distrib, nterms)
    plot!(p, q, f.(q), label="Saddlepoint")
    for i=2:4
        e = edgeworth_sum(d, nterms, i)
        # e is the density of X = sum(Y) / sqrt(n)
        # so if we want to evaluate the density of Z = sum(Y)/n = X/sqrt(n)
        # we have se(z) = |dz/dx|e(x(z)) = e(sqrt(n)*z)/sqrt(n)
        # note here that s = sum(Y),  so x(z) = x(z(s)) = s/sqrt(n)
        se(s) = e(s/sqrt(nterms))/sqrt(nterms)
        plot!(p, q, se.(q), label="Edgeworth-$i")
    end
    p
end

function plot_approximations_err(
    nterms,
    distrib,
    true_distrib,
    cgf,
    mle,
    real_param
    )

    f = saddlepoint_expfam(cgf, mle, real_param, nterms)    
    d = FromCGF{Float64}(cgf)

    sample = sample_sum(distrib, nterms, 100_000);
    q = LinRange(minimum(sample), maximum(sample), 1000)
    tp = pdf(true_distrib, q)

    p = plot(q, log10.(abs.(tp - f.(q)) ./ tp), label="Saddlepoint Err / True")
    for i=2:4
        e = edgeworth_sum(d, nterms, i)
        se(s) = e(s/sqrt(nterms))/sqrt(nterms)
        plot!(p, q, log10.(abs.(tp - se.(q)) ./ tp), label="Edgeworth-$i Err / True")
    end
    p
end


nterms = 10

# Γ(1, 1)
plot_approximations(nterms, 
    Gamma(1, 1),
    (t) -> -log(1-t),
    (s) -> -nterms/s,
    -1.0
)

plot_approximations_err(nterms, 
    Gamma(1, 1),
    Gamma(nterms, 1),
    (t) -> -log(1-t),
    (s) -> -nterms/s,
    -1.0
)

# N(μ, 1)
μ = 1
plot_approximations(nterms, 
    Normal(μ, 1),
    (t) -> μ*t + t^2/2, 
    (s) -> s/nterms,
    μ
)
plot_approximations_err(nterms, 
    Normal(μ, 1),
    Normal(nterms*μ, nterms),
    (t) -> μ*t + t^2/2, 
    (s) -> s/nterms,
    μ
)

# Exponential(1)
plot_approximations(nterms, 
    Exponential(1),
    (t) -> log(-1 / (-1 + t)), 
    (s) -> -nterms/s,
    -1.0
)∑
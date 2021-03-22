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

function plot_gamma(nterms)
    # Γ(1, 1)
    f = saddlepoint_expfam(
        (t) -> -log(1 - t), 
        (s) -> -nterms/s, 
        -1.0, nterms
    )

    p, q = plot_empirical(Gamma(1, 1), nterms)
    plot!(p, q, pdf(Gamma(nterms, 1), q), label="True density")
    plot!(p, q, f.(q), label="Saddlepoint")
end

function plot_normal(μ, nterms)
    # N(μ, 1)
    f = saddlepoint_expfam(
        (t) -> μ*t + t^2/2, 
        (s) -> s/nterms, 
        μ, nterms
    )

    p, q = plot_empirical(Normal(μ, 1), nterms)
    plot!(p, q, pdf(Normal(nterms*μ, sqrt(nterms)), q), label="True density", color="black")
    plot!(p, q, f.(q), label="Saddlepoint", color="red")
end

function plot_exp(nterms)
    # Exp(1)
    f = saddlepoint_expfam(
        (t) -> log(-1 / (-1+t)), 
        (s) -> -nterms/s, 
        -1.0, nterms
    )

    p, q = plot_empirical(Exponential(1), nterms)
    plot!(p, q, f.(q), label="Saddlepoint")
end

# base distribution
d = FromCGF{Float64}((t) -> t^2/2) # N(0,1)
d = FromCGF{Float64}((t) -> -log(1-t)) # Γ(1,1)
d = FromCGF{Float64}((t) -> log(-1 / (-1 + t))) # Exp(1)

plot_normal(1, 3)
plot_gamma(3)
plot_exp(3)
    

nterms = 10
q = LinRange(60, 140, 1000);

# plot densities
p = plot(q, tp, label="True; n=$nterms")
for i=2:4
    e = edgeworth_sum(d, nterms, i)
    plot!(p, q, e.(q), label="Edgeworth-$i")
end
ylims!(0, 0.7)
p

# absolute error 
p = plot()
for i=2:4
    e = edgeworth_sum(d, nterms, i)
    p = plot!(p, q, abs.(tp - e.(q)), label="Edgeworth-$i")
end
p

# relative error
p = plot()
for i=2:4
    e = edgeworth_sum(d, nterms, i)
    p = plot!(p, q, abs.(tp - e.(q)) ./ tp, label="Edgeworth-$i")
end
p





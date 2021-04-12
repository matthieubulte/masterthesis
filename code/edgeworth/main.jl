# ---------------------------------------------------------------

using TaylorSeries, SpecialPolynomials, SymPy, Distributions, Plots, ReverseDiff, LaTeXStrings

include("edgeworth/vendor_extra.jl")
include("edgeworth/cgflib.jl")
include("edgeworth/genedgeworth.jl")

∇²(f) = x -> ReverseDiff.hessian(t -> f(t[1]), [x])[1]

function sample_sum(d, nterms, samplesize)
    result = zeros(samplesize)
    for i=1:nterms
        result .+= rand(d, samplesize)
    end
    result
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

function plot_empirical(d, nterms; dosqrt=false)
    sample = sample_sum(d, nterms, 100_000);
    if dosqrt
        sample ./= sqrt(nterms)
    end

    p = histogram(sample, normalize=true, color="grey", alpha=0.3, label="Empirical")
    q = LinRange(minimum(sample), maximum(sample), 1000);
    p, q
end

function plot_approximations(
    nterms,
    cgf,
    mle,
    real_param,
    real_distrib;
    saddlepoint=false,
    xlim=nothing
    )

    if !isnothing(xlim)
        q = LinRange(xlim[1], xlim[2], 1000)
    else
        _, q = plot_empirical(distrib, nterms; dosqrt=true)
    end
    p = plot(q, pdf(real_distrib, q), label=L"\textrm{Truth}; n \textbf{=} %$nterms", color=:black)
    
    if saddlepoint
        f = saddlepoint_expfam(cgf, mle, real_param, nterms)    
        plot!(p, q, f.(q), label="Saddlepoint")
    end
    
    ls = [:dash, :dot, :dashdot]
    for i=2:4
        e = edgeworth_sum(cgf, nterms, i)
        # e is the density of X = sum(Y) / sqrt(n)
        # so if we want to evaluate the density of Z = sum(Y)/n = X/sqrt(n)
        # we have se(z) = |dz/dx|e(x(z)) = e(sqrt(n)*z)/sqrt(n)
        # note here that s = sum(Y),  so x(z) = x(z(s)) = s/sqrt(n)
        # se(s) = e(s/sqrt(nterms))/sqrt(nterms)
        se(s) = e(s)
        plot!(p, q, se.(q), label=L"\textrm{Edgeworth-%$i}", color=:black, linestyle=ls[i-1])
    end
    xlabel!(L"\textrm{y}")
    ylabel!(L"\textrm{f(y)}")
    p
end

function plot_approximations_err(
    nterms,
    cgf,
    true_distrib,
    mle,
    real_param;
    saddlepoint=false,
    xlim=nothing,
    ylim=nothing,
    relative=true,
    kwargs...
    )

    if isnothing(xlim)
        sample = sample_sum(distrib, nterms, 100_000) ./ sqrt(nterms);
        q = LinRange(minimum(sample), maximum(sample), 1000)
    else
        q = LinRange(xlim[1], xlim[2], 1000)
    end
    tp = pdf(true_distrib, q)

    if saddlepoint
        f = saddlepoint_expfam(cgf, mle, real_param, nterms)    
        p = plot(q, log10.(abs.(tp - f.(q)) ./ tp), label="Saddlepoint Err / True")
    else
        p = plot()
    end

    ls = [:dash, :dot, :dashdot]
    for i=2:4
        e = edgeworth_sum(cgf, nterms, i)
        # se(s) = e(s/sqrt(nterms))/sqrt(nterms)
        se = e

        if relative
            err = log10.(abs.(tp - se.(q)) ./ tp)
        else
            err = abs.(tp - se.(q))
        end

        plot!(p, q, err, label=L"\textrm{Edgeworth-%$i}", color=:black, linestyle=ls[i-1], kwargs...)
    end
    if !isnothing(ylim)
        ylims!(ylim)
    end
    xlabel!(L"\textrm{y}")
    if relative
        ylabel!(L"\textrm{relative error (log}_{10})")
    else
        ylabel!(L"\textrm{absolute error}")
    end
    p
end

inkscapegen(s) = println("inkscape code/plots/$(s).svg -o writing/figures/$(s).eps --export-ignore-filters --export-ps-level=3")

nterms=10; α = 2.0; θ = 1;
p = plot_approximations(nterms, 
    gamma(α, θ),
    (s) -> -nterms/s,
    -1.0,
    Gamma(nterms*α, θ/sqrt(nterms));
    saddlepoint=false,
    xlim=(2, 10)
)
plot!(p, size=(400,500), legendfontsize=10, legend=:topright)

# Γ(2, 1)
for nterms = [1; 10]
    α = 2.0; θ = 1.0;
    lims = nterms == 1 ? (0, 6) : (2, 10)

    p = plot_approximations(nterms, 
        gamma(α, θ),
        (s) -> -nterms/s,
        -1.0,
        Gamma(nterms*α, θ/sqrt(nterms));
        saddlepoint=false,
        xlim=lims
    )
    plot!(p, size=(400,500), legendfontsize=10, legend=:topright)
    Plots.svg(p, "plots/edgeworth_gamma21_$(nterms)_terms")
    inkscapegen("edgeworth_gamma21_$(nterms)_terms")
end

# Γ(1, 1)
for nterms = [1; 10]
    α = 1.0; θ = 1.0;
    p = plot_approximations(nterms, 
        gamma(α, θ),
        (s) -> -nterms/s,
        -1.0,
        Gamma(nterms*α, θ/sqrt(nterms));
        saddlepoint=false,
        xlim=(0, 6)
    )
    plot!(p, size=(400,500), legendfontsize=10, legend=:topright)
    Plots.svg(p, "plots/edgeworth_gamma11_$(nterms)_terms")
    inkscapegen("edgeworth_gamma11_$(nterms)_terms")
end

# Err plots Γ(1, 1)
for relative=[true; false]
    α = 1.0; θ = 1.0; nterms = 10; rel= relative ? "rel" : "abs";
    p = plot_approximations_err(nterms, 
        gamma(α, θ),
        Gamma(nterms*α, θ/sqrt(nterms)),
        (s) -> -nterms/s,
        -1.0;
        relative=relative,
        xlim=(0,6)
    )
    plot!(p, size=(400,500), legendfontsize=10, legend=:topright)
    Plots.svg(p, "plots/edgeworth_err_$(rel)_gamma11_10_terms")
    inkscapegen("edgeworth_err_$(rel)_gamma11_10_terms")
end

# Err plots Γ(2, 1)
for relative=[true; false]
    α = 2.0; θ = 1.0; nterms = 10; rel= relative ? "rel" : "abs";
    p = plot_approximations_err(nterms, 
        gamma(α, θ),
        Gamma(nterms*α, θ/sqrt(nterms)),
        (s) -> -nterms/s,
        -1.0;
        relative=relative,
        xlim=(2, 10)
    )
    plot!(p, size=(400,500), legendfontsize=10, legend=:topright)
    Plots.svg(p, "plots/edgeworth_err_$(rel)_gamma21_10_terms")
    inkscapegen("edgeworth_err_$(rel)_gamma21_10_terms")
end
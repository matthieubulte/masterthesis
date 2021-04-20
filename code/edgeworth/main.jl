# ---------------------------------------------------------------

using TaylorSeries, SpecialPolynomials, SymPy, Distributions, Plots, ReverseDiff, LaTeXStrings

include("edgeworth/vendor_extra.jl")
include("edgeworth/cgflib.jl")
include("edgeworth/genedgeworth.jl")
include("edgeworth/gensaddlepoint.jl")

inkscapegen(s) = println("inkscape code/plots/$(s).svg -o writing/figures/$(s).eps --export-ignore-filters --export-ps-level=3")

function sample_sum(d, nterms, samplesize)
    result = zeros(samplesize)
    for i=1:nterms
        result .+= rand(d, samplesize)
    end
    result
end

function plot_approximations(
    nterms,
    cgf,
    real_distrib;
    incl_saddlepoint=true,
    incl_edgeworth=true,
    xlim=nothing
    )

    q = LinRange(xlim[1], xlim[2], 1000)
    p = plot(q, pdf(real_distrib, q), label=L"\textrm{Truth}; n \textbf{=} %$nterms", color=:black)
    
    if incl_saddlepoint
        f = dscale(saddlepoint(cgf, nterms), 1/sqrt(nterms))
        plot!(p, q, f.(q), label="Saddlepoint")
    end
    
    if incl_edgeworth
        ls = [:dash, :dot, :dashdot]
        for i=2:4
            e = edgeworth_sum(cgf, nterms, i)
            plot!(p, q, e.(q), label=L"\textrm{Edgeworth-%$i}", color=:black, linestyle=ls[i-1])
        end
    end

    xlabel!(L"\textrm{y}")
    ylabel!(L"\textrm{f(y)}")
    p
end

function plot_approximations_err(
    nterms,
    cgf,
    true_distrib;
    incl_saddlepoint=true,
    incl_edgeworth=true,
    xlim=nothing,
    relative=true,
    kwargs...
    )

    q = LinRange(xlim[1], xlim[2], 1000)
    tp = pdf(true_distrib, q)

    if incl_saddlepoint
        f = dscale(saddlepoint(cgf, nterms), 1/sqrt(nterms))
        p = plot(q, log10.(abs.(tp - f.(q)) ./ tp), label=L"\textrm{Saddlepoint}", color=:black)
    else
        p = plot()
    end

    if incl_edgeworth
        ls = [:dash, :dot, :dashdot]
        for i=2:4
            e = edgeworth_sum(cgf, nterms, i)
            err = abs.(tp - e.(q))
            if relative
                err = log10.(err ./ tp)
            end

            plot!(p, q, err, label=L"\textrm{Edgeworth-%$i}", color=:black, linestyle=ls[i-1], kwargs...)
        end
    end

    xlabel!(L"\textrm{y}")
    ylab = relative ? L"\textrm{relative error (log}_{10})"
                    : L"\textrm{absolute error}"
    ylabel!(ylab)

    p
end


nterms=10; α = 2.0; θ = 1.0; q = LinRange(1e-8, 2.5, 1000);

f = dscale(saddlepoint(gamma(α, θ), nterms), 1/sqrt(nterms))
f = dscale(saddlepoint(_uniform(0, 1), nterms), 1/sqrt(nterms))


histogram(sample_sum(Uniform(0, 1), nterms, 10000) ./ sqrt(nterms); normalize=true, color="grey", alpha=0.3); plot!(q, f, label="Saddlepoint")




plot!(q, pdf.(Gamma(nterms*α, θ/sqrt(nterms)), q), label="Truth")


p = plot_approximations_err(nterms, 
    gamma(α, θ),
    Gamma(nterms*α, θ/sqrt(nterms));
    compsaddlepoint=true,
    xlim=(0, 6)
)
plot!(p, size=(400,500), legendfontsize=10, legend=:topright)

# Γ(2, 1)
for nterms = [1; 10]
    α = 2.0; θ = 1.0;
    lims = nterms == 1 ? (0, 6) : (2, 10)

    p = plot_approximations(nterms, 
        gamma(α, θ),
        Gamma(nterms*α, θ/sqrt(nterms));
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
        Gamma(nterms*α, θ/sqrt(nterms));
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
        relative=relative,
        xlim=(2, 10)
    )
    plot!(p, size=(400,500), legendfontsize=10, legend=:topright)
    Plots.svg(p, "plots/edgeworth_err_$(rel)_gamma21_10_terms")
    inkscapegen("edgeworth_err_$(rel)_gamma21_10_terms")
end




using Symbolics, SymbolicUtils, LinearAlgebra

@variables x[1:10]

e = exp(-x'x)

Dx = Differential.(x)

substitute(
    expand_derivatives((Dx[2] ∘ Dx[1])(e))    
, Dict([ x[i] => 0 for i=1:10]))


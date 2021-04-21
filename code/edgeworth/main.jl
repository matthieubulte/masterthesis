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
    truepdf;
    incl_saddlepoint=true,
    incl_edgeworth=true,
    xlim=nothing
    )

    q = LinRange(xlim[1], xlim[2], 1000)
    p = plot(q, truepdf, label=L"\textrm{Truth}; n \textbf{=} %$nterms", color=:black)
    
    if incl_saddlepoint
        f = dscale(saddlepoint(cgf, nterms), 1/sqrt(nterms))
        plot!(p, q, f.(q), label="Saddlepoint")
    end
    
    if incl_edgeworth
        ls = [:dash, :dot, :dashdot]
        for i=2:4
            e = edgeworth_sum(cgf, nterms, i)
            plot!(p, q, e.(q), label=L"\textrm{Edgeworth-%$i}", linestyle=ls[i-1])
        end
    end

    xlabel!(L"\textrm{y}")
    ylabel!(L"\textrm{f(y)}")
    p
end

function plot_approximations_err(
    nterms,
    cgf,
    truepdf;
    incl_saddlepoint=true,
    incl_edgeworth=true,
    xlim=nothing,
    relative=true,
    kwargs...
    )

    q = LinRange(xlim[1], xlim[2], 1000)
    tp = truepdf.(q)

    relerror(f) = log10.(abs.(tp - f.(q)) ./ tp)
    abserror(f) = abs.(tp - f.(q))
    err = relative ? relerror : abserror

    p = plot()

    if incl_saddlepoint
        f = dscale(saddlepoint(cgf, nterms), 1/sqrt(nterms))
        p = plot!(p, q, err(f),
            label=L"\textrm{Saddlepoint}",
            color=:black)
    end

    if incl_edgeworth
        ls = [:dash, :dot, :dashdot]
        for i=2:4
            e = edgeworth_sum(cgf, nterms, i)
            plot!(p, q, err(e), 
                label=L"\textrm{Edgeworth-%$i}", 
                color=:black, linestyle=ls[i-1],
                kwargs...)
        end
    end

    xlabel!(L"\textrm{y}")
    ylab = relative ? L"\textrm{relative error (log}_{10})" : L"\textrm{absolute error}"
    ylabel!(ylab)

    p
end


nterms=10; α = 2.0; θ = 1.0;

p = plot_approximations(nterms, 
    gamma(α, θ),
    _pdf(Gamma(nterms*α, θ/sqrt(nterms)));
    xlim=(2, 10)
)

p = plot_approximations_err(nterms, 
    gamma(α, θ),
    _pdf(Gamma(nterms*α, θ/sqrt(nterms)));
    xlim=(2, 10),
    incl_saddlepoint=false,
    relative=false
)

# Γ(2, 1)
for nterms = [1; 10]
    α = 2.0; θ = 1.0;
    lims = nterms == 1 ? (0, 6) : (2, 10)

    p = plot_approximations(nterms, 
        gamma(α, θ),
        _pdf(Gamma(nterms*α, θ/sqrt(nterms)));
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
        _pdf(Gamma(nterms*α, θ/sqrt(nterms)));
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
        _pdf(Gamma(nterms*α, θ/sqrt(nterms))),
        relative=relative,
        xlim=(2, 10)
    )
    plot!(p, size=(400,500), legendfontsize=10, legend=:topright)
    Plots.svg(p, "plots/edgeworth_err_$(rel)_gamma21_10_terms")
    inkscapegen("edgeworth_err_$(rel)_gamma21_10_terms")
end
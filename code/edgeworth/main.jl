# ---------------------------------------------------------------

using TaylorSeries, SpecialPolynomials, SymPy, Distributions, Plots, ReverseDiff, LaTeXStrings

include("edgeworth/vendor_extra.jl")
include("edgeworth/cgflib.jl")
include("edgeworth/genedgeworth.jl")
include("edgeworth/gensaddlepoint.jl")
include("edgeworth/genpstar.jl")
include("plotutils.jl")

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
    truedistrib;
    incl_saddlepoint=true,
    incl_edgeworth=true,
    xlim=nothing
    )

    q = LinRange(xlim[1], xlim[2], 1000)
    p = plot(q, _pdf(truedistrib), label=L"\textrm{Truth}; n \textbf{=} %$nterms", color=:black)
    
    if incl_saddlepoint
        f = dscale(saddlepoint(cgf, nterms), 1/sqrt(nterms))
        plot!(p, q, f.(q), label=L"\textrm{Saddlepoint}", color=:black)
    end
    
    if incl_edgeworth
        ls = [:dash, :dot, :dashdot]
        for i=2:4
            e = edgeworth_sum(cgf, nterms, i)
            plot!(p, q, e.(q), label=L"\textrm{Edgeworth-%$i}", linestyle=ls[i-1], color=:black)
        end
    end

    xlabel!(L"\textrm{y}")
    ylabel!(L"\textrm{f(y)}")
    p
end

function plot_approximations_err(
    nterms,
    cgf,
    truedistrib;
    incl_saddlepoint=true,
    incl_edgeworth=true,
    xlim=nothing,
    relative=true,
    kwargs...
    )

    p = plot()

    if incl_saddlepoint
        plot_approximation_err!(p,
            truedistrib,
            dscale(saddlepoint(cgf, nterms), 1/sqrt(nterms));
            xlim=xlim,
            relative=relative,
            label=L"\textrm{Saddlepoint}",
            color=:black
        )
    end

    if incl_edgeworth
        ls = [:dash, :dot, :dashdot]
        for i=2:4
            plot_approximation_err!(p,
                truedistrib,
                edgeworth(cgf, nterms, i);
                xlim=xlim,
                relative=relative,
                label=L"\textrm{Edgeworth-%$i}",
                color=:black, linestyle=ls[i-1]
            )
        end
    end

    p
end

function plot_approximation_err!(p,
    truedistrib,
    approx;
    xlabel=L"\textrm{y}",
    xlim=nothing,
    ylim=nothing,
    relative=true,
    kwargs...
    )

    q = LinRange(xlim[1], xlim[2], 1000)
    tp = pdf(truedistrib, q)

    relerror(f) = log10.(abs.(tp - f.(q)) ./ tp)
    abserror(f) = abs.(tp - f.(q))
    err = relative ? relerror : abserror

    plot!(p, q, err(approx); kwargs...)

    xlabel!(p, xlabel)
    ylab = relative ? L"\textrm{relative error (log}_{10})" : L"\textrm{absolute error}"
    ylabel!(p, ylab)
    if !isnothing(ylim)
        ylims!(ylim[1], ylim[2])
    end

    p
end


begin
    ?? = 2.0; ?? = -??; nterms=10;
    s = sample_sum(Exponential(1/??), nterms, 100000) ./ nterms; sort!(s); ???? = -1 ./ s;

    pstar???? = pstar((??, ????) -> nterms*(log(-??) - ??/????), ??, nterms)

    ???? = -????;
    pstar???? = (??) -> pstar????(-??)

    rel = true; xlim=(0.2, 5)
    p = plot_approximation_err!(plot(),
        InverseGamma(nterms, ??*nterms),
        pstar????;
        xlim=xlim,
        relative=rel,
        color=:black, label=L"p^*"
    )
    plot_approximation_err!(p,
        InverseGamma(nterms, ??*nterms),
        _pdf(Normal(??, ?? ./ sqrt(nterms)));
        xlim=xlim,
        relative=rel,
        color=:black, label=L"\textrm{Normal}",
        linestyle=:dot, xlabel=L"\hat\lambda", ylim=(-4,5)
    )
    plot!(p, size=(400,500), legendfontsize=10, legend=:topright)

    Plots.svg(p, "plots/pstar_exp_err")
    inkscapegen("pstar_exp_err", "pstar_exp_err")
end

begin
    ?? = 2.0; ?? = -??; nterms=10;
    s = sample_sum(Exponential(1/??), nterms, 100000) ./ nterms; sort!(s); ???? = -1 ./ s;

    pstar???? = pstar((??, ????) -> nterms*(log(-??) - ??/????), ??, nterms)

    ???? = -????;
    pstar???? = (??) -> pstar????(-??)

    xlim=(0.2, 5)
    q = LinRange(xlim[1], xlim[2], 1000)
    p = plot(q, pdf(InverseGamma(nterms, ??*nterms), q), label=L"\textrm{Truth}", color=:black)

    plot!(p, q, pstar????.(q), color=:black, label=L"p^*", linestyle=:dash)

    plot!(p, q, pdf(Normal(??, ?? ./ sqrt(nterms)), q), color=:black, label=L"\textrm{Normal}", linestyle=:dot)

    xlabel!(p, L"\hat\lambda")
    ylabel!(p, L"f(\hat\lambda)")
    plot!(p, size=(400,500), legendfontsize=10, legend=:topright)

    Plots.svg(p, "plots/pstar_exp_dens")
    inkscapegen("pstar_exp_dens")
end

# ??(2, 1)
for nterms = [1; 10]
    ?? = 2.0; ?? = 1.0;
    lims = nterms == 1 ? (0, 6) : (2, 10)

    p = plot_approximations(nterms, 
        gamma(??, ??),
        Gamma(nterms*??, ??/sqrt(nterms));
        xlim=lims, incl_saddlepoint=false
    )
    plot!(p, size=(400,500), legendfontsize=10, legend=:topright)
    Plots.svg(p, "plots/edgeworth_gamma21_$(nterms)_terms")
    inkscapegen("edgeworth_gamma21_$(nterms)_terms")
end

# ??(1, 1)
for nterms = [1; 10]
    ?? = 1.0; ?? = 1.0;
    p = plot_approximations(nterms, 
        gamma(??, ??),
        Gamma(nterms*??, ??/sqrt(nterms));
        xlim=(0, 6), incl_saddlepoint=false
    )
    plot!(p, size=(400,500), legendfontsize=10, legend=:topright)
    Plots.svg(p, "plots/edgeworth_gamma11_$(nterms)_terms")
    inkscapegen("edgeworth_gamma11_$(nterms)_terms")
end

# Err plots ??(1, 1)
for relative=[true; false]
    ?? = 1.0; ?? = 1.0; nterms = 10; rel= relative ? "rel" : "abs";
    p = plot_approximations_err(nterms, 
        gamma(??, ??),
        Gamma(nterms*??, ??/sqrt(nterms)),
        relative=relative,
        xlim=(0,6), incl_saddlepoint=false
    )
    plot!(p, size=(400,500), legendfontsize=10, legend=:topright)
    Plots.svg(p, "plots/edgeworth_err_$(rel)_gamma11_10_terms")
    inkscapegen("edgeworth_err_$(rel)_gamma11_10_terms")
end

# Err plots ??(2, 1)
for relative=[true; false]
    ?? = 2.0; ?? = 1.0; nterms = 10; rel= relative ? "rel" : "abs";
    p = plot_approximations_err(nterms, 
        gamma(??, ??),
        Gamma(nterms*??, ??/sqrt(nterms)),
        relative=relative,
        xlim=(2, 10), incl_saddlepoint=false
    )
    plot!(p, size=(400,500), legendfontsize=10, legend=:topright)
    Plots.svg(p, "plots/edgeworth_err_$(rel)_gamma21_10_terms")
    inkscapegen("edgeworth_err_$(rel)_gamma21_10_terms")
end

# Err plots ??(2, 1) with saddlepoint
for relative=[true; false]
    ?? = 2.0; ?? = 1.0; nterms = 10; rel= relative ? "rel" : "abs";
    p = plot_approximations_err(nterms, 
        gamma(??, ??),
        Gamma(nterms*??, ??/sqrt(nterms)),
        relative=relative,
        xlim=(2, 10)
    )
    plot!(p, size=(400,500), legendfontsize=10, legend=:topright)
    Plots.svg(p, "plots/saddlepoint_and_edgeworth_err_$(rel)_gamma21_10_terms")
    inkscapegen("saddlepoint_and_edgeworth_err_$(rel)_gamma21_10_terms")
end 
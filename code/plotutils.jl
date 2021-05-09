using Plots, StatsPlots, Dates, LaTeXStrings

inkscapegen(st, ss) = println("inkscape code/plots/$(ss).svg -o writing/figures/$(st).eps --export-ignore-filters --export-ps-level=3")

function plot_hist(pvals, ps, title, filename; xlabel=L"\textrm{Dimension p}", ylabel=L"")
    labels = ["$p" for p = ps]
    
    p = boxplot(pvalues[1, :], legend=false, color=:black, fillalpha=0.1, markersize=2.5)
    
    for i = 2:size(ps)[1]
        p = boxplot!(pvals[i, :], legend=false, color=:black, fillalpha=0.1, markersize=2.5)
    end

    title!(title)
    xticks!(1:size(ps)[1], labels)
    xlabel!(xlabel)
    ylabel!(ylabel)
    
    now = Dates.format(Dates.now(), "yyyymmddHHMM")
    file = "$(now)_$(filename)"
    Plots.svg(p, "./plots/$file")
    inkscapegen(filename, file)
    
    p
end 
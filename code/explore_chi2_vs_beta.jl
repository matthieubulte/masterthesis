### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 2d9b772e-784a-11eb-0abb-31705d4479d2
begin
	using Plots
	using StatsPlots
	using Distributions
	using PlutoUI
end

# ╔═╡ 7f4853de-784b-11eb-33c1-6fc9f96a70bb
@bind ln Slider(1:10)

# ╔═╡ 40fac932-784a-11eb-048f-a1be603cce9b
begin
	n = [3; 5; 7; 10; 25; 50; 75; 100; 1_000; 10_000][ln]
	f = n-2
	qvals = LinRange(0,  1, 100)
	q2nvals = qvals.^(2/n)
	wvals = -2*log.(qvals)
	
	qχ² = 1.0 .- cdf(Chisq(1), wvals);
	qβ = cdf(Beta((n - f - 1)/2, 1/2), q2nvals);
	"n = $(n), f = $(f)";
end

# ╔═╡ 40e49f0e-784a-11eb-0d0c-e1be715ee64e
begin
	p = plot(subplot=1, qvals, qχ², label="χ²(1)",layout = (1, 2))
	plot!(subplot=1, p, qvals, qβ, label="Beta")
	ylims!(subplot=1, 0, 1)
	plot!(subplot=2, qvals, log10.(abs.(qχ² - qβ)), label="log10 |χ²(w) - Beta(q̃)|")
	ylims!(subplot=2, -6, 0)
	xlabel!("q")
	p
end

# ╔═╡ Cell order:
# ╠═2d9b772e-784a-11eb-0abb-31705d4479d2
# ╠═7f4853de-784b-11eb-33c1-6fc9f96a70bb
# ╠═40fac932-784a-11eb-048f-a1be603cce9b
# ╠═40e49f0e-784a-11eb-0d0c-e1be715ee64e

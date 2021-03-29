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

# ╔═╡ ed672e60-900d-11eb-2eb0-b978142dc0e3
using Distributions, Plots, ReverseDiff, PlutoUI

# ╔═╡ 073f9a84-900e-11eb-3b09-875e786245ad
begin
	nterms = 10
	q = 0:0.1:30
	d = Gamma(nterms, 1)
	
	∇²(f) = x -> ReverseDiff.hessian(t -> f(t[1]), [x])[1]
	
	function saddlepoint_expfam(K, compθ̂, θ, nsum)
		function density(s)
			θ̂ = compθ̂(s)
			λ̂ = θ̂ - θ

			w = exp(nsum*K(λ̂) - s*λ̂)
			d = sqrt(2*pi*nsum*∇²(K)(λ̂))

			w/d
		end
	end
	
	K(t) = -log(1-t)
	θ̂(s) = -nterms/s
	θ = -1.0
	
	saddle_approx = saddlepoint_expfam(K, θ̂, θ, nterms);
end

# ╔═╡ 3cfd1a5c-900e-11eb-1ca3-679926e7eae8
@bind s₀ Slider(1:0.1:30)

# ╔═╡ 126cb1ee-900e-11eb-396a-87406553bb40
begin
	f = begin
		θ̂₀ = θ̂(s₀)
		λ̂ = θ̂₀ - θ

		(s) -> exp(nterms*K(λ̂) - s*λ̂) / sqrt(2*pi*nterms*∇²(K)(λ̂))
	end
	
	ps = pdf(d, q)
	ss = f.(q)
	u = maximum(ps)*1.5
	
	p = plot(q, ps, label="True density")
	plot!(p, q, saddle_approx.(q), label="Saddlepoint approx")
	plot!(p, q[ss .< u], ss[ss .< u], label="Exp family of saddlepoint approx at s₀=$(s₀)")
	plot!(p, [s₀; s₀], [0; f(s₀)], color=:black, linestyle=:dash, label=nothing)
	ylims!(0, u)
end

# ╔═╡ Cell order:
# ╠═ed672e60-900d-11eb-2eb0-b978142dc0e3
# ╠═073f9a84-900e-11eb-3b09-875e786245ad
# ╠═3cfd1a5c-900e-11eb-1ca3-679926e7eae8
# ╠═126cb1ee-900e-11eb-396a-87406553bb40

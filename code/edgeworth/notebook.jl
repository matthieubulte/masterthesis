### A Pluto.jl notebook ###
# v0.12.18

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

# ╔═╡ 08457f16-513e-11eb-30db-53a363bacee1
using PlutoUI, Plots

# ╔═╡ c847e7aa-513d-11eb-1e0f-47640b25782d
include("main.jl")

# ╔═╡ f58d42f0-513d-11eb-360d-8b3b25357d73
d = EFExponential(2.0)

# ╔═╡ f754270c-513d-11eb-29e7-8d84b061cca0
@bind order Slider(2:10; show_value=true)

# ╔═╡ 25da9a82-513e-11eb-0f37-0de479f02a37
@bind nterms Slider(1:5:100; show_value=true)

# ╔═╡ 3ac99420-513e-11eb-0ca3-61e2136b6699
edge = edgeworth_sum(d, nterms, order)

# ╔═╡ 56efc932-513e-11eb-3aff-315be14ffcff
plot(edge, -5:0.1:5)

# ╔═╡ Cell order:
# ╠═08457f16-513e-11eb-30db-53a363bacee1
# ╠═c847e7aa-513d-11eb-1e0f-47640b25782d
# ╠═f58d42f0-513d-11eb-360d-8b3b25357d73
# ╠═f754270c-513d-11eb-29e7-8d84b061cca0
# ╠═25da9a82-513e-11eb-0f37-0de479f02a37
# ╠═3ac99420-513e-11eb-0ca3-61e2136b6699
# ╠═56efc932-513e-11eb-3aff-315be14ffcff

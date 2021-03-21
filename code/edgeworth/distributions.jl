include("vendor_extra.jl")

# ---------------------------------------------------------------

abstract type CDistribution{T <: Number} end

function cumulant_gen_fn(::CDistribution{T}, ::T) where {T <: Number}
    error("unimplemented")
end

function cumulant_gen_fn(d::CDistribution{T}, t::S) where {T<:Number, S<:Number}
    return cumulant_gen_fn(convert(S, d), t)
end

function to_distribution(::CDistribution)
    error("unimplemented")
end

function Base.convert(::Type{S}, ::T)::CDistribution{S} where {T<:CDistribution,S<:Number}
    error("unimplemented")
end

function Base.convert(::Type{T}, d::CDistribution{T})::CDistribution{T} where {T<:CDistribution}
    return d
end

function cumulants(d::CDistribution{T}, order) where {T <: Number}
    t = Taylor1(T, order)
    scaled_cumulants = cumulant_gen_fn(convert(typeof(t), d), t).coeffs
    scaling = exp(t).coeffs
    return (scaled_cumulants ./ scaling)[2:end]
end

# ---------------------------------------------------------------

struct CumulantDist{T <: Number} <: CDistribution{T}
    cumulants::Vector{T}
end

function Base.convert(s::Type{S}, d::CumulantDist{T})::CumulantDist{S} where {T<:Number,S<:Number}
    return CumulantDist(convert(Vector{S}, d.cumulants))
end

function cumulant_gen_fn(d::CumulantDist{T}, t::T) where {T <: Number}
  x = Taylor1(eltype(t), length(d.cumulants))
  coeffs = exp(x).coeffs[2:end] .* d.cumulants

  poly = fill(t, (length(d.cumulants), 1))
  for i=1:length(d.cumulants)
    poly[i] = t^i
  end
  return sum(poly .* coeffs)
end

function SymbolicDist(ncumulants)::CumulantDist{Sym}
  symstr = join(["κ$i" for i=1:ncumulants], " ")
  cumulants = [sym for sym = symbols(symstr)]
  return CumulantDist{Sym}(cumulants)
end

# ---------------------------------------------------------------

struct FromCGF{T <: Number} <: CDistribution{T}
    cum_gen_fn::Function
end

function Base.convert(s::Type{S}, d::FromCGF{T})::FromCGF{S} where {T<:Number,S<:Number}
    return FromCGF{S}(d.cum_gen_fn)
end

function cumulant_gen_fn(d::FromCGF{T}, t::T) where {T <: Number}
    return d.cum_gen_fn(t)
end

# ---------------------------------------------------------------

struct IIDSum{T <: Number, U<:Number} <: CDistribution{T}
    distrib::CDistribution{T}
    n::U
end

function Base.convert(s::Type{S}, d::IIDSum{T})::IIDSum{S} where {T<:Number,S<:Number}
    return IIDSum(convert(S, d.distrib), convert(S, d.n))
end

function cumulant_gen_fn(d::IIDSum{T}, t::T) where {T <: Number}
    return d.n*cumulant_gen_fn(d.distrib, t)
end

# ---------------------------------------------------------------

struct AffineTransformed{T <: Number} <: CDistribution{T}
    distrib::CDistribution{T}
    location
    scale
end

function Base.convert(s::Type{S}, d::AffineTransformed{T})::AffineTransformed{S} where {T<:Number,S<:Number}
    return AffineTransformed(convert(S, d.distrib), convert(S, d.location), convert(S, d.scale))
end

function cumulant_gen_fn(d::AffineTransformed{T}, t::T) where {T <: Number}
    return cumulant_gen_fn(d.distrib, t * d.scale) + t * d.location * d.scale
end
include("distributions.jl")

# ---------------------------------------------------------------

abstract type MMExponentialFamily{T <: Number} <: CDistribution{T} end

function nat_param(::MMExponentialFamily{T})::Vector{T} where {T <: Number}
  error("unimplemented")
end

function from_nat_param(::T, nat_param::Vector{S})::T where {S<:Number, T<:MMExponentialFamily}
  error("unimplemented")
end

function log_partition(::MMExponentialFamily)
  error("unimplemented")
end

function Base.convert(s::Type{S}, d::MMExponentialFamily{T})::MMExponentialFamily{S} where {T<:Number,S<:Number}
  nparam = convert(Vector{S}, nat_param(d))
  return from_nat_param(d, nparam)
end

function cumulant_gen_fn(fam::MMExponentialFamily{T}, t::T) where {T <: Number}
  shifted = from_nat_param(fam, nat_param(fam) .+ t)
  return log_partition(shifted) - log_partition(fam)
end

# ---------------------------------------------------------------

struct EFExponential{T <: Number} <: MMExponentialFamily{T}
  lambda::T
end

function nat_param(e::EFExponential{T})::Vector{T} where {T <: Number}
  return [-e.lambda]
end

function from_nat_param(::EFExponential, nat_param::Vector{T}) where {T <: Number}
  return EFExponential(-nat_param[1],)
end

function to_distribution(e::EFExponential{T}) where {T <: Number}
  return Exponential(1/e.lambda)
end

function log_partition(e::EFExponential{T}) where {T <: Number}
  return -log(e.lambda)
end

# ---------------------------------------------------------------

struct EFNormal{T <: Number} <: MMExponentialFamily{T}
  mean
end

function nat_param(e::EFNormal)
  return [e.mean]
end

function from_nat_param(::EFNormal, nat_param)
  return EFNormal(nat_param[1])
end

function to_distribution(e::EFNormal)
  return Normal(e.mean, 1)
end

function log_partition(e::EFNormal)
  return e.mean .^ 2
end

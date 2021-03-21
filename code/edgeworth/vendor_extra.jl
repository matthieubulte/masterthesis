using TaylorSeries, SymPy

Base.:*(b::T, a::Taylor1{T}) where {T<:Number} = a * b

Base.promote_rule(::Type{Sym}, ::Type{T}) where {T<:Number} = Sym
Base.promote_rule(::Type{T}, ::Type{Sym}) where {T<:Number} = Sym
Base.promote_rule(::Type{Sym}, ::Type{Sym}) where {T<:Number} = Sym

function Base.:*(a::Taylor1{T}, b::T) where {T<:Number}
    coeffs = copy(a.coeffs)
    v = similar(a.coeffs)
    v = a.coeffs .* b
    return Taylor1{T}(v, a.order)
end

function Real(s)
    return symbols(s, real=True)
end

function Pos(s)
    return symbols(s, real=True, positive=True)
end

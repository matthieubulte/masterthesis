using TaylorSeries, SymPy, ReverseDiff

Base.:*(b::T, a::Taylor1{T}) where {T<:Number} = a * b

Base.promote_rule(::Type{Sym}, ::Type{T}) where {T<:Number} = Sym
Base.promote_rule(::Type{T}, ::Type{Sym}) where {T<:Number} = Sym
Base.promote_rule(::Type{Bool}, ::Type{Sym}) = Sym
Base.promote_rule(::Type{Sym}, ::Type{Bool}) = Sym
Base.promote_rule(::Type{Sym}, ::Type{Sym}) where {T<:Number} = Sym

function Base.:*(a::Taylor1{T}, b::T) where {T<:Number}
    coeffs = copy(a.coeffs)
    v = similar(a.coeffs)
    v = a.coeffs .* b
    return Taylor1{T}(v, a.order)
end

function Real(s)
    return symbols(s, real=true)
end

function Pos(s)
    return symbols(s, real=true, positive=true)
end

function Neg(s)
    return symbols(s, real=true, positive=false)
end

function truncate_order(expr, term, opower)
    infty = SymPy.sympy.oo
    bigo = SymPy.sympy.O(term^opower, (term, infty))
    return (expr + bigo).removeO()
end

∇²(f) = x -> ReverseDiff.hessian(t -> f(t[1]), [x])[1]
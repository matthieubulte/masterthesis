using TaylorSeries, SymPy

Base.:*(b::T, a::Taylor1{T}) where {T<:Number} = a * b

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

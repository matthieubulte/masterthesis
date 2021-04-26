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

function Pos(s)
    return symbols(s, real=true, positive=true)
end

function truncate_order(expr, term, opower)
    infty = SymPy.sympy.oo
    bigo = SymPy.sympy.O(term^opower, (term, infty))
    return (expr + bigo).removeO()
end

∇²(f) = x -> ReverseDiff.hessian(t -> f(t[1]), [x])[1]


macro vars(x...)
    q = Expr(:block)
    syms = []
    for expr in x
        sym = nothing
        varname = nothing
        assumptions = []

        res = parserename(expr)
        if !isnothing(res)
            expr, varname = res
        end

        if isa(expr, Symbol)
            sym = expr
        end
            
        res = parseassumptions(expr)
        if !isnothing(res)
            sym, assumptions = res
        end

        if isnothing(varname)
            varname = String(sym)
        end
        
        asstokw(a) = Expr(:kw, esc(a), true)
        ex = :($(esc(sym)) = $(esc(symbols))($(varname), $(map(asstokw, assumptions)...)))

        push!(syms, sym)
        push!(q.args, ex)
    end
    push!(q.args, Expr(:tuple, map(esc,syms)...))
    q
end

function parserename(s)
    isa(s, Expr) || return nothing
    (s.head == :call && s.args[1] == :(=>)) || return nothing
    expr, strname = s.args[2:end]
    return expr, String(strname)
end

function parseassumptions(s)
    isa(s, Expr) || return nothing
    s.head == :(::) || return nothing
    length(s.args) == 2 || return nothing

    sym, assumptions = s.args

    if isa(assumptions, Symbol)
        assumptions = (assumptions,)
    elseif isa(assumptions, Expr) && assumptions.head == :tuple
        assumptions = assumptions.args
    else
        return nothing
    end

    sym, assumptions
end

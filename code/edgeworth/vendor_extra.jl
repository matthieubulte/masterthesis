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

using SymPy

macro syms(xs...)
    # If the user separates declaration with commas, the top-level expression is a tuple
    if length(xs) == 1 && xs[1].head == :tuple
        return :(@syms($(xs[1].args...)))
    elseif length(xs) == 0
        return nothing
    end

    asstokw(a) = Expr(:kw, esc(a), true)
    
    # Each declaration is parsed and generates a declaration using `symbols`
    symdefs = map(xs) do expr
        varname, sym, assumptions, isfun = parsedecl(expr)
        ctor = isfun ? :SymFunction : :symbols
        sym, :($(esc(sym)) = $(ctor)($(varname), $(map(asstokw, assumptions)...)))
    end
    syms, defs = collect(zip(symdefs...))

    # The macro returns a tuple of Symbols that were declared
    Expr(:block, defs..., :(tuple($(map(esc,syms)...))))
end

function parsedecl(expr)
    # @vars x
    if isa(expr, Symbol)
        return String(expr), expr, [], false
    # @vars x::assumptions, where assumption = assumptionkw | (assumptionkw...)
    elseif isa(expr, Expr) && expr.head == :(::)
        symexpr, assumptions = expr.args
        _, sym, _, isfun = parsedecl(symexpr)
        assumptions = isa(assumptions, Symbol) ? (assumptions,) : assumptions.args
        return String(sym), sym, assumptions, isfun
    # @vars x=>"name" 
    elseif isa(expr, Expr) && expr.head == :call && expr.args[1] == :(=>)
        length(expr.args) == 3 || parseerror()
        isa(expr.args[3], String) || parseerror()

        expr, strname = expr.args[2:end]
        _, sym, assumptions, isfun = parsedecl(expr)
        return strname, sym, assumptions, isfun
    # @vars x()
    elseif isa(expr, Expr) && expr.head == :call && expr.args[1] != :(=>)
        length(expr.args) == 1 || parseerror()
        isa(expr.args[1], Symbol) || parseerror()

        sym = expr.args[1]
        return String(sym), sym, [], true
    else
        parseerror()
    end
end

function parseerror()
    error("Incorrect @syms syntax. Try `@syms x::(real,positive)=>\"x₀\" y() z::complex n::integer` for instance.")
end

@macroexpand @syms x::(real,positive)=>"x₀", y, z::complex, n::integer
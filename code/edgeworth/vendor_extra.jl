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

macro syms(xs...)
    # If the user separates declaration with commas, the top-level expression is a tuple
    if length(xs) == 1 && isa(xs[1], Expr) && xs[1].head == :tuple
        _gensyms(xs[1].args...)
    elseif length(xs) > 0
        _gensyms(xs...)
    end
end

function _gensyms(xs...)
    asskws = asstokw(a) = Expr(:kw, esc(a), true)
    
    # Each declaration is parsed and generates a declaration using `symbols`
    symdefs = map(xs) do expr
        decl = Decl.parsedecl(expr)
        symname = Decl.sym(decl)
        symname, Decl.decl(decl)
    end
    syms, defs = collect(zip(symdefs...))

    # The macro returns a tuple of Symbols that were declared
    Expr(:block, defs..., :(tuple($(map(esc,syms)...))))
end

module Decl
    using SymPy

    # The map_subscripts function is stolen from Symbolics.jl
    const IndexMap = Dict{Char,Char}(
        '0' => '₀',
        '1' => '₁',
        '2' => '₂',
        '3' => '₃',
        '4' => '₄',
        '5' => '₅',
        '6' => '₆',
        '7' => '₇',
        '8' => '₈',
        '9' => '₉')
        
    function map_subscripts(indices)
        str = string(indices)
        join(IndexMap[c] for c in str)
    end

    abstract type VarDecl end

    struct SymDecl <: VarDecl
        sym :: Symbol
    end

    struct NamedDecl <: VarDecl
        name :: String
        rest :: VarDecl
    end

    struct FunctionDecl <: VarDecl 
        rest :: VarDecl
    end

    struct TensorDecl <: VarDecl
        ranges :: Vector{AbstractRange}
        rest :: VarDecl
    end

    struct AssumptionsDecl <: VarDecl
        assumptions :: Vector{Symbol}
        rest :: VarDecl
    end

    
    sym(x::SymDecl) = x.sym
    sym(x::NamedDecl) = sym(x.rest)
    sym(x::FunctionDecl) = sym(x.rest)
    sym(x::TensorDecl) = sym(x.rest)
    sym(x::AssumptionsDecl) = sym(x.rest)

    ctor(::SymDecl) = :symbols
    ctor(x::NamedDecl) = ctor(x.rest)
    ctor(::FunctionDecl) = :SymFunction
    ctor(x::TensorDecl) = ctor(x.rest)
    ctor(x::AssumptionsDecl) = ctor(x.rest)

    assumptions(::SymDecl) = []
    assumptions(x::NamedDecl) = assumptions(x.rest)
    assumptions(x::FunctionDecl) = assumptions(x.rest)
    assumptions(x::TensorDecl) = assumptions(x.rest)
    assumptions(x::AssumptionsDecl) = x.assumptions

    genreshape(expr, ::SymDecl) = expr
    genreshape(expr, x::NamedDecl) = genreshape(expr, x.rest)
    genreshape(expr, x::FunctionDecl) = genreshape(expr, x.rest)
    genreshape(expr, x::TensorDecl) = let 
        shape = tuple(length.(x.ranges)...)
        :(reshape(collect($(expr)), $(shape)))
    end
    genreshape(expr, x::AssumptionsDecl) = genreshape(expr, x.rest)

    # To find out the name, we need to traverse in both directions to make sure that each node can get
    # information from parents and children about possible name.
    # This is done because the expr tree will always look like NamedDecl -> ... -> TensorDecl -> ... -> SymDecl
    # and the TensorDecl node will need to know if it should create names base on a NamedDecl parent or 
    # based on the SymDecl leaf.
    name(x::SymDecl, parentname) = coalesce(parentname, String(x.sym))
    name(x::NamedDecl, parentname) = coalesce(name(x.rest, x.name), x.name)
    name(x::FunctionDecl, parentname) = name(x.rest, parentname)
    name(x::AssumptionsDecl, parentname) = name(x.rest, parentname)
    name(x::TensorDecl, parentname) = let
        basename = name(x.rest, parentname)
        # we need to double reverse the indices to make sure that we traverse them in the natural order
        namestensor = map(Iterators.product(x.ranges...)) do ind
            sub = join(map(map_subscripts, ind), "_")
            string(basename, sub)
        end
        join(namestensor[:], ", ")
    end

    function decl(x::VarDecl)
        asskws = asstokw(a) = Expr(:kw, esc(a), true)
        val = :($(ctor(x))($(name(x, missing)), $(map(asstokw, assumptions(x))...)))
        :($(esc(sym(x))) = $(genreshape(val, x)))
    end
    
    function parsedecl(expr)
        # @syms x
        if isa(expr, Symbol)
            return SymDecl(expr)
        
        # @syms x::assumptions, where assumption = assumptionkw | (assumptionkw...)
        elseif isa(expr, Expr) && expr.head == :(::)
            symexpr, assumptions = expr.args
            assumptions = isa(assumptions, Symbol) ? [assumptions] : assumptions.args
            return AssumptionsDecl(assumptions, parsedecl(symexpr))
        
        # @syms x=>"name" 
        elseif isa(expr, Expr) && expr.head == :call && expr.args[1] == :(=>)
            length(expr.args) == 3 || parseerror()
            isa(expr.args[3], String) || parseerror()

            expr, strname = expr.args[2:end]
            return NamedDecl(strname, parsedecl(expr))
        
        # @syms x()
        elseif isa(expr, Expr) && expr.head == :call && expr.args[1] != :(=>)
            length(expr.args) == 1 || parseerror()
            return FunctionDecl(parsedecl(expr.args[1]))

        # @syms x[1:5, 3:9]
        elseif isa(expr, Expr) && expr.head == :ref
            length(expr.args) > 1 || parseerror()
            ranges = map(parserange, expr.args[2:end])
            return TensorDecl(ranges, parsedecl(expr.args[1]))
        else
            parseerror()
        end
    end

    function parserange(expr)
        range = eval(expr)
        isa(range, AbstractRange) || parseerror()
        range
    end

    function parseerror()
        error("Incorrect @syms syntax. Try `@syms x::(real,positive)=>\"x₀\" y() z::complex n::integer` for instance.")
    end
    
end
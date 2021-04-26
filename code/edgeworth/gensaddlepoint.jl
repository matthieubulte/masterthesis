using NLsolve

include("./vendor_extra.jl")
include("./cgflib.jl")

function symβ̂(K)
    @vars t s
    dK = diff(K(t), t)

    try
        sol = solve(dK - s, t)
        @assert length(sol) == 1
        return lambdify(sol[1])
    catch
        return nothing
    end
end

function numβ̂(K)
    @vars t
    dK = lambdify(diff(K(t), t))
    
    function(s)
        f! = (F, x) -> F[1] = dK(x[1]) - s
        o = nlsolve(f!, [s], autodiff=:forward, method=:newton)
        if converged(o)
            o.zero[1]
        end
    end
end

function makeβ̂(K)
    β̂ = symβ̂(K)
    isnothing(β̂) ? numβ̂(K) : β̂
end

function saddlepoint(K, n)
    _β̂ = makeβ̂(K)
    ddK = ∇²(K)
    function density(s)
        β̂ = _β̂(s)
        if isnothing(β̂)
            return 0
        end
        
        w = sqrt(n)*exp(n*(K(β̂) - s*β̂))
        d = sqrt(2*pi*ddK(β̂))
        w/d
    end
end
using ReverseDiff

function dθ̂(f)
    function ff(ψₚ, λₚ, ψ̂, λ̂, n)
        partial_f = (x) -> f(ψₚ, λₚ, x[1], x[2:end], n)
        j = ReverseDiff.jacobian(partial_f, [ψ̂; λ̂])
        transpose(j)
    end
    return ff
end

function dθ(f)
    function ff(ψₚ, λₚ, ψ̂, λ̂, n)
        partial_f = (x) -> f(x[1], x[2:end], ψ̂, λ̂, n)
        j = ReverseDiff.jacobian(partial_f, [ψₚ; λₚ])
        transpose(j)
    end
    return ff
end

function dψ(f)
    function ff(ψₚ, λₚ, ψ̂, λ̂, n)
        partial_f = (x) -> f(x[1], λₚ, ψ̂, λ̂, n)
        j = ReverseDiff.jacobian(partial_f, [ψₚ])
        transpose(j)
    end
    return ff
end

function dψ̂(f)
    function ff(ψₚ, λₚ, ψ̂, λ̂, n)
        partial_f = (x) -> f(ψₚ, λₚ, x, λ̂, n)
        j = ReverseDiff.jacobian(partial_f, [ψ̂])
        transpose(j)
    end
    return ff
end


function dλ(f)
    function ff(ψₚ, λₚ, ψ̂, λ̂, n)
        partial_f = (x) -> f(ψₚ, x, ψ̂, λ̂, n)
        j = ReverseDiff.jacobian(partial_f, λₚ)
        transpose(j)
    end
    return ff
end


function dλ̂(f)
    function ff(ψₚ, λₚ, ψ̂, λ̂, n)
        partial_f = (x) -> f(ψₚ, λₚ, ψ̂, x, n)
        j = ReverseDiff.jacobian(partial_f, λ̂)
        transpose(j)
    end
    return ff
end

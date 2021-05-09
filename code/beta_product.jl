using Distributions

function ecdf(data)
    sorted = sort(data)
    n = size(sorted)[1]
    (x) -> begin
        for i = 1:n
            val = @inbounds sorted[i]
            if val >= x
                return i/n
            end
        end
        1.0
    end
end

function construct_cdf(params, n)
    result = ones(n)
    for (αᵢ, βᵢ) = params
        result .*= rand(Beta(αᵢ, βᵢ), n)
    end
    ecdf(result)
end

function removeedge!(E, e)
    filter!(ee -> ee != e, E)
end

function neighbours(E, i)
    nei = []
    for e = E
        if i == e[1]
            push!(nei, e[2])
        end
        if i == e[2]
            push!(nei, e[1])
        end
    end
    Set(nei)
end

function cyclebetaparams(p)
    edges = [ [i, j] for i = 1:p for j = (i+1):p ]
    params = []
    for i = 1:p
        for j = (i+2):p
            if i == 1 && j == p
                continue
            end
            removeedge!(edges, [i, j])
            C = setdiff(intersect(
                neighbours(edges, i),
                neighbours(edges, j)
            ), Set([i j]))
            push!(params, length(C))
        end
    end
    return params
end
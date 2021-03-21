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
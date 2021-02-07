using LinearAlgebra

function eye(p)
    Matrix{Float64}(I, p, p)
end

function vecToSymm(vec::Vector{T}) where {T <: Number}
    veclen = size(vec)[1]
    p = Int((sqrt(1 + 8*veclen)-1)/2)
    A = zeros(T, p, p)
    vstart = 1
    for i = 1:p
        ilen = p - i
        @inbounds A[i, i:end] = vec[vstart:vstart+ilen]
        vstart += ilen + 1
    end
    Symmetric(A)
end

function symmToVec(A::Matrix{T}) where {T <: Number}
    p = size(A)[1]
    veclen = Int(p*(p+1)/2)
    vec = zeros(veclen)
    vstart = 1
    for i = 1:p
        ilen = p - i + 1
        @inbounds vec[vstart:vstart+ilen-1] = A[i, i:end]
        vstart += ilen
    end
    vec
end

function vecIdxToMatIdx(idx, veclen)
    # size of top-level triangle
    p = Int((sqrt(1 + 8*veclen)-1)/2)
    # size of triangle starting at the row of current index
    pm1 = Int(ceil((sqrt(1 + 8*(veclen-idx+1))-1)/2))
    # vec idx of element at start of row
    idxStart = veclen + 1 - Int(pm1*(pm1+1)/2)

    i = p + 1 - pm1
    i, idx - idxStart + i
end
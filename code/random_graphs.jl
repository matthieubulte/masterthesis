
# A simple chain, this can be written as a chain of regressions
# from each node on the previous one.
# See Lecture on Algebraic Statistics, eq 3.3.7 for the construction
# of the Cholesky factorization of the precision.
function chain(p)
    cliques = []
    Λ = zeros(Float64, p, p)
    Ω = Diagonal(rand(Chisq(1), p))
    for i = 1:(p-1)
        push!(cliques, [i, i+1])
        @inbounds Λ[i, i+1] = rand(Normal())
        @inbounds Λ[i, i] = 1
    end
    @inbounds Λ[p, p] = 1
    U = Λ*sqrt(Ω)
    K = U*U'
    
    cliques, PDMat(K, Cholesky(U, :U, 0))
end

# Generate a cycle of length p
function cycle(p)
    d = TruncatedNormal(0, 1, -0.5, 0.5)
    cliques = []
    K = zeros(Float64, p, p)
    
    for i = 1:(p-1)
        push!(cliques, [i, i+1])
        @inbounds K[i, i+1] = rand(d)
        @inbounds K[i+1, i] = K[i, i+1]
        @inbounds K[i, i] = 1
    end
    @inbounds K[p, p] = 1
    @inbounds K[1, p] = rand(d)
    @inbounds K[p, 1] = K[1, p]
    push!(cliques, [1, p])
    
    if p < 4
        cliques = [collect(1:p)]
    end

    cliques, PDMat(K)
end

# The graph is based on the DAG 1 -> 2 -> ... -> p <- 1
# the moral graph from this DAG will then be a cycle with 
# a chord between 1 and p-1.
# Again, see Lecture on Algebraic Statistics, eq 3.3.7 for the construction
# of the Cholesky factorization of the precision.
function cycleWithOneChord(p)
    cliques = []
    Λ = zeros(Float64, p, p)
    Ω = Diagonal(rand(Chisq(1), p))
    for i = 1:(p-1)
        push!(cliques, [i, i+1])
        @inbounds Λ[i, i+1] = rand(Normal())
        @inbounds Λ[i, i] = 1
    end
    @inbounds Λ[p, p] = 1
    @inbounds Λ[1, p] = rand(Normal())
    U = Λ*sqrt(Ω)
    K = U*U'

    # the last clique that was added is the edge [p-1, p]
    # but because of the chord added between 1 and p-1, the
    # last clique is now a triangle containing 1, p-1, p
    pop!(cliques) 
    push!(cliques, [1, p-1, p])

    # in the special case were p=4, adding the chord means
    # that the graph decomposes in two triangular cliques
    #    2
    #  /   \
    # 1 --- 3
    #  \   /
    #    4
    if p == 4
        cliques = [ [1;2;3], [1;3;4] ]
    end

    cliques, PDMat(K, Cholesky(U, :U, 0))
end



function cycleWithOneChordAndNoTriade(p)
    @assert p > 6
    
    d = TruncatedNormal(0, 1, -0.5, 0.5)
    cliques = []
    K = zeros(Float64, p, p)
    
    for i = 1:(p-1)
        push!(cliques, [i, i+1])
        @inbounds K[i, i+1] = rand(d)
        @inbounds K[i+1, i] = K[i, i+1]
        @inbounds K[i, i] = 1
    end
    @inbounds K[p, p] = 1
    
    @inbounds K[1, p] = rand(d)
    @inbounds K[p, 1] = K[1, p]
    push!(cliques, [1, p])

    i = Int(floor(p/2))
    @inbounds K[1, i] = rand(d)
    @inbounds K[i, 1] = K[1, i]
    push!(cliques, [1, i])

    cliques, PDMat(K)
end

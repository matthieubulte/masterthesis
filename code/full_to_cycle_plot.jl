

using Plots, StatsPlots, JLD, Distributions, HypothesisTests, Dates, ProgressMeter

include("beta_product.jl")

test_level = 0.05
ps = [5; 10; 20; 50; 75; 90]
sims=250
repls=100
n=100

pvalues = zeros(Float32, size(ps)[1], repls);
pvalues_w = zeros(Float32, size(ps)[1], repls);


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


# p = 5
p = 5;
results = load("./202105072217_output/results_$(p).jld")["data"];

beta_params = [((n - C - 1)/2, 1/2) for C = cyclebetaparams(p)]
beta_prod_cdf = construct_cdf(beta_params, 100_000);

p1 = beta_prod_cdf.(results[1,:].^(2/n));
p2 = pdf(Chisq(p*(p-1)/2 - p), -2*log.(results[1,:]));

l = 1; for p = ps
    Cs = cyclebetaparams(p)
    beta_params = [((n - C - 1)/2, 1/2) for C = Cs]
    beta_prod_cdf = construct_cdf(beta_params, 100_000);

    results = load("./202105072217_output/results_$(p).jld")["data"];
    pvalues[l, :] = [
        pvalue(ExactOneSampleKSTest(
            beta_prod_cdf.(results[i,:].^(2/n)),
            Uniform()))
        for i = 1:repls
    ];
    pvalues_w[l, :] = [
        pvalue(ExactOneSampleKSTest(
            -2*log.(results[i,:]),
            Chisq(p*(p-1)/2 - p)))
        for i = 1:repls
    ]
    l += 1
end


function plots(pvals, ps, title, filename)
    labels = ["$(p)" for p = ps]
    
    p = boxplot(pvalues[1, :], legend=false)
    
    for i = 2:size(ps)[1]
        p = boxplot!(pvals[i, :], legend=false)
    end

    title!(title)
    xticks!(1:size(ps)[1], labels)
    xlabel!("Dimension p")
    ylabel!("Average power")
    
    now = Dates.format(Dates.now(), "yyyymmddHHMM")
    # savefig(p, "./plots/$(now)_$(filename)")
    
    p
end

plots(pvalues_w, ps, "p-cycle vs Complete, χ²", "chisq_based_test")


plots(pvalues, ps, "p-cycle vs Complete, Beta product", "beta_prod_based_test")


# p=5
# beta_params = ( (n - (p - 2) - 1)/2, 1/2 )
# beta_prod_cdf = construct_cdf([beta_params, beta_params], 100_000);
# results = load("./202105031741_output/results_$(p).jld")["data"];

# a = [
#     pvalue(ExactOneSampleKSTest(
#         beta_prod_cdf.(results[i,:].^(2/n)),
#         Uniform()))
#     for i = 1:repls
# ];

# a

# boxplot(a)
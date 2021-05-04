

using Plots, StatsPlots, JLD, Distributions, HypothesisTests, Dates, ProgressMeter

include("beta_product.jl")

test_level = 0.05
ps = [5; 10; 20; 50; 75; 90]
sims=250
repls=100
n=100

pvalues = zeros(Float32, size(ps)[1], repls);
pvalues_w = zeros(Float32, size(ps)[1], repls);

l = 1; for p = ps
    beta_params = ( (n - (p - 2) - 1)/2, 1/2 )
    beta_prod_cdf = construct_cdf([beta_params, beta_params], 100_000);

    results = load("./202105031741_output/results_$(p).jld")["data"];



    pvalues[l, :] = [
        mean(beta_prod_cdf.(results[i,:].^(2/n)) .< test_level)
        for i = 1:repls
    ];

    
    pvalues_w[l, :] = [
        mean(cdf(Chisq(2), -2*log.(results[i,:])) .< test_level)
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
    savefig(p, "./plots/$(now)_$(filename)")
    
    p
end

plots(pvalues_w, ps, "Complete with Chordless cycle vs Complete, χ²", "chisq_based_test");


plots(pvalues, ps, "Complete with Chordless cycle vs Complete, Beta product", "beta_prod_based_test");


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
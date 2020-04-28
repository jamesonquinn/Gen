# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: jl:light,ipynb
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Julia 1.4.0
#     language: julia
#     name: julia-1.4
# ---

# This cell may take a few seconds to run.
using Gen
using PyPlot
using StatsBase

# +
second(x) = x[2] #used to pull logweight from (trace, logweight)

function AIDE_compare(f, g, #generative functions to compare. Should return the important value.
        input, #any input to generative functions; must be same for both
        ret_to_f, ret_to_g, #take a return value from either generative, and produce constraints for one generative
        Nf, Ng, #Number of traces to draw from each
        Mf, Mg  #Number of runs of each to use to estimate score
    )
    cm = choicemap()
    f_vals = [Gen.get_retval(generate(f, input, cm)[1]) for i in 1:Nf]
    g_vals = [Gen.get_retval(generate(g, input, cm)[1]) for i in 1:Ng]
    
    f_scores_of_g = [mean(
                        [second(generate(f, input, ret_to_f(trace))) 
                            for j in 1:Mf]) 
                    for trace in g_vals]
    g_scores_of_f = [mean(
                        [second(generate(g, input, ret_to_g(trace)))
                            for j in 1:Mf])
                    for trace in f_vals]
    f_scores_of_f = [mean(
                        [second(generate(f, input, ret_to_f(trace)))
                            for j in 1:Mf]) 
                    for trace in f_vals]
    g_scores_of_g = [mean(
                        [second(generate(g, input, ret_to_g(trace)))
                            for j in 1:Mf]) 
                    for trace in g_vals]
    
    KLfg = mean(f_scores_of_f) - mean(g_scores_of_f)
    KLgf = mean(g_scores_of_g) - mean(f_scores_of_g)
    
    KLfg + KLgf
end


# +
@gen function binom_prior(α, β)
    p ~ beta(α, β)
    p
end

@gen function basic_binom(n, α, β)
    p ~ beta(α, β)
    h ~ binom(n, p)
end

function importance_cond_binom_for(samps)
    @gen function i_c_b(h, n, α = 1., β = 1.)
        #@warn "i_c_b" samps h n α
        constraint = choicemap((:h, h))
        logweights = Vector{Float64}(undef, samps)
        ps = Vector{Float64}(undef, samps)
        for s in 1:samps
            p = @trace(binom_prior(α, β),:data => s)
            constraint[:p] = p
            ps[s] = p
            (_, logweights[s]) = generate(basic_binom, (n, α, β), constraint)
                    #in this simple case, this is just a more-complicated way of calling
                        #Distributions.Binomial(...).logpdf. However, this pattern generalizes
                        #better to arbitrary generative functions.
        end
        logweights .-= max(logweights...)
        weights = exp.(logweights)
        weights ./= sum(weights)
        whichOne ~ categorical(weights)
        p = ps[whichOne]
        p
    end
end

@gen function analytic_cond_binom(h, n, α = 1., β = 1.)
    p ~ beta(α + h, β + n - h)
    p
end

function ret_to_importance_constraint(p)
    constraint = choicemap((:whichOne, 1))
    constraint[:data => 1 => :p] = p
    constraint
end

function ret_to_analytic_constraint(p)
    choicemap((:p, p))
end

        
        
# -

importance_cond_binom_for(5)(0,10)

# +
ns = 5
ms = 5

[AIDE_compare(importance_cond_binom_for(i), analytic_cond_binom, #generative functions to compare. Should return the important value.
        (0,10), #any input to generative functions; must be same for both
        ret_to_importance_constraint, ret_to_analytic_constraint, #take a return value from either generative, and produce constraints for one generative
        ns, ns, #Number of traces to draw from each
        ms, ms  #Number of runs of each to use to estimate score
    ) for i in 1:5]

# +
ns = 50
ms = 50

[AIDE_compare(importance_cond_binom_for(i), analytic_cond_binom, #generative functions to compare. Should return the important value.
        (0,10), #any input to generative functions; must be same for both
        ret_to_importance_constraint, ret_to_analytic_constraint, #take a return value from either generative, and produce constraints for one generative
        ns, ns, #Number of traces to draw from each
        ms, ms  #Number of runs of each to use to estimate score
    ) for i in 1:5]
# -





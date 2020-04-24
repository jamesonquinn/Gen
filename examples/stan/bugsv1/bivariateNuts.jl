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

# +
using Gen
using Statistics
include("DistributionsBacked.jl")
using AdvancedHMC

const my_normal = DistributionsBacked{Float64}((mu, sigma) -> 
                            Distributions.Normal(mu, sigma), [true, true], true)
const my_unif = DistributionsBacked{Float64}((lo, hi) -> 
                            Distributions.Uniform(lo, hi), [true, true], true)
;

# +
@gen function corbiv_model() #correlated bivariate normal
    ρ ~ my_unif(-1., 1.)
    x ~ my_normal(0., 1.)
    y ~ my_normal(ρ*x, sqrt(1. - ρ^2))
end

@gen function cormiv_model() #correlated bivariate normal
    ρ ~ my_unif(0., 1.) #correlation between x and y
    ρ2 ~ my_unif(0., ρ) #correlation between z and (x or y) 
    zy_cond_zx = (1-ρ2^2)*ρ2
    x ~ my_normal(0., 1.)
    y ~ my_normal(ρ*x, sqrt(1. - ρ^2))
    z ~ my_normal(ρ2*x + zy_cond_zx*y, sqrt(1. - ρ2^2 - zy_cond_zx^2))
end
;

# +
function constrainρ(ρ::Float64)
    constraints = Gen.choicemap()
    constraints[:ρ] = ρ
    constraints
end

function constrainρ2(ρ::Float64, ρ2, z)
    constraints = Gen.choicemap()
    constraints[:ρ] = ρ
    constraints[:ρ2] = ρ2
    constraints[:z] = z
    
    
    constraints
end

function mcmc_inference(ρ, num_iters, update, selection)
    observation = constrainρ(ρ)
    (trace, _) = generate(corbiv_model, (), observation)
    samples = Array{Float64}(undef,num_iters,2)
    for i=1:num_iters
        trace = update(trace, selection)
        ch = get_choices(trace)
        samples[i,1] = ch[:x]
        samples[i,2] = ch[:y]
    end
    samples
end

function mcmc_m_inference(ρ, ρ2, z, num_iters, update)
    observation = constrainρ2(ρ, ρ2, z)
    (trace, _) = generate(cormiv_model, (), observation)
    samples = Array{Float64}(undef,num_iters,2)
    for i=1:num_iters
        trace = update(trace)
        ch = get_choices(trace)
        samples[i,1] = ch[:x]
        samples[i,2] = ch[:y]
    end
    samples
end

function block_mh(tr, selection)
    (tr, _) = mh(tr, select(:x, :y))
    tr
end


function simple_hmc(tr, selection)
    (tr, _) = hmc(tr, select(:x, :y))
    tr
end


;
# -

iters = 10_000
show = 5
ρ = -.5
samps = mcmc_inference(ρ, iters, block_mh, select(:x,:y))
samps[(iters-show+1):iters,:]



iters = 100
show = 5
ρ = .8
samps = mcmc_inference(ρ, iters, simple_hmc)
samps[(iters-show+1):iters,:]

println(mean(samps))
println(cor(samps[:,1],samps[:,2]))


# Disable AdvancedHMC's NUTS logging

# +
using Logging
using LoggingExtras

function ignore_sampling_filter(log_args)
    !(occursin("sampling steps",log_args.message) || occursin("adapation steps",log_args.message))
end
logger = ActiveFilteredLogger(ignore_sampling_filter, global_logger())


if !(@isdefined old_logger) #do this only once
    old_logger = global_logger(logger)
end
# -

function my_nuts(trace, selection, 
        n_postadapt_steps = 2,
        n_adapts = 1,
        initial_ϵ_reduce_fac = 10)
    
    n_NUTS_steps = n_postadapt_steps + n_adapts
    
    filtered_choices = get_selected(get_choices(trace), selection)
    cur_xy = to_array(filtered_choices, Float64)
    dimension = length(cur_xy)
    metric = DiagEuclideanMetric(dimension)
    
    retval_grad = nothing #accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
    
    function update_xy(val)
        extra_constraints = from_array(filtered_choices, val)
        update(trace, (), (NoChange(),), extra_constraints)
    end
    
    function val_to_lp_plus_c(val)
        (new_trace, weight, discard, retdiff) = update_xy(val)
        weight
    end
    
    function val_to_grad(val)
        (new_trace, weight, discard, retdiff) = update_xy(val)
        (retval_grad_out, values_trie, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
        grad = [gradient_trie[:x], gradient_trie[:y]]
        (weight, grad)
    end
    
    # Define a Hamiltonian system, using metric defined globally above
    hamiltonian = Hamiltonian(metric, val_to_lp_plus_c, val_to_grad)
    
    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, cur_xy) ./ initial_ϵ_reduce_fac
    integrator = Leapfrog(initial_ϵ)
    
    # Define an HMC sampler, with the following components
    #   - multinomial sampling scheme,
    #   - generalised No-U-Turn criteria, and
    #   - windowed adaption for step-size and diagonal mass matrix
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    
    samples, stats = sample(hamiltonian, proposal, cur_xy, n_NUTS_steps, adaptor, n_adapts; progress=false)
    
    #println(samples[3])
    
    
    (new_trace, weight, discard, retdiff) = update_xy(samples[n_NUTS_steps])
    new_trace
end


# +
iters = 200
show = 5
ρ = .99


samps = mcmc_inference(ρ, iters, my_nuts, select(:x,:y))
samps[(iters-show+1):iters,:]
# -

println(cor(samps[1:iters-1,1],samps[2:iters,1])) #serial correlation; lower is better
println(ρ^4) #for comparison, gibbs would be ρ² for each step; ρ⁴ for two steps

positive



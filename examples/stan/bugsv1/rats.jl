# -*- coding: utf-8 -*-
# Load necessary libraries

# +
using Gen
using Statistics
using ProfileView #Make sure this is loaded before PyPlot to avoid stupid gtk version issues
using Profile
using PyPlot

include("DistributionsBacked.jl")
include("IterDeep.jl")
include("AnimatedPyplot.jl")
include("nuts.jl")

include("unrealizedNormals.jl")

const my_inv_gamma = DistributionsBacked{Float64}((alpha, theta) -> Distributions.InverseGamma(alpha, theta), [true, true], true)
# -
# Space for declaring constants.

disable_sample_logging()
INV_GAMMA_PRIOR_CONSTANT = 0.1;


# Canonical Rats data, from from section 6 of Gelfand et al (1990). See http://www.openbugs.net/Examples/Rats.html

ys_raw = ([151, 145, 147, 155, 135, 159, 141, 159, 177, 134,
    160, 143, 154, 171, 163, 160, 142, 156, 157, 152, 154, 139, 146,
    157, 132, 160, 169, 157, 137, 153, 199, 199, 214, 200, 188, 210,
    189, 201, 236, 182, 208, 188, 200, 221, 216, 207, 187, 203, 212,
    203, 205, 190, 191, 211, 185, 207, 216, 205, 180, 200, 246, 249,
    263, 237, 230, 252, 231, 248, 285, 220, 261, 220, 244, 270, 242,
    248, 234, 243, 259, 246, 253, 225, 229, 250, 237, 257, 261, 248,
    219, 244, 283, 293, 312, 272, 280, 298, 275, 297, 350, 260, 313,
    273, 289, 326, 281, 288, 280, 283, 307, 286, 298, 267, 272, 285,
    286, 303, 295, 289, 258, 286, 320, 354, 328, 297, 323, 331, 305,
    338, 376, 296, 352, 314, 325, 358, 312, 324, 316, 317, 336, 321,
    334, 302, 302, 323, 331, 345, 333, 316, 291, 324])
ys = reshape([Float64(y) for y = ys_raw],30,5)
xs = [8.0, 15.0, 22.0, 29.0, 36.0]
ys = ys ./ 100.
(N,T) = size(ys)

# These data are the weights of 30 rats, measured at 5 common age values in days. The model is a simple hierarchical/random effects one; a random intercept and random slope for each rat, plus an error term (presumably representing not measurement error, but just the random variation of individual growth curves).

for i in 1:30
    plot(xs,ys[i,:])
end
xlabel("Age (days)")
ylabel("Weight (g)")
title("Rat growth")




# +
@gen function oneRat(xs, T, mu_alpha, sigmasq_alpha, mu_beta, sigmasq_beta, sigmasq_y)

    alpha ~ normal(mu_alpha, sqrt(sigmasq_alpha))
    beta ~ normal(mu_beta, sqrt(sigmasq_beta))
    y ~ broadcasted_normal([alpha + beta * (xs[t]) for t = 1:T],
                                        sqrt(sigmasq_y))
end

@gen function ratsModel(xs::Vector,N::Int32,)
    T = length(xs)
    xbar = mean(xs) #could be precomputed, but YKWTS about premature optimization...

    mu_alpha ~ normal(0, 10.0)
    mu_beta ~ normal(0, 1.0)
    sigmasq_y ~ my_inv_gamma(INV_GAMMA_PRIOR_CONSTANT, INV_GAMMA_PRIOR_CONSTANT)
    sigmasq_alpha ~ my_inv_gamma(INV_GAMMA_PRIOR_CONSTANT, INV_GAMMA_PRIOR_CONSTANT)
    sigmasq_beta ~ my_inv_gamma(INV_GAMMA_PRIOR_CONSTANT, INV_GAMMA_PRIOR_CONSTANT)

    alpha = Vector{Float64}(undef,N)
    beta = Vector{Float64}(undef,N)
    y = Vector{Vector{Float64}}(undef,N)
    for n in 1:N
        {:data => n} ~ oneRat(xs .- xbar, T, mu_alpha, sigmasq_alpha, mu_beta, sigmasq_beta, sigmasq_y)
    end
end

model_transformations = choicemap()
model_transformations[:sigmasq_y] = transform_log
model_transformations[:sigmasq_alpha] = transform_log
model_transformations[:sigmasq_beta] = transform_log

;
# -

function make_constraints(ys::Array)
    (N,T) = size(ys)
    constraints = Gen.choicemap()
    for n in 1:N
        constraints[:data => n => :y] = ys[n,:]
    end
    constraints
end
;

function mcmc_inference(xs, ys, num_iters, update, model = ratsModel)
    results = Vector{Gen.DynamicDSLTrace{DynamicDSLFunction{Any}}}(undef, num_iters+1)
    N = size(ys)[1]
    observations = make_constraints(ys)
    (trace, _) = generate(model, (xs, N), observations)
    results[1] = trace
    selection = Gen.complement(select(observations))
    for iter=1:num_iters
        trace = update(trace,N,selection)
        results[iter+1] = trace
    end
    results
end
;

# +

        
        
@gen function oneDeferredRat(xs, T, mu_alpha, sigmasq_alpha, mu_beta, sigmasq_beta, sigmasq_y)

    alpha ~ unrealizedIndyNormal(mu_alpha, sqrt(sigmasq_alpha))
    beta ~ unrealizedIndyNormal(mu_beta, sqrt(sigmasq_beta))
    for t = 1:T
        @trace(observedNormal(alpha + beta * (xs[t]), sqrt(sigmasq_y)), "y$(t)")
    end
end


@gen function deferredRatsModel(xs::Vector,N::Int32,)
    T = length(xs)
    xbar = mean(xs) #could be precomputed, but YKWTS about premature optimization...

    mu_alpha ~ normal(0, 10.0)
    mu_beta ~ normal(0, 1.0)
    sigmasq_y ~ my_inv_gamma(INV_GAMMA_PRIOR_CONSTANT, INV_GAMMA_PRIOR_CONSTANT)
    sigmasq_alpha ~ my_inv_gamma(INV_GAMMA_PRIOR_CONSTANT, INV_GAMMA_PRIOR_CONSTANT)
    sigmasq_beta ~ my_inv_gamma(INV_GAMMA_PRIOR_CONSTANT, INV_GAMMA_PRIOR_CONSTANT)

    alpha = Vector{Float64}(undef,N)
    beta = Vector{Float64}(undef,N)
    y = Vector{Vector{Float64}}(undef,N)
    for n in 1:N
        {:data => n} ~ oneDeferredRat(xs .- xbar, T, mu_alpha, sigmasq_alpha, mu_beta, sigmasq_beta, sigmasq_y)
    end
end

model_transformations = choicemap()
model_transformations[:sigmasq_y] = transform_log
model_transformations[:sigmasq_alpha] = transform_log
model_transformations[:sigmasq_beta] = transform_log

;

# +
Profile.clear()

function block_mh_d(tr,N,selection)
    (tr, _) = mh(tr, select(:mu_alpha, :mu_beta))
    (tr, _) = mh(tr, select(:sigmasq_alpha, :sigmasq_beta, :sigmasq_y))

    for n in 1:N
        (tr, _) = mh(tr, select(:data => n => :y1,:data => n => :y2,:data => n => :y3,:data => n => :y4,
                                :data => n => :alpha,:data => n => :beta))
    end
    tr
end
;

VIZ_INTERVAL = 1
VIZ_FRAMES = 20
trs3= Profile.@profile mcmc_inference(xs, ys, VIZ_INTERVAL * VIZ_FRAMES, block_mh_d, deferredRatsModel)
tr3 = trs3[end]
print("mu_alpha: $(tr3[:mu_alpha]), mu_beta: $(tr3[:mu_beta]), sigma_y: $(sqrt(tr3[:sigmasq_y]))\n")
;


# +
c = make_constraints(ys)
alphahats = mean(ys,dims=2)
alphahat = mean(alphahats)
betahats = transpose(ys)\xs
betahat = mean(betahats)
c[:mu_alpha] = alphahat
c[:mu_beta] = betahat

c[:sigmasq_beta] = var(betahats) * 2.1601
c[:sigmasq_alpha] = var(alphahats) * 2.1601
c[:sigmasq_y] = 0.000001

tr, w = generate(deferredRatsModel, (ys, N), c) #w == -N/2

s = get_score(tr)

c[:sigmasq_y] = 1000.

c[:sigmasq_beta] = var(betahats) * 2.1603
c[:sigmasq_alpha] = var(alphahats) * 2.1603
c[:sigmasq_y] = 0.000001

tr2, w2 = generate(deferredRatsModel, (ys, N), c) #w == -N/2

s2 = get_score(tr2)

w, s, w2, s2
# -

#
# ...
#
# ...
#
# ...
#
# ...
#
# ...
#
# ...
#
# ...
#
# ...
#
# ...
#
# ...
#
# ...
#
#
#
#
#
#
#
#

function block_mh(tr,N,selection)
    (tr, _) = mh(tr, select(:mu_alpha, :mu_beta))
    (tr, _) = mh(tr, select(:sigmasq_alpha, :sigmasq_beta, :sigmasq_y))

    for n in 1:N
        (tr, _) = mh(tr, select(:data => n => :y,
                                :data => n => :alpha,:data => n => :beta))
    end
    tr
end
;


function simple_hmc(tr,N,selection)
    (tr, _) = hmc(tr, selection)
    tr
end
;

VIZ_INTERVAL = 2
VIZ_FRAMES = 10
trs= mcmc_inference(xs, ys, VIZ_INTERVAL * VIZ_FRAMES, block_mh)
tr = trs[end]
get_choices(tr)
;

print("mu_alpha: $(tr[:mu_alpha]), mu_beta: $(tr[:mu_beta]), sigma_y: $(sqrt(tr[:sigmasq_y]))")


# +
fewcolors = ["r", "b", "g"] #In order to avoid busy graphs, I graph just 3 rats per plot
numPlots = 2 #...but tile 2 plots
numRats = length(fewcolors)
CONF95SIGS = 1.96

function make_axes()
    fig = figure("A few rats",figsize=(5,5))
    my_axes = fig.subplots(nrows=numPlots)
    for ax in my_axes
        ax.set_xlim((5,40))
    end
    (fig,my_axes)
end

function visualize_rats(trace,my_axes)
    xbar = mean(xs)
    xgrid = collect(min(xs...):.5:max(xs...))
    xlims = [min(xs...), max(xs...)]
    centeredgrid = xgrid .- xbar
    linesigmas = sqrt.(trace[:sigmasq_alpha] .+ trace[:sigmasq_beta] .* centeredgrid .^ 2)
    meanline = trace[:mu_alpha] .+ trace[:mu_beta] .* centeredgrid
    upperline = meanline .+ CONF95SIGS .* linesigmas
    lowerline = meanline .- CONF95SIGS .* linesigmas
    for (j,ax) in enumerate(my_axes)
        ax.clear()
        ax.fill_between(xgrid,lowerline,upperline,alpha=.1,color="black")
        ax.fill_between([6.,7],
                        fill(250. + CONF95SIGS * sqrt(trace[:sigmasq_y]),2),
                        fill(250. - CONF95SIGS * sqrt(trace[:sigmasq_y]),2),
            alpha=1,color="black")

        ax.set_ylim((min(1.00,lowerline[1],lowerline[end]),
                max(4.00,upperline[1],upperline[end])))
        for k in 1:numRats
            i = j*numRats + k
            ax.plot(xs,ys[i,:],color=fewcolors[k], alpha=1.)

            ratline = trace[:data => i => :alpha] .+ trace[:data => i => :beta] .* (xlims .- xbar)
            ax.plot(xlims, ratline,color=fewcolors[k], alpha=.3)
        end
        ax.set_ylabel("Weight (g)")
        if j == 1
            ax.set_title("Rat growth")
        else
            ax.set_xlabel("Age (days)")
        end

    end
end



# +

visualize_rats(tr,make_axes()[2])
# -

display_animation("rats_mh", make_axes,
    (ax)->visualize_rats(trs[1],ax),
    (ax,i)->visualize_rats(trs[1+i*VIZ_INTERVAL],ax),
    VIZ_FRAMES)




VIZ_INTERVAL = 2
VIZ_FRAMES = 10
trs2= mcmc_inference(xs, ys, VIZ_INTERVAL * VIZ_FRAMES, simple_hmc)
tr2 = trs2[end]
get_choices(tr2)
;

print("mu_alpha: $(tr2[:mu_alpha]), mu_beta: $(tr2[:mu_beta]), sigma_y: $(sqrt(tr2[:sigmasq_y]))\n")

if false
    display_animation("rats_hmc", make_axes,
        (ax)->visualize_rats(trs[1],ax),
        (ax,i)->visualize_rats(trs[1+i*VIZ_INTERVAL],ax),
        VIZ_FRAMES)
end


# +
Profile.clear()


VIZ_INTERVAL = 1
VIZ_FRAMES = 20
trs3= Profile.@profile mcmc_inference(xs, ys, VIZ_INTERVAL * VIZ_FRAMES, (tr, N, selection)->my_nuts(tr, selection, model_transformations))
tr3 = trs3[end]
print("mu_alpha: $(tr3[:mu_alpha]), mu_beta: $(tr3[:mu_beta]), sigma_y: $(sqrt(tr3[:sigmasq_y]))\n")
;
# -


ProfileView.view()

open("profile.txt", "w") do s
    Profile.print(IOContext(s, :displaysize => (24, 500)))
end


display_animation("rats_nuts", make_axes,
    (ax)->visualize_rats(trs3[1],ax),
    (ax,i)->visualize_rats(trs3[1+i*VIZ_INTERVAL],ax),
    VIZ_FRAMES,200)

# The below is just a sketch of how I think "semi-automatic conjugacy" might work. It's not working code, yet.

l = [(4,6)]
for (a, (b,c)) in enumerate(l)
    @warn "hi" a b c
end



using Gen
using Statistics



@gen function model_prematurely_optimized(xs::Vector,N::Int32,) #single objects across rats (the dim of size N)
    T = length(xs)
    xbar = mean(xs) #could be precomputed, but YKWTS about premature optimization...

    mu_alpha ~ normal(0, 100)
    mu_beta ~ normal(0, 100)
    sigmasq_y ~ inv_gamma(0.01, 0.01)
    sigmasq_alpha ~ inv_gamma(0.01, 0.01)
    sigmasq_beta ~ inv_gamma(0.01, 0.01)

    alpha ~ normal(fill(mu_alpha,N), sqrt(sigmasq_alpha)) # vectorized
    beta ~ normal(fill(mu_beta,N), sqrt(sigmasq_beta))  # vectorized
    ys ~ normal([alpha[n] + beta[n] * (x[t] - xbar) for n in 1:N, t in 1:T], sqrt(sigmasq_y))
end

@gen function model(xs::Vector,N::Int32,)
    T = length(xs)
    xbar = mean(xs) #could be precomputed, but YKWTS about premature optimization...

    mu_alpha ~ normal(0, 100)
    mu_beta ~ normal(0, 10)
    sigmasq_y ~ inv_gamma(0.01, 0.01)
    sigmasq_alpha ~ inv_gamma(0.01, 0.01)
    sigmasq_beta ~ inv_gamma(0.01, 0.01)

    alpha = Vector{Float64}(undef,N)
    beta = Vector{Float64}(undef,N)
    y = Vector{Vector{Float64}}(undef,N)
    for n in 1:N
        alpha[n] = ({:data => n => :alpha} ~ normal(mu_alpha, sqrt(sigmasq_alpha))) # vectorized
        beta[n] = ({:data => n => :beta} ~ normal(mu_beta, sqrt(sigmasq_beta)))  # vectorized
        y[n] = ({:data => n => :y} ~ broadcasted_normal([alpha[n] + beta[n] * (xs[t] - xbar) for t = 1:T],
                                            sqrt(sigmasq_y)))
    end
end

function make_constraints(ys::Array)
    (N,T) = size(ys)
    constraints = Gen.choicemap()
    for n in 1:N
        constraints[:data => n => :y] = ys[n]
    constraints
end

function mcmc_inference(xs, ys, num_iters, update)
    N = size(ys)[1]
    observations = make_constraints(ys)
    (trace, _) = generate(model, (xs, N), observations)
    for iter=1:num_iters
        trace = update(trace,N)
    end
    trace
end

function block_mh(tr,N)
    (tr, _) = mh(tr, select(:mu_alpha, :mu_beta))
    (tr, _) = mh(tr, select(:sigmasq_alpha, :sigmasq_beta, :sigmasq_y))

    for n in 1:N
        (tr, _) = mh(tr, select(:data => n => :y,
                                :data => n => :alpha,:data => n => :beta))
    end
    tr
end

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
tr = mcmc_inference(xs, ys, 200, block_mh)
get_choices(tr)

print("mu_alpha: $(tr[:mu_alpha]), mu_beta: $(tr[:mu_beta]), sigma_y: $(sqrt(tr[:sigmasq_y]))")

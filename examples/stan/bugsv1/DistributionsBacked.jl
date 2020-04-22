using Distributions
using DistributionsAD
using Zygote
using Gen
struct DistributionsBacked{T} <: Gen.Distribution{T}
    to_dist
    has_arg_grads :: Vector{Bool}
    has_out_grad :: Bool
end

Gen.random(d::DistributionsBacked, args...) = rand(d.to_dist(args...))
Gen.logpdf(d::DistributionsBacked{T}, v::T, args...) where T  = Distributions.logpdf(d.to_dist(args...), v)
Gen.logpdf_grad(d::DistributionsBacked{T}, v::T,
            args...) where T = Zygote.gradient((v, args...) -> Distributions.logpdf(d.to_dist(args...), v), v, args...)
(d::DistributionsBacked)(args...) = Gen.random(d, args...)
Gen.has_output_grad(d::DistributionsBacked) = d.has_out_grad
Gen.has_argument_grads(d::DistributionsBacked) = d.has_arg_grads

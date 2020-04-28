using Distributions
using DistributionsAD
using ForwardDiff
using Gen
struct DistributionsForwardBacked{T} <: Gen.Distribution{T}
    to_dist
    has_arg_grads :: Vector{Bool}
    has_out_grad :: Bool
end

Gen.random(d::DistributionsForwardBacked, args...) = rand(d.to_dist(args...))
Gen.logpdf(d::DistributionsForwardBacked{T}, v::T, args...) where T  = Distributions.logpdf(d.to_dist(args...), v)
Gen.logpdf_grad(d::DistributionsForwardBacked{T}, v::T,
            args...) where T = ForwardDiff.gradient((v, args...) -> Distributions.logpdf(d.to_dist(args...), v), v, args...)
(d::DistributionsForwardBacked)(args...) = Gen.random(d, args...)
Gen.has_output_grad(d::DistributionsForwardBacked) = d.has_out_grad
Gen.has_argument_grads(d::DistributionsForwardBacked) = d.has_arg_grads

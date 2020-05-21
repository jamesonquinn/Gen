using Gen
using Statistics

import Base.+
import Base.*
import Base.==


#### Type defs

abstract type UnrealizedValue end
abstract type UnrealizedNormalValue <: UnrealizedValue end

abstract type UnrealizedNormalLeafValue <: UnrealizedNormalValue end

mutable struct UnrealizedNormalIndyLeafValue <: UnrealizedNormalLeafValue
    #Use only when all instances are posterior independent by construction. This means committing a priori to which observations will be made.
    baseMean::Float64
    currentCondSD::Float64
    currentMean::Float64
    resetCount::Int64
end

UnrealizedNormalIndyLeafValue(μ, σ) = UnrealizedNormalIndyLeafValue(μ, σ, μ)

UnrealizedNormalIndyLeafValue(μ, σ, μt) = UnrealizedNormalIndyLeafValue(μ, σ, μt, 0)

function copyTo!(source::UnrealizedNormalIndyLeafValue, dest::UnrealizedNormalIndyLeafValue)
    dest.baseMean = source.baseMean
    dest.currentCondSD = source.currentCondSD
    dest.currentMean = source.currentMean
    dest.resetCount = dest.resetCount + 1
end

mutable struct UnrealizedNormalNodeValue <: UnrealizedNormalValue
    offset::Float64
    components::Vector{Tuple{Float64,UnrealizedNormalLeafValue}} #should be unique
end

mutable struct ObservedNormal
    unrealized::UnrealizedNormalNodeValue
    SD::Float64
end

struct UnrealizedDistribution{T} <: Gen.Distribution{T}
    func
end

##### commute definitions (so we only have to define once)

Base.:+(x::Int64,y::UnrealizedNormalValue) = y+x #definitions below should put the "more abstract" thing 1st.
Base.:+(x::UnrealizedNormalLeafValue,y::UnrealizedNormalNodeValue) = y+x #definitions below should put the "more abstract" thing 1st.
Base.:*(x::Int64,y::UnrealizedNormalValue) = y*x #definitions below should put the "more abstract" thing 1st.
Base.:-(x::UnrealizedValue,y::UnrealizedValue) = x + y * -1. #subtraction as addition


##### define ==

function ==(x::UnrealizedNormalIndyLeafValue,y::UnrealizedNormalIndyLeafValue)
    x.baseMean == y.baseMean #FIXME: Incomplete.
end

##### define +

Base.:+(x::UnrealizedNormalLeafValue,y::Float64) = UnrealizedNormalNodeValue(y,[(1.,x)])

Base.:+(x::UnrealizedNormalNodeValue,y::Float64) = UnrealizedNormalNodeValue(x.offset + y,x.components)

Base.:+(x::UnrealizedNormalLeafValue,y::UnrealizedNormalLeafValue) = UnrealizedNormalNodeValue(0.,[(1.,x),(1.,y)])

function Base.:+(anode::UnrealizedNormalNodeValue,aleaf::UnrealizedNormalLeafValue, amul = 1.)
    c = copy(anode.components)
    ret = UnrealizedNormalNodeValue(anode.offset,c)
    for (i, (nmul, nleaf)) in enumerate(c)
        if aleaf === nleaf
            #TODO: remove if nmul + amul == 0.
            c[i] = (nmul + amul, aleaf)
            return ret
        end
    end
    push!(c, (amul,aleaf))
    ret
end


function Base.:+(anode::UnrealizedNormalNodeValue,bnode::UnrealizedNormalNodeValue)
    ret = anode
    for (bmul, bleaf) in bnode.components
        ret = +(ret, bleaf, bmul)
    end
    ret.offset = anode.offset + bnode.offset
    ret
end


##### define *

function Base.:*(x::UnrealizedNormalLeafValue,y::Float64)
    if y == 0.
        UnrealizedNormalNodeValue(0.,[])
    end
    UnrealizedNormalNodeValue(0.,[(y,x)])
end

function Base.:*(anode::UnrealizedNormalNodeValue,y::Float64)
    if y == 0.
        @debug "Why multiply by zero?"
    end
    ret = UnrealizedNormalNodeValue(anode.offset * y,copy(anode.components))
    for (i,(nmul,nleaf)) in enumerate(ret.components)
        ret.components[i] = (nmul * y, nleaf)
        if nmul == 0.
            @debug "Why nmul == zero?"
        end
    end
    ret
end


##### utility functions

function meanvarmean(node::UnrealizedNormalNodeValue)
    (μ0, σ2, μt) = node.offset, 0., node.offset
    for (mul,l) in node.components
        μ0 += l.baseMean * mul
        μt += l.currentMean * mul
        σ2 += (l.currentCondSD * mul)^2
        if isnan(μt)
            @debug "bad leaf?" l node
        end
    end
    (μ0, σ2, μt)
end

function meanvarmean(o::ObservedNormal)
    (μ0, σ2, μt) = meanvarmean(o.unrealized)
    σ2 += o.SD^2
    (μ0, σ2, μt)
end


##### define as if distributions. There must be a better way.

#makeUnrealizedIndyNormal(μ::Float64, σ::Float64) = UnrealizedNormalIndyLeafValue(μ, σ)
unrealizedIndyNormal = UnrealizedDistribution{UnrealizedNormalIndyLeafValue}(UnrealizedNormalIndyLeafValue)

ObservedNormal(μ::UnrealizedNormalLeafValue,σ::Float64) = ObservedNormal(μ + 0., σ)
observedNormal = UnrealizedDistribution{Float64}(ObservedNormal)

##### implement distribution behavior

function sampleFrom!(unilv::UnrealizedNormalIndyLeafValue)
    #@debug "sampleFrom 1"
    if unilv.currentCondSD > 0.
        val = normal(unilv.currentMean, unilv.currentCondSD)
        unilv.currentMean = val
        unilv.currentCondSD = 0.
    else
        val = unilv.currentMean
    end
    val
end

function sampleFrom!(unnv::UnrealizedNormalNodeValue)
    #@debug "sampleFrom 2"
    val = unnv.offset
    for (mul, leaf) in unnv.components
        val += mul * sampleFrom!(leaf)
    end
    val
end

function sampleFrom!(obsn::ObservedNormal)
    #@debug "sampleFrom 3"
    sampleFrom!(obsn.unrealized) + normal(0.,obsn.SD)
end



Gen.random(d::UnrealizedDistribution, args...) = d.func(args...) #This is the whole point - instead of sampling, just record

function Gen.logpdf(d::UnrealizedDistribution{UnrealizedNormalIndyLeafValue}, v::UnrealizedValue, args...)
    #@debug "logpdf UnrealizedNormalIndyLeafValue; resetting" v.resetCount+1
    baseVal = d.func(args...)
    copyTo!(baseVal, v)

    return 0.
    # if (d.func(args...) == v)
    #     return 0.
    # else
    #     @debug "unrealized value must equal itself" d.func(args...) v
    # end
end

function Gen.logpdf_grad(d::UnrealizedDistribution{UnrealizedNormalIndyLeafValue}, v::UnrealizedValue, args...)

    return 0.
    # if (d.func(args...) == v)
    #     return 0.
    # else
    #     @debug "logpdf_grad: unrealized value must equal itself" d.func(args...) v
    # end
end
(d::UnrealizedValue)(args...) = Gen.random(d, args...)
Gen.has_output_grad(d::UnrealizedValue) = true
Gen.has_argument_grads(d::DistributionsBacked) = false

function Gen.random(wrapped::UnrealizedDistribution{Float64}, args...)
    #@debug "random outer"
    d = wrapped.func(args...)
    sampleFrom!(d)
end

(d::UnrealizedDistribution{Float64})(args...) = Gen.random(d, args...)

Gen.logpdf_grad(d::UnrealizedDistribution{Float64}, v::UnrealizedValue, args...) = throw("unimplemented")
Gen.has_output_grad(d::UnrealizedDistribution{Float64}) = false
Gen.has_argument_grads(d::UnrealizedDistribution{Float64}) = false

function Gen.logpdf(wrapped::UnrealizedDistribution{Float64}, v::Float64, args...) #Observe the value. Side effects! This is a hack, for now!
    #@debug "logpdf outer"
    d = wrapped.func(args...)
    Gen.logpdf(d, v)
end





function Gen.logpdf(d::ObservedNormal, v::Float64)
    μ0, σ2, μt = meanvarmean(d)
    lpdf = Gen.logpdf(normal, v, μt, sqrt(σ2))

    #@debug "logpdf" d μ0 σ2 μt v lpdf
    rawDelta = v - μ0
    un = d.unrealized
    com = un.components

    #update all independent leafs
    for (i, (mul, leaf)) in enumerate(com)
        if leaf.currentCondSD > 0. && mul != 0.
            delta = rawDelta + mul * (leaf.baseMean - leaf.currentMean) #use unconditional mean for other components, as they're independent
            likePrec = 1/d.SD^2 #likelihood precision
            totalPrec = likePrec + 1/(mul * leaf.currentCondSD)^2 #ignore variance from other (inpependent) components; that is, condition on point values
            cm, cσ = leaf.currentMean, leaf.currentCondSD
            leaf.currentMean += delta * likePrec / totalPrec / mul
            leaf.currentCondSD = sqrt(1 / totalPrec / mul^2)
            if isnan(leaf.currentMean) || leaf.currentCondSD == 0
                #@debug "updating..." μ0 σ2 μt v delta cm cσ delta/d.SD d.SD likePrec mul totalPrec leaf.currentMean leaf.currentCondSD
            end
        end
    end
    lpdf
end

function compareModels(integratedModel, explicitModel, args, constraints, locus, meansteps = meansteps)
    tr, w = generate(integratedModel, args, constraints)
    tr2s, logws, w2 = importance_sampling(explicitModel, args, constraints, meansteps)
    ws = AnalyticWeights(exp.(logws[1:meansteps]))
    locusvals = [tr2s[i][locus] for i in 1:meansteps]
    m = mean(locusvals, ws)
    vv = var(locusvals, ws)
    @warn "Comparing models:" w w2 m tr[locus].currentMean vv tr[locus].currentCondSD^2
end

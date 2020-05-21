
include("unrealizedNormals.jl")

meansteps = 10000
tol = .2 #kinda loose, but not too loose; ideally this would be calculated analytically but this'll do

function approxEq(a, b, tol = tol)
    a-tol < b < a+tol
end

function compareModels(integratedModel, explicitModel, args, constraints, locus, meansteps = meansteps)
    tr, w = generate(integratedModel, args, constraints)
    tr2s, logws, w2 = importance_sampling(explicitModel, args, constraints, meansteps)
    ws = AnalyticWeights(exp.(logws[1:meansteps]))
    locusvals = [tr2s[i][locus] for i in 1:meansteps]
    m = mean(locusvals, ws)
    vv = var(locusvals, ws, corrected=true)
    @info "Comparing models:" locus w w2 m tr[locus].currentMean vv tr[locus].currentCondSD^2
    @assert approxEq(w, w2) "Estimated weights unequal"
    @assert approxEq(m, tr[locus].currentMean) "Estimated means at locus unequal"
    @assert approxEq(vv, tr[locus].currentCondSD^2) "Estimated variances at locus unequal"
end

mul2 = 2.

@gen function testItExplicit()

    w ~ normal(0., 1.)
    x ~ normal(0., 1.)
    y ~ normal(x+w, 2.)
    z ~ normal(x-w, 2.)
    y2 ~ normal(x+w*mul2, 2.)
    z2 ~ normal(x-w*mul2, 2.)

end

@gen function testItIntegrated()

    w ~ unrealizedIndyNormal(0., 1.)
    x ~ unrealizedIndyNormal(0., 1.)
    y ~ observedNormal(x+w, 2.)
    z ~ observedNormal(x-w, 2.)
    y2 ~ observedNormal(x+w*mul2, 2.)
    z2 ~ observedNormal(x-w*mul2, 2.)
    #Note: must manually ensure that w and x are independent conditional on y, z, y2, z2.
end

if @isdefined doTestUnrealizedNormals
    for v in 0.:.1:1.
        c = Gen.choicemap()
        c[:y] = v
        c[:z] = v
        c[:y2] = v
        c[:z2] = v
        compareModels(testItIntegrated, testItExplicit, (), c, :w)
    end
end

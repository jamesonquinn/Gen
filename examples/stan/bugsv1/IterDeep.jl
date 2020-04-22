using Gen

function iter_deep(c::Gen.ChoiceMap)
  Iterators.flatten([
      Gen.get_values_shallow(c),
      (Dict(Pair{Any, Any}(k => kk, vv) for (kk, vv) in iter_deep(v))
       for (k, v) in Gen.get_submaps_shallow(c))...,
  ])
end

selection(choicemap::Gen.ChoiceMap) = select(keys(iter_deep(choicemap))...)

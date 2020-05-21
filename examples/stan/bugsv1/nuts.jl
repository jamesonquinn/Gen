using Gen
using AdvancedHMC
using Logging
using LoggingExtras

function disable_sample_logging()
    function ignore_sampling_filter(log_args)
        !(occursin("sampling steps",log_args.message) || occursin("adapation steps",log_args.message))
    end

    logger = ActiveFilteredLogger(ignore_sampling_filter, global_logger())


    if !(@isdefined old_logger) #do this only once
        global old_logger = global_logger(logger)
    end
end

function my_nuts(trace, selection,
        transformations,
        n_postadapt_steps = 2,
        n_adapts = 1,
        initial_ϵ_reduce_fac = 10)

    n_NUTS_steps = n_postadapt_steps + n_adapts

    filtered_choices = get_selected(get_choices(trace), selection)
    cur_xy = to_array_transformed(filtered_choices, transformations, 1, Float64)
    dimension = length(cur_xy)
    metric = DiagEuclideanMetric(dimension)

    retval_grad = nothing #accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing

    function update_xy(val)
        extra_constraints, grad_val = from_array_transformed(filtered_choices, transformations, val)
        (new_trace, weight, discard, retdiff) = update(trace, get_args(trace),
                                                    map((_)->NoChange(), get_args(trace)),
                                                    extra_constraints)
        (new_trace, weight, grad_val)
    end

    function val_to_lp_plus_c(val)
        (new_trace, weight, grad_val) = update_xy(val)
        weight
    end

    function val_to_grad(val)
        (new_trace, weight, grad_val) = update_xy(val)
        (retval_grad_out, values_trie, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
        raw_grad = to_array(gradient_trie, Float64)
        grad = raw_grad .* grad_val #chain rule
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


    (new_trace, weight, grad_val) = update_xy(samples[n_NUTS_steps])
    new_trace
end


transform_log = (log, exp, exp)
    #(restricted->unrestricted,
        #unrestricted->restricted,
        #unrestricted->d(restricted)/d(unrestricted))


function to_array_transformed(choices::ChoiceMap, transformations::ChoiceMap, whichtrans, ::Type{T}) where {T}
    arr = Vector{T}(undef, 32)
    n = _fill_array_transformed!(choices, transformations, whichtrans, arr, 1)
    @assert n <= length(arr)
    resize!(arr, n)
    arr
end

function _fill_array_transformed!(choices::DynamicChoiceMap, transformations::ChoiceMap, whichtrans,
                arr::Vector{T}, start_idx::Int) where {T}
    leaf_keys_sorted = sort(collect(keys(choices.leaf_nodes)))
    internal_node_keys_sorted = sort(collect(keys(choices.internal_nodes)))
    idx = start_idx
    for key in leaf_keys_sorted
        value = choices.leaf_nodes[key]
        if has_value(transformations,key)
            n_written = _fill_array_transformed!(value, transformations[key][whichtrans], arr, idx)
        else
            n_written = Gen._fill_array!(value, arr, idx)
        end
        idx += n_written
    end
    for key in internal_node_keys_sorted
        if key in keys(transformations.internal_nodes)
            n_written = _fill_array_transformed!(get_submap(choices, key), get_submap(transformations, key), whichtrans, arr, idx)
        else
            n_written = Gen._fill_array!(get_submap(choices, key), arr, idx)
        end
        idx += n_written
    end
    idx - start_idx
end


function _fill_array_transformed!(value::T, transformation, arr::Vector{T}, start_idx::Int) where {T}
    if length(arr) < start_idx
        resize!(arr, 2 * start_idx)
    end
    arr[start_idx] = transformation(value)
    1
end

function _fill_array_transformed!(value::Vector{T}, transformation, arr::Vector{T}, start_idx::Int) where {T}
    if length(arr) < start_idx + length(value)
        resize!(arr, 2 * (start_idx + length(value)))
    end
    arr[start_idx:start_idx+length(value)-1] = transformation.(value)
    length(value)
end

function from_array_transformed(proto_choices::ChoiceMap, transformations::ChoiceMap, arr::Vector)
    grad_arr = copy(arr)
    (n, choices) = _from_array_transformed!(proto_choices, transformations, grad_arr, 1)
    if n != length(arr)
        error("Dimension mismatch: $n, $(length(arr))")
    end
    choices, grad_arr
end

function applyTransformation(transformation, value::T) where {T}
    transformation(value)
end
function applyTransformation(transformation, value::Vector{T}) where {T}
    transformation.(value)
end

function _from_array_transformed!(proto_choices::DynamicChoiceMap, transformations::ChoiceMap, arr::Vector{T}, start_idx::Int) where {T}
    @assert length(arr) >= start_idx
    choices = DynamicChoiceMap()
    leaf_keys_sorted = sort(collect(keys(proto_choices.leaf_nodes)))
    internal_node_keys_sorted = sort(collect(keys(proto_choices.internal_nodes)))
    idx = start_idx
    for key in leaf_keys_sorted
        (n_read, value) = Gen._from_array(proto_choices.leaf_nodes[key], arr, idx)
        if has_value(transformations,key)
            transformedValue = applyTransformation(transformations[key][2],value) #restricted to unrestricted
            choices.leaf_nodes[key] = transformedValue
            if transformations[key][2] != transformations[key][3] #not exp, so grad != inverse
                transformedValue = applyTransformation(transformations[key][3],value) #restricted to grad unrestricted
            end
            if n_read == 1
                arr[idx] = transformedValue
            else
                arr[idx:idx+n_read-1] = transformedValue
            end
        else
            choices.leaf_nodes[key] = value
            arr[idx:idx+n_read-1] = ones(T, n_read) #identity transform has a grad of 1
        end
        idx += n_read
    end
    for key in internal_node_keys_sorted
        if key in keys(transformations.internal_nodes)
            (n_read, node) = _from_array_transformed!(get_submap(proto_choices, key),
                                                get_submap(transformations, key), arr, idx)
        else
            (n_read, node) = Gen._from_array(get_submap(proto_choices, key), arr, idx)
            arr[idx:idx+n_read-1] = ones(T, n_read) #identity transform has a grad of 1
        end
        idx += n_read
        choices.internal_nodes[key] = node
    end
    (idx - start_idx, choices)
end

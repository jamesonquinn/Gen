# -*- coding: utf-8 -*-
# +
using AdvancedHMC, Distributions, ForwardDiff

# Choose parameter dimensionality and initial parameter value
D = 10; initial_θ = rand(D)

# Define the target distribution
ℓπ(θ) = logpdf(MvNormal(zeros(D), ones(D)), θ)

# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 2_000, 1_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
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
samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)
# +
using Gen

s = select(:a)
sn = complement(s)
cm = Gen.choicemap()
cm[:b] = 3.
cm[:c] = log
cm[:d => :a] = exp
println(to_array(cm, Any))
println(has_value(cm,:d))
println(get_submap(cm,:d))
println(get_submap(cm,:c))

# +
a = [3]

function add!(x)
    x[1] += 1
end

add!(a)

a
# -


Gen.choicemap

Gen.ChoiceMap

# +
using PyPlot
using PyCall
using Base64
@pyimport matplotlib.animation as anim

#Construct Figure and Plot Data
fig = figure("MyFigure",figsize=(5,5))
ax = PyPlot.axes(xlim = (0,10),ylim=(0,10))


println("hi")


global line1 = ax[:plot]([],[],"r-")[1]
global line2 = ax[:plot]([],[],"g-")[1]
global line3 = ax[:plot]([],[],"b-")[1]

# Define the init function, which draws the first frame (empty, in this case)
function init()
    global line1
    global line2
    global line3
    line1[:set_data]([],[])
    line2[:set_data]([],[])
    line3[:set_data]([],[])
    return (line1,line2,line3,Union{})  # Union{} is the new word for None
end

# Animate draws the i-th frame, where i starts at i=0 as in Python.
function animate(i)
    global line1
    global line2
    global line3
    x = (0:i)/10.0
    line1[:set_data](x,x)
    line2[:set_data](1 .+ x,x)
    line3[:set_data](2 .+ x,x)
    return (line1,line2,line3,Union{})
end

# Create the animation object by calling the Python function FuncAnimaton
myanim = anim.FuncAnimation(fig, animate, init_func=init, frames=100, interval=20)

# Convert it to an MP4 movie file and saved on disk in this format.
#myanim[:save]("3Lines.mp4", bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
myanim[:save]("test1.mp4", bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    
# Function for creating an embedded video given a filename
function html_video(filename)
    open(filename) do f
        base64_video = Base64.base64encode(f)
        """<video controls src="data:video/x-m4v;base64,$base64_video">"""
    end
end

# Display the movie in a Julia cell as follows. Note it has animation controls for the user.
#display("text/html", html_video("3Lines.mp4"))
display("text/html", html_video("test1.mp4"))
# -



# +
using PyPlot
using PyCall
using Base64
@pyimport matplotlib.animation as anim

println("before")
#Construct Figure and Plot Data
fig = figure("MyFigure",figsize=(5,5))
println("middle")
ax = PyPlot.axes(xlim = (-3,3),ylim=(-3,3))

n = 1000
raw_x = 
println("after")

# Define the init function, which draws the first frame (empty, in this case)
function init()
    points = ax.sca([],[],"r-")[1]
    xline2 = ax.plot([],[],"g-")[1]
    xline3 = ax.plot([],[],"b-")[1]
    xline1.set_data([],[])
    xline2.set_data([],[])
    xline3.set_data([],[])
    return (line1,line2,line3,Union{})  # Union{} is the new word for None
end

# Animate draws the i-th frame, where i starts at i=0 as in Python.
function animate(i)
    xline1 = ax.plot([],[],"r-")[1]
    xline2 = ax.plot([],[],"g-")[1]
    xline3 = ax.plot([],[],"b-")[1]
    x = (0:i)/10.0
    xline1.set_data(x,x)
    xline2.set_data(1 .+ x,x)
    xline3.set_data(2 .+ x,x)
    return (line1,line2,line3,Union{})
end

println("4")
# Create the animation object by calling the Python function FuncAnimaton
myanim = anim.FuncAnimation(fig, animate, init_func=init, frames=100, interval=20)

println("5")
# Convert it to an MP4 movie file and saved on disk in this format.
#myanim[:save]("3Lines.mp4", bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
myanim.save("test1.mp4", bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    
println("6")
# Function for creating an embedded video given a filename
function html_video(filename)
    open(filename) do f
        base64_video = Base64.base64encode(f)
        """<video controls src="data:video/x-m4v;base64,$base64_video">"""
    end
end

println("pre-end")
# Display the movie in a Julia cell as follows. Note it has animation controls for the user.
#display("text/html", html_video("3Lines.mp4"))
display("text/html", html_video("test1.mp4"))


println("end")

PyPlot.close(fig)
;

# +

include("AnimatedPyplot.jl")

test_animation()
# -

ff = () -> 42
ff()

# +
using Logging
using LoggingExtras

function ignore_sampling_filter(log_args)
    !occursin("sampling steps",log_args.message)
end
logger = ActiveFilteredLogger(ignore_sampling_filter, global_logger())
old_logger = global_logger(logger)
@debug "debug"
@info "info sampling steps"
@warn "warn"

print(occursin("sd","asdt"))
# -

isdefined(:logger)

@isdefined loggerxx

true || false

tuple(1:10:.5)

agrid = collect(1:.5:5)

sqrt.(agrid)

agrid[end]

c = choicemap()

log(5)

println(1.5 in 1:5)

ones(Int64,3)



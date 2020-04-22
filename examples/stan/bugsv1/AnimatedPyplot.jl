using PyPlot
using PyCall
using Base64
anim = pyimport( "matplotlib.animation")

function display_animation(name, make_axis, init, animate, frames=100, interval=20)

    #Construct Figure and do basic setup
    (fig, axis) = make_axis()

    # Create the animation object by calling the Python function FuncAnimaton
    myanim = anim.FuncAnimation(fig, (i) -> animate(axis,i), init_func= ()-> init(axis), frames=frames, interval=interval)

    # Convert it to an MP4 movie file and saved on disk in this format.
    #myanim[:save]("3Lines.mp4", bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    myanim.save(name*".mp4", bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])

    # Function for creating an embedded video given a filename
    function html_video(filename)
        open(filename) do f
            base64_video = Base64.base64encode(f)
            """<video controls src="data:video/x-m4v;base64,$base64_video">"""
        end
    end

    # Display the movie in a Julia cell as follows. Note it has animation controls for the user.
    #display("text/html", html_video("3Lines.mp4"))
    display("text/html", html_video(name*".mp4"))


    PyPlot.close(fig)
end

function test_animation()
    function make_axis()
        fig = figure("MyFigure",figsize=(5,5))
        ax = PyPlot.axes(xlim = (0,10),ylim=(0,10))
        (fig, ax)
    end

    # Define the init function, which draws the first frame (empty, in this case)
    function init(ax)
        xline1 = ax.plot([],[],"r-")[1]
        xline2 = ax.plot([],[],"g-")[1]
        xline3 = ax.plot([],[],"b-")[1]
        xline1.set_data([],[])
        xline2.set_data([],[])
        xline3.set_data([],[])
        return (xline1,xline2,xline3,Union{})  # Union{} is the new word for None
    end

    # Animate draws the i-th frame, where i starts at i=0 as in Python.
    function animate(ax,i)
        xline1 = ax.plot([],[],"r-")[1]
        xline2 = ax.plot([],[],"g-")[1]
        xline3 = ax.plot([],[],"b-")[1]
        x = (0:i)/10.0
        xline1.set_data(x,x)
        xline2.set_data(1 .+ x,x)
        xline3.set_data(2 .+ x,x)
        return (xline1,xline2,xline3,Union{})
    end

    display_animation("test1",make_axis,init,animate)
end

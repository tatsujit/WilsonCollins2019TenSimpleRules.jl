using DrWatson
@quickactivate "WilsonCollins2019TenSimpleRules.jl"

# Here you may include files from the source directory
using MultiBandits
using Random, DataFrames
using LaTeXStrings
# include(srcdir("plots.jl"))

# make the docstring for the function
"""
    box2_fig1!(ax, histories, model_names::Vector{String})

Draw the figure on the axis `ax` for the box 2, figure 1.

# Arguments
- `ax`: Axis object to draw the figure on
- `histories::Vector{History}`: histories of the models
- `model_names::Vector{String}`: names of the models
"""
function box2_fig1A!(ax, histories, model_names::Vector{String})
    @assert (n = length(histories)) == length(model_names)
    p_stays = Vector{Vector{Float64}}()

    for (i, history) in enumerate(histories)
        # p_stays[i, :] = prob_stay(history.actions, history.rewards)
        push!(p_stays, prob_stay(history.actions, history.rewards))
    end

    # font size for titles, xlabels, ylabels
    (titlesize, xlabelsize, ylabelsize) = (32, 24, 24)

    ax.title = "stay behavior"
    ax.xlabel = L"\text{previous reward}"
    ax.ylabel = L"p(\text{stay})"
    ax.titlesize = titlesize
    ax.xlabelsize = xlabelsize
    ax.ylabelsize = ylabelsize
    ϵ = 0.1
    ax.limits = ((0-ϵ, 1+ϵ), (0-ϵ, 1+ϵ))

    # lines!(ax, 0-ϵ:ϵ:1+ϵ, 0-ϵ:ϵ:1+ϵ)
    for (i, p_stay) in enumerate(p_stays)
        lines!(ax, [0, 1], p_stay, label=labels[i], )
        scatter!(ax, [0, 1], p_stay, # label=labels[i], 
                marker=:circle, markersize=20)
    end 
    axislegend(ax, position=:cb, backgroundcolor=:transparent)
    return ax
end


# save and display
# fig |> display



using CairoMakie

################################################################
# Plotting Functions
################################################################

"""
    plot_utility_functions(;
        x_range=(-2.0, 3.0),
        n_points=1000,
        figure_size=(1600, 400),
        line_width=3,
        colors=[:blue, :red, :green, :orange, :purple]
    )

Plot all five utility functions in a single row layout.

# Arguments
- `x_range`: Range of x values to plot
- `n_points`: Number of points for smooth curves
- `figure_size`: Figure size in pixels
- `line_width`: Width of the function lines
- `colors`: Colors for each function
"""
function plot_utility_functions(;
    x_range=(-2.0, 3.0),
    n_points=1000,
    figure_size=(1600, 400),
    line_width=3,
    colors=[:blue, :red, :green, :orange, :purple]
)

    # Create x values
    x = range(x_range[1], x_range[2], length=n_points)

    # Function configurations with parameters
    function_configs = [
        (name="Step Function",
         func=step_utility_function,
         params=(origin=0.5, scale=1.0),
         color=colors[1]),

        (name="Sigmoid Function",
         func=sigmoid_utility_function,
         params=(origin=0.5, gain=3.0),
         color=colors[2]),

        (name="Piecewise Linear\nSigmoid",
         func=piecewise_linear_sigmoid_utility_function,
         params=(low=0.0, high=1.5, scale=1.0),
         color=colors[3]),

        (name="Piecewise Linear",
         func=piecewise_linear_utility_function,
         params=(low=0.0, high=1.5, slope1=0.2, slope2=1.0, slope3=0.3),
         color=colors[4]),

        (name="Discontinuous Linear",
         func=discontinuous_linear_utility_function,
         params=(origin=0.5, slope1=0.3, slope2=1.2),
         color=colors[5])
    ]

    # Create figure
    fig = Figure(size=figure_size)

    # Plot each function
    for (i, config) in enumerate(function_configs)
        ax = Axis(fig[1, i],
                  title=config.name,
                  xlabel="x",
                  ylabel="Utility",
                  titlesize=14,
                  xlabelsize=12,
                  ylabelsize=12)

        # Calculate y values
        y = [config.func(xi; config.params...) for xi in x]

        # Plot the function
        lines!(ax, x, y,
               color=config.color,
               linewidth=line_width,
               label=config.name)

        # Add grid
        ax.xgridvisible = true
        ax.ygridvisible = true
        ax.xgridcolor = :lightgray
        ax.ygridcolor = :lightgray

        # Set reasonable y limits
        y_min = minimum(y) - 0.1 * (maximum(y) - minimum(y))
        y_max = maximum(y) + 0.1 * (maximum(y) - minimum(y))
        ylims!(ax, y_min, y_max)

        # Add special markers for key points
        add_special_markers!(ax, config, x_range)
    end

    return fig
end

"""
    add_special_markers!(ax, config, x_range)

Add special markers to highlight key points of each utility function.
"""
function add_special_markers!(ax, config, x_range)
    if config.name == "Step Function"
        # Mark the step point
        origin = config.params.origin
        if x_range[1] <= origin <= x_range[2]
            scatter!(ax, [origin], [0], color=:black, markersize=8, marker=:circle)
            scatter!(ax, [origin], [config.params.scale], color=:black, markersize=8, marker=:circle)
        end

    elseif config.name == "Sigmoid Function"
        # Mark the inflection point
        origin = config.params.origin
        if x_range[1] <= origin <= x_range[2]
            y_inflection = config.func(origin; config.params...)
            scatter!(ax, [origin], [y_inflection], color=:black, markersize=8, marker=:circle)
        end

    elseif config.name == "Piecewise Linear\nSigmoid"
        # Mark the transition points
        low, high = config.params.low, config.params.high
        if x_range[1] <= low <= x_range[2]
            scatter!(ax, [low], [0], color=:black, markersize=8, marker=:circle)
        end
        if x_range[1] <= high <= x_range[2]
            scatter!(ax, [high], [config.params.scale], color=:black, markersize=8, marker=:circle)
        end

    elseif config.name == "Piecewise Linear"
        # Mark the breakpoints
        low, high = config.params.low, config.params.high
        if x_range[1] <= low <= x_range[2]
            y_low = config.func(low; config.params...)
            scatter!(ax, [low], [y_low], color=:black, markersize=8, marker=:circle)
        end
        if x_range[1] <= high <= x_range[2]
            y_high = config.func(high; config.params...)
            scatter!(ax, [high], [y_high], color=:black, markersize=8, marker=:circle)
        end

    elseif config.name == "Discontinuous Linear"
        # Mark the discontinuity point
        origin = config.params.origin
        if x_range[1] <= origin <= x_range[2]
            y_left = config.func(origin - 1e-10; config.params...)
            y_right = config.func(origin; config.params...)
            scatter!(ax, [origin], [y_left], color=:black, markersize=8, marker=:circle)
            scatter!(ax, [origin], [y_right], color=:black, markersize=8, marker=:circle,
                    strokecolor=:black, strokewidth=2)
        end
    end
end

"""
    plot_utility_functions_comparison(;
        x_range=(-2.0, 3.0),
        n_points=1000,
        figure_size=(800, 600)
    )

Plot all utility functions on a single axis for comparison.
"""
function plot_utility_functions_comparison(;
    x_range=(-2.0, 3.0),
    n_points=1000,
    figure_size=(800, 600),
    colors=[:blue, :red, :green, :orange, :purple]
)

    # Create x values
    x = range(x_range[1], x_range[2], length=n_points)

    # Function configurations
    function_configs = [
        (name="Step", func=step_utility_function, params=(origin=0.5, scale=1.0), color=colors[1]),
        (name="Sigmoid", func=sigmoid_utility_function, params=(origin=0.5, gain=3.0), color=colors[2]),
        (name="Piecewise Linear Sigmoid", func=piecewise_linear_sigmoid_utility_function,
         params=(low=0.0, high=1.5, scale=1.0), color=colors[3]),
        (name="Piecewise Linear", func=piecewise_linear_utility_function,
         params=(low=0.0, high=1.5, slope1=0.2, slope2=1.0, slope3=0.3), color=colors[4]),
        (name="Discontinuous Linear", func=discontinuous_linear_utility_function,
         params=(origin=0.5, slope1=0.3, slope2=1.2), color=colors[5])
    ]

    # Create figure
    fig = Figure(size=figure_size)
    ax = Axis(fig[1, 1],
              title="Utility Functions Comparison",
              xlabel="x",
              ylabel="Utility",
              titlesize=16,
              xlabelsize=14,
              ylabelsize=14)

    # Plot each function
    for config in function_configs
        y = [config.func(xi; config.params...) for xi in x]
        lines!(ax, x, y,
               color=config.color,
               linewidth=3,
               label=config.name)
    end

    # Add legend
    axislegend(ax, position=:rt, backgroundcolor=:white, framecolor=:gray)

    # Add grid
    ax.xgridvisible = true
    ax.ygridvisible = true
    ax.xgridcolor = :lightgray
    ax.ygridcolor = :lightgray

    return fig
end

"""
    plot_utility_functions_with_parameters()

Interactive-style plot showing how parameters affect the functions.
"""
function plot_utility_functions_with_parameters()
    x_range = (-2.0, 3.0)
    x = range(x_range[1], x_range[2], length=1000)

    fig = Figure(size=(1600, 800))

    # Row 1: Default parameters
    for (i, (name, func)) in enumerate(pairs(utility_functions))
        ax = Axis(fig[1, i],
                  title="$name (default)",
                  xlabel="x",
                  ylabel="Utility")

        # Default parameters for each function
        if name == "step"
            y = [func(xi, origin=0.5, scale=1.0) for xi in x]
        elseif name == "sigmoid"
            y = [func(xi, origin=0.5, gain=3.0) for xi in x]
        elseif name == "piecewise_linear_sigmoid"
            y = [func(xi, low=0.0, high=1.5, scale=1.0) for xi in x]
        elseif name == "piecewise_linear"
            y = [func(xi, low=0.0, high=1.5, slope1=0.2, slope2=1.0, slope3=0.3) for xi in x]
        elseif name == "discontinuous_linear"
            y = [func(xi, origin=0.5, slope1=0.3, slope2=1.2) for xi in x]
        end

        lines!(ax, x, y, color=:blue, linewidth=3)
        ax.xgridvisible = true
        ax.ygridvisible = true
    end

    # Row 2: Alternative parameters
    for (i, (name, func)) in enumerate(pairs(utility_functions))
        ax = Axis(fig[2, i],
                  title="$name (alternative)",
                  xlabel="x",
                  ylabel="Utility")

        # Alternative parameters for each function
        if name == "step"
            y = [func(xi, origin=1.0, scale=1.5) for xi in x]
        elseif name == "sigmoid"
            y = [func(xi, origin=1.0, gain=1.0) for xi in x]
        elseif name == "piecewise_linear_sigmoid"
            y = [func(xi, low=-0.5, high=2.0, scale=1.2) for xi in x]
        elseif name == "piecewise_linear"
            y = [func(xi, low=-0.5, high=2.0, slope1=0.5, slope2=0.8, slope3=0.1) for xi in x]
        elseif name == "discontinuous_linear"
            y = [func(xi, origin=1.0, slope1=0.8, slope2=0.4) for xi in x]
        end

        lines!(ax, x, y, color=:red, linewidth=3)
        ax.xgridvisible = true
        ax.ygridvisible = true
    end

    return fig
end

# Example usage:
fig1 = plot_utility_functions()
display(fig1)
save(joinpath(output_dir, "utility_functions_plot.pdf"), fig1)#, resolution=(300, 300))

fig2 = plot_utility_functions_comparison()
display(fig2)
save(joinpath(output_dir, "utility_functions_comparison_plot.pdf"), fig2)#, resolution=(300, 300))


fig3 = plot_utility_functions_with_parameters()
display(fig3)
save(joinpath(output_dir, "utility_functions_with_parameters_plot.pdf"), fig3)#, resolution=(300, 300))

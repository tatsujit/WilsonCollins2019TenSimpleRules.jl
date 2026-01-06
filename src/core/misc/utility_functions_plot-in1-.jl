using CairoMakie

################################################################
# Combined Utility Functions Visualization
################################################################

"""
    plot_utility_functions_combined(;
        x_range=(-2.0, 3.0),
        n_points=1000,
        figure_size=(1800, 1200),
        line_width=2.5,
        colors=[:blue, :red, :green, :orange, :purple]
    )

Create a comprehensive visualization combining all three utility function plots:
1. Individual functions (top row)
2. Comparison plot (bottom left)
3. Parameter variations (bottom right)

# Arguments
- `x_range`: Range of x values to plot
- `n_points`: Number of points for smooth curves
- `figure_size`: Overall figure size in pixels
- `line_width`: Width of the function lines
- `colors`: Colors for each function
"""
function plot_utility_functions_combined(;
    x_range=(-2.0, 3.0),
    n_points=1000,
    figure_size=(1800, 1200),
    line_width=2.5,
    colors=[:blue, :red, :green, :orange, :purple]
)

    # Create x values
    x = range(x_range[1], x_range[2], length=n_points)

    # Function configurations with parameters
    function_configs = [
        (name="Step Function", short_name="Step",
         func=step_utility_function,
         params=(origin=0.5, scale=1.0),
         alt_params=(origin=1.0, scale=1.5),
         color=colors[1]),

        (name="Sigmoid Function", short_name="Sigmoid",
         func=sigmoid_utility_function,
         params=(origin=0.5, gain=3.0),
         alt_params=(origin=1.0, gain=1.0),
         color=colors[2]),

        (name="Piecewise Linear\nSigmoid", short_name="PL Sigmoid",
         func=piecewise_linear_sigmoid_utility_function,
         params=(low=0.0, high=1.5, scale=1.0),
         alt_params=(low=-0.5, high=2.0, scale=1.2),
         color=colors[3]),

        (name="Piecewise Linear", short_name="Piecewise",
         func=piecewise_linear_utility_function,
         params=(low=0.0, high=1.5, slope1=0.2, slope2=1.0, slope3=0.3),
         alt_params=(low=-0.5, high=2.0, slope1=0.5, slope2=0.8, slope3=0.1),
         color=colors[4]),

        (name="Discontinuous Linear", short_name="Discontinuous",
         func=discontinuous_linear_utility_function,
         params=(origin=0.5, slope1=0.3, slope2=1.2),
         alt_params=(origin=1.0, slope1=0.8, slope2=0.4),
         color=colors[5])
    ]

    # Create main figure with GridLayout
    fig = Figure(size=figure_size, fontsize=12)

    # Create a structured layout
    # Top row: Individual function plots (5 columns)
    # Bottom: 2x2 grid with comparison and parameter plots

    # Top section: Individual functions
    top_section = fig[1, 1:5] = GridLayout()

    for (i, config) in enumerate(function_configs)
        ax = Axis(top_section[1, i],
                  title=config.name,
                  xlabel="x",
                  ylabel="Utility",
                  titlesize=11,
                  xlabelsize=10,
                  ylabelsize=10)

        # Calculate y values
        y = [config.func(xi; config.params...) for xi in x]

        # Plot the function
        lines!(ax, x, y,
               color=config.color,
               linewidth=line_width)

        # Add grid
        ax.xgridvisible = true
        ax.ygridvisible = true
        ax.xgridcolor = :lightgray
        ax.ygridcolor = :lightgray
        ax.xgridwidth = 0.5
        ax.ygridwidth = 0.5

        # Set reasonable y limits
        y_range = maximum(y) - minimum(y)
        padding = max(0.1 * y_range, 0.05)
        ylims!(ax, minimum(y) - padding, maximum(y) + padding)

        # Add special markers
        add_special_markers!(ax, config, x_range)
    end

    # Bottom left: Comparison plot
    ax_comp = Axis(fig[2, 1:2],
                   title="All Functions Comparison",
                   xlabel="x",
                   ylabel="Utility",
                   titlesize=14,
                   xlabelsize=12,
                   ylabelsize=12)

    for config in function_configs
        y = [config.func(xi; config.params...) for xi in x]
        lines!(ax_comp, x, y,
               color=config.color,
               linewidth=line_width + 0.5,
               label=config.short_name)
    end

    axislegend(ax_comp, position=:rt, backgroundcolor=:white,
               framecolor=:gray, labelsize=10)
    ax_comp.xgridvisible = true
    ax_comp.ygridvisible = true
    ax_comp.xgridcolor = :lightgray
    ax_comp.ygridcolor = :lightgray

    # Bottom right: Parameter comparison (2x3 mini grid)
    param_section = fig[2, 3:5] = GridLayout()

    # Create 2 rows of parameter comparisons
    for row in 1:2
        for (i, config) in enumerate(function_configs)
            if (row == 1 && i <= 3) || (row == 2 && i > 3)
                col = row == 1 ? i : i - 3
                ax = Axis(param_section[row, col],
                          title=config.short_name,
                          xlabel="x",
                          ylabel="Utility",
                          titlesize=10,
                          xlabelsize=9,
                          ylabelsize=9,
                          xticklabelsize=8,
                          yticklabelsize=8)

                # Default parameters
                y_default = [config.func(xi; config.params...) for xi in x]
                lines!(ax, x, y_default,
                       color=config.color,
                       linewidth=line_width - 0.5,
                       label="Default")

                # Alternative parameters
                y_alt = [config.func(xi; config.alt_params...) for xi in x]
                lines!(ax, x, y_alt,
                       color=config.color,
                       linewidth=line_width - 0.5,
                       linestyle=:dash,
                       label="Alternative")

                ax.xgridvisible = true
                ax.ygridvisible = true
                ax.xgridcolor = :lightgray
                ax.ygridcolor = :lightgray
                ax.xgridwidth = 0.5
                ax.ygridwidth = 0.5

                # Add legend to first subplot only
                if row == 1 && col == 1
                    axislegend(ax, position=:rt, labelsize=8)
                end
            end
        end
    end

    # Add overall title
    Label(fig[0, 1:5], "Utility Functions: Individual, Comparison, and Parameter Analysis",
          fontsize=16, font=:bold, halign=:center)

    # Adjust spacing
    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 15)

    return fig
end

"""
    add_special_markers!(ax, config, x_range)

Add special markers to highlight key points of each utility function.
"""
function add_special_markers!(ax, config, x_range)
    marker_size = 6

    if config.name == "Step Function"
        origin = config.params.origin
        if x_range[1] <= origin <= x_range[2]
            scatter!(ax, [origin], [0], color=:black, markersize=marker_size, marker=:circle)
            scatter!(ax, [origin], [config.params.scale], color=:black, markersize=marker_size, marker=:circle)
        end

    elseif config.name == "Sigmoid Function"
        origin = config.params.origin
        if x_range[1] <= origin <= x_range[2]
            y_inflection = config.func(origin; config.params...)
            scatter!(ax, [origin], [y_inflection], color=:black, markersize=marker_size, marker=:circle)
        end

    elseif config.name == "Piecewise Linear\nSigmoid"
        low, high = config.params.low, config.params.high
        if x_range[1] <= low <= x_range[2]
            scatter!(ax, [low], [0], color=:black, markersize=marker_size, marker=:circle)
        end
        if x_range[1] <= high <= x_range[2]
            scatter!(ax, [high], [config.params.scale], color=:black, markersize=marker_size, marker=:circle)
        end

    elseif config.name == "Piecewise Linear"
        low, high = config.params.low, config.params.high
        for point in [low, high]
            if x_range[1] <= point <= x_range[2]
                y_point = config.func(point; config.params...)
                scatter!(ax, [point], [y_point], color=:black, markersize=marker_size, marker=:circle)
            end
        end

    elseif config.name == "Discontinuous Linear"
        origin = config.params.origin
        if x_range[1] <= origin <= x_range[2]
            y_left = config.func(origin - 1e-10; config.params...)
            y_right = config.func(origin; config.params...)
            scatter!(ax, [origin], [y_left], color=:black, markersize=marker_size, marker=:circle)
            scatter!(ax, [origin], [y_right], color=:black, markersize=marker_size, marker=:circle,
                    strokecolor=:black, strokewidth=1)
        end
    end
end

"""
    create_utility_functions_summary()

Create a more compact summary version with just the essentials.
"""
function create_utility_functions_summary(;
    x_range=(-2.0, 3.0),
    figure_size=(1400, 800)
)

    fig = Figure(size=figure_size, fontsize=11)

    # Define functions and parameters
    x = range(x_range[1], x_range[2], length=1000)
    colors = [:blue, :red, :green, :orange, :purple]

    function_configs = [
        (name="Step", func=step_utility_function, params=(origin=0.5, scale=1.0)),
        (name="Sigmoid", func=sigmoid_utility_function, params=(origin=0.5, gain=3.0)),
        (name="PL Sigmoid", func=piecewise_linear_sigmoid_utility_function, params=(low=0.0, high=1.5, scale=1.0)),
        (name="Piecewise", func=piecewise_linear_utility_function, params=(low=0.0, high=1.5, slope1=0.2, slope2=1.0, slope3=0.3)),
        (name="Discontinuous", func=discontinuous_linear_utility_function, params=(origin=0.5, slope1=0.3, slope2=1.2))
    ]

    # Top row: Individual functions
    for (i, (name, func, params)) in enumerate(function_configs)
        ax = Axis(fig[1, i], title=name, xlabel="x", ylabel="Utility")
        y = [func(xi; params...) for xi in x]
        lines!(ax, x, y, color=colors[i], linewidth=2.5)
        ax.xgridvisible = true
        ax.ygridvisible = true
    end

    # Bottom: Comparison
    ax_comp = Axis(fig[2, 1:5], title="Comparison of All Utility Functions",
                   xlabel="x", ylabel="Utility")

    for (i, (name, func, params)) in enumerate(function_configs)
        y = [func(xi; params...) for xi in x]
        lines!(ax_comp, x, y, color=colors[i], linewidth=3, label=name)
    end

    axislegend(ax_comp, position=:rt, backgroundcolor=:white)
    ax_comp.xgridvisible = true
    ax_comp.ygridvisible = true

    return fig
end

# Usage examples:

# Main comprehensive plot
fig_combined = plot_utility_functions_combined()
display(fig_combined)
save(joinpath(output_dir, "utility_functions_comprehensive.pdf"), fig_combined)

# Compact summary version
fig_summary = create_utility_functions_summary()
display(fig_summary)
save(joinpath(output_dir, "utility_functions_summary.pdf"), fig_summary)

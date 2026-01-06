# More focused version: Q values and expectations only
function plot_q_expectations(history::EstimatorHistory;
                            figsize = (800, 500))
    n_trials = length(history.actions)
    trials = 1:n_trials

    fig = Figure(size = figsize)
    ax = Axis(fig[1, 1],
              title = "Q Values vs Expectations",
              xlabel = "Trial",
              ylabel = "Value")

    colors = Makie.wong_colors()[1:history.n_arms]

    for arm in 1:history.n_arms
        # Q values (solid lines)
        lines!(ax, trials, history.Qss[:, arm],
               color = colors[arm], linewidth = 2.5,
               label = "Q Arm $arm")

        # Expectations (dashed lines)
        expectations_arm = [exp[arm] for exp in history.expectations]
        lines!(ax, trials, expectations_arm,
               color = colors[arm], linestyle = :dash,
               linewidth = 2, alpha = 0.8,
               label = "E Arm $arm")
    end

    axislegend(ax, position = :rt)
    fig
end

# Compact version with subplots for each variable
function plot_estimator_evolution_compact(history::EstimatorHistory;
                                     figsize = (1000, 600))
    fig = Figure(size = figsize)

    variables = [
        (history.Qss, "Q Values"),
        (history.Vss, "V Values"),
        (history.Css, "C Values"),
        (history.Wss, "W Values"),
        (history.Nss, "N Values")
    ]

    trials = 1:length(history.actions)
    colors = Makie.wong_colors()[1:history.n_arms]

    for (i, (data, title)) in enumerate(variables)
        row, col = divrem(i-1, 3) .+ (1, 1)
        ax = Axis(fig[row, col], title = title, xlabel = "Trial")

        for arm in 1:history.n_arms
            if title == "N Values"
                stairs!(ax, trials, data[:, arm],
                       color = colors[arm], linewidth = 2,
                       label = "Arm $arm")
            else
                lines!(ax, trials, data[:, arm],
                      color = colors[arm], linewidth = 2,
                      label = "Arm $arm")
            end
        end

        # Add expectations to Q plot
        if title == "Q Values"
            for arm in 1:history.n_arms
                expectations_arm = [exp[arm] for exp in history.expectations]
                lines!(ax, trials, expectations_arm,
                       color = colors[arm], linestyle = :dash,
                       linewidth = 1.5, alpha = 0.7)
            end
        end

        axislegend(ax, position = :rt)
    end

    fig
end

# Interactive version with action highlighting
function plot_estimator_interactive(history::EstimatorHistory)
    fig = Figure(resolution = (1200, 800))

    # Create main plot
    ax_main = Axis(fig[1, 1], title = "Model Evolution",
                   xlabel = "Trial", ylabel = "Value")

    # Control panel
    menu_var = Menu(fig[2, 1], options = ["Q", "V", "C", "W", "N"])
    toggle_exp = Toggle(fig[2, 2])
    Label(fig[2, 3], "Show Expectations")

    # Reactive plotting
    current_data = Observable(history.Qss)
    show_exp = Observable(true)

    on(menu_var.selection) do selected
        if selected == "Q"
            current_data[] = history.Qss
        elseif selected == "V"
            current_data[] = history.Vss
        elseif selected == "C"
            current_data[] = history.Css
        elseif selected == "W"
            current_data[] = history.Wss
        elseif selected == "N"
            current_data[] = Float64.(history.Nss)
        end
    end

    on(toggle_exp.active) do active
        show_exp[] = active
    end

    # Plot function
    colors = Makie.wong_colors()[1:history.n_arms]
    trials = 1:length(history.actions)

    for arm in 1:history.n_arms
        lines!(ax_main, trials, @lift($(current_data)[:, arm]),
               color = colors[arm], linewidth = 2)
    end

    fig
end

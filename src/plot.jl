using CairoMakie



"""
    Calculate moving averages of action selections

    Args:
        actions: Vector of selected actions (1-indexed)
        n_arms: Number of arms/actions
        window_size: Size of moving average window

    Returns:
        Matrix (n_trials × n_arms) of moving average selection frequencies
"""
function calculate_action_moving_averages(actions, n_arms, window_size)
    n_trials = length(actions)
    moving_averages = zeros(n_trials, n_arms)

    for t in 1:n_trials
        # Define window bounds
        start_idx = max(1, t - window_size + 1)
        end_idx = t

        # Get actions in current window
        window_actions = actions[start_idx:end_idx]
        window_length = length(window_actions)

        # Calculate frequency for each arm in current window
        for arm in 1:n_arms
            count = sum(window_actions .== arm)
            moving_averages[t, arm] = count / window_length
        end
    end

    return moving_averages
end
"""
plot the pretty whole history of the estimator
"""
function plot_estimator_history(history::EstimatorHistory;
                                figsize = (1200, 800),
                                show_expectations = true,
                                moving_avg_window = 20,
                                figure_title = "Estimator Evolution History $(history.params)")
    n_trials = length(history.actions)
    trials = 1:n_trials
    n_arms = history.n_arms

    fig = Figure(size = figsize)


    # 2x3 grid layout
    ax_q = Axis(fig[1, 1], title = "Q Values", xlabel = "Trial", ylabel = "Q")
    ax_v = Axis(fig[1, 2], title = "V=utility(Q) Values", xlabel = "Trial", ylabel = "V")
    ax_c = Axis(fig[1, 3], title = "C Values", xlabel = "Trial", ylabel = "C")
    ax_w = Axis(fig[1, 4], title = "W=βV+φC Values", xlabel = "Trial", ylabel = "W")
    ax_n = Axis(fig[2, 1], title = "N Values", xlabel = "Trial", ylabel = "N")
    ax_a = Axis(fig[2, 2], title = "Action Selection", xlabel = "Trial", ylabel = "Action Selection")
    ax_a.yticksvisible = false
    ax_a.yticklabelsvisible = false
    ax_a.ylabelvisible = false
    ax_ma = Axis(fig[2, 3], title = "Moving Average of Actions", xlabel = "Trial", ylabel = "Selection Frequency")
    # ax_r = Axis(fig[2, 3], title = "Rewards", xlabel = "Trial", ylabel = "Reward")
    ax_p = Axis(fig[2, 4], title = "Selection Probabilities", xlabel = "Trial", ylabel = "Selection Probabilities")

    Label(fig[0, :], text = figure_title,
          fontsize = 16, font = :bold, halign = :center,
          tellwidth = false, tellheight = true)

    # Display reward probabilities from the first trial's expectations
    reward_probs = history.expectations[1]
    reward_probs_str = "reward probs.: [" * join([string(round(p, digits=3)) for p in reward_probs], ", ") * "]"
    Label(fig[-1, 1:4], text = reward_probs_str,
          fontsize = 16, halign = :center,
          tellwidth = false, tellheight = true)

    # Color palette for arms
    if history.n_arms <= 7
        colors = Makie.wong_colors()[1:history.n_arms]
    else
        # For more than 7 arms, use a colormap and resample
        colors = Makie.resample_cmap(:Set1_9, history.n_arms)
    end

    # Calculate cumulative probabilities for stacking
    cumulative_probs = zeros(n_trials, n_arms + 1)
    for arm in 1:n_arms
        cumulative_probs[:, arm + 1] = cumulative_probs[:, arm] + history.Pss[:, arm]
    end

    # Plot each arm's evolution
    for arm in 1:history.n_arms
        lines!(ax_q, trials, history.Qss[:, arm],
               color = colors[arm], linewidth = 2, label = "Arm $arm")
        lines!(ax_v, trials, history.Vss[:, arm],
               color = colors[arm], linewidth = 2, label = "Arm $arm")
        lines!(ax_c, trials, history.Css[:, arm],
               color = colors[arm], linewidth = 2, label = "Arm $arm")
        lines!(ax_w, trials, history.Wss[:, arm],
               color = colors[arm], linewidth = 2, label = "Arm $arm")
        stairs!(ax_n, trials, history.Nss[:, arm],
                color = colors[arm], linewidth = 2, label = "Arm $arm")
        band!(ax_p, trials, cumulative_probs[:, arm], cumulative_probs[:, arm+1],
              color = (colors[arm]),#, alpha),
              label = "Arm $arm")
    end
    # Calculate and plot moving averages of actual action selections
    moving_averages = calculate_action_moving_averages(history.actions, n_arms, moving_avg_window)

    for arm in 1:n_arms
        lines!(ax_ma, trials, moving_averages[:, arm],
               color = colors[arm], linewidth = 2, label = "Arm $arm")
    end

    # Set y-axis limits for moving average plot
    ylims!(ax_ma, 0, 1)

    # Plot expectations overlaid on Q values if requested
    if show_expectations
        for arm in 1:history.n_arms
            expectations_arm = [exp[arm] for exp in history.expectations]
            lines!(ax_q, trials, expectations_arm,
                   color = colors[arm], linestyle = :dash,
                   linewidth = 1.5, alpha = 0.7)
        end

        # Add legend entry for expectations
        lines!(ax_q, [0], [0], color = :black, linestyle = :dash,
               linewidth = 1.5, alpha = 0.7, label = "Expectations")
    end

    # Plot rewards
    # scatter!(ax_r, trials, history.rewards,
    #          color = :black, markersize = 4, alpha = 0.6)

    # Add legends
    legend_background = (:white, 0.7)
    axislegend(ax_q, position = :rt, backgroundcolor = legend_background)
    axislegend(ax_v, position = :rt, backgroundcolor = legend_background)
    axislegend(ax_c, position = :rt, backgroundcolor = legend_background)
    axislegend(ax_w, position = :rt, backgroundcolor = legend_background)
    axislegend(ax_n, position = :rt, backgroundcolor = legend_background)

    # Highlight selected actions
    for t in trials
        action = history.actions[t]
        if action > 0  # valid action
            # Mark selected arm with vertical line
            vlines!(ax_a, t, color = colors[action], alpha = 1.0, linewidth = 1.0)
        end
    end

    fig
end


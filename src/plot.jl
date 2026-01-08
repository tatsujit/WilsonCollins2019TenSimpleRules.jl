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

using Statistics
"""
Plot performance metrics for History type (for simple policies like RandomResponding)
"""
function plot_history_performance(history::History;
                                 figsize = (1200, 800),
                                 moving_avg_window = 50,
                                 figure_title = "Performance Metrics")
    n_trials = length(history.actions)
    trials = 1:n_trials
    n_arms = history.n_arms

    fig = Figure(size = figsize)

    # Calculate performance metrics
    cumulative_rewards = cumsum(history.rewards)
    average_rewards = cumulative_rewards ./ trials
    moving_avg_rewards = zeros(n_trials)
    for t in 1:n_trials
        start_idx = max(1, t - moving_avg_window + 1)
        moving_avg_rewards[t] = Statistics.mean(history.rewards[start_idx:t])
    end
    
    # Calculate action selection frequencies
    action_frequencies = calculate_action_moving_averages(history.actions, n_arms, moving_avg_window)
    
    # Get reward probabilities (expectations)
    reward_probs = history.expectations[1]
    optimal_arm = argmax(reward_probs)
    optimal_reward = reward_probs[optimal_arm]
    
    # Calculate regret (difference from optimal)
    optimal_cumulative = optimal_reward * trials
    regret = optimal_cumulative .- cumulative_rewards

    # Create subplots: 2x3 grid
    ax_cum = Axis(fig[1, 1], title = "Cumulative Reward", xlabel = "Trial", ylabel = "Cumulative Reward")
    ax_avg = Axis(fig[1, 2], title = "Average Reward", xlabel = "Trial", ylabel = "Average Reward")
    ax_mov = Axis(fig[1, 3], title = "Moving Average Reward (window=$moving_avg_window)", xlabel = "Trial", ylabel = "Reward")
    ax_reg = Axis(fig[2, 1], title = "Cumulative Regret", xlabel = "Trial", ylabel = "Regret")
    ax_freq = Axis(fig[2, 2], title = "Action Selection Frequency", xlabel = "Trial", ylabel = "Frequency")
    ax_rew = Axis(fig[2, 3], title = "Rewards Over Time", xlabel = "Trial", ylabel = "Reward")

    Label(fig[0, :], text = figure_title,
          fontsize = 16, font = :bold, halign = :center,
          tellwidth = false, tellheight = true)

    # Display reward probabilities
    reward_probs_str = "reward probs.: [" * join([string(round(p, digits=3)) for p in reward_probs], ", ") * "]"
    Label(fig[-1, 1:3], text = reward_probs_str,
          fontsize = 16, halign = :center,
          tellwidth = false, tellheight = true)

    # Color palette for arms
    if n_arms <= 7
        colors = Makie.wong_colors()[1:n_arms]
    else
        colors = Makie.resample_cmap(:Set1_9, n_arms)
    end

    # Plot cumulative reward
    lines!(ax_cum, trials, cumulative_rewards, color = :blue, linewidth = 2, label = "Cumulative")
    lines!(ax_cum, trials, optimal_cumulative, color = :red, linestyle = :dash, linewidth = 2, label = "Optimal")
    axislegend(ax_cum, position = :lt)

    # Plot average reward
    lines!(ax_avg, trials, average_rewards, color = :blue, linewidth = 2, label = "Average")
    hlines!(ax_avg, optimal_reward, color = :red, linestyle = :dash, linewidth = 2, label = "Optimal")
    axislegend(ax_avg, position = :lt)

    # Plot moving average reward
    lines!(ax_mov, trials, moving_avg_rewards, color = :blue, linewidth = 2, label = "Moving Avg")
    hlines!(ax_mov, optimal_reward, color = :red, linestyle = :dash, linewidth = 2, label = "Optimal")
    axislegend(ax_mov, position = :lt)

    # Plot cumulative regret
    lines!(ax_reg, trials, regret, color = :orange, linewidth = 2)
    
    # Plot action selection frequencies
    for arm in 1:n_arms
        lines!(ax_freq, trials, action_frequencies[:, arm],
               color = colors[arm], linewidth = 2, label = "Arm $arm")
    end
    ylims!(ax_freq, 0, 1)
    axislegend(ax_freq, position = :rt)

    # Plot rewards over time (scatter with low alpha for density)
    scatter!(ax_rew, trials, history.rewards, 
             color = :black, markersize = 2, alpha = 0.3, label = "Rewards")
    lines!(ax_rew, trials, moving_avg_rewards, 
           color = :blue, linewidth = 2, label = "Moving Avg")
    hlines!(ax_rew, optimal_reward, color = :red, linestyle = :dash, linewidth = 2, label = "Optimal")
    axislegend(ax_rew, position = :rt)

    fig
end

"""
Plot performance metrics for multiple History types (comparison across agents)
- 5 columns: cumulative reward, average reward, moving average reward, cumulative regret, rewards over time
- N rows: N different agents
"""
function plot_history_comparison(histories::Vector{History}, 
                                labels::Vector{String};
                                figsize = (1600, 1000),
                                moving_avg_window = 50,
                                figure_title = "Performance Metrics Comparison")
    n_agents = length(histories)
    @assert length(labels) == n_agents "Number of labels must match number of histories"
    
    # Get common parameters from first history
    n_trials = length(histories[1].actions)
    trials = 1:n_trials
    reward_probs = histories[1].expectations[1]
    optimal_arm = argmax(reward_probs)
    optimal_reward = reward_probs[optimal_arm]
    optimal_cumulative = optimal_reward * trials

    fig = Figure(size = figsize)

    # fig[0]: Environment info (reward probabilities)
    reward_probs_str = "reward probs.: [" * join([string(round(p, digits=3)) for p in reward_probs], ", ") * "]"
    Label(fig[0, :], text = reward_probs_str,
          fontsize = 16, halign = :center,
          tellwidth = false, tellheight = true)

    # fig[1]: Plot title
    Label(fig[1, :], text = figure_title,
          fontsize = 16, font = :bold, halign = :center,
          tellwidth = false, tellheight = true)

    # Create axes: N rows x 5 columns (starting from row 3)
    axes_grid = Matrix{Axis}(undef, n_agents, 5)
    for row in 1:n_agents
        plot_row = row + 2  # Offset by 2 (for fig[0] and fig[1] and fig[2])
        axes_grid[row, 1] = Axis(fig[plot_row, 1], title = row == 1 ? "Cumulative Reward" : "",
                                 xlabel = row == n_agents ? "Trial" : "", ylabel = "")
        axes_grid[row, 2] = Axis(fig[plot_row, 2], title = row == 1 ? "Average Reward" : "",
                                 xlabel = row == n_agents ? "Trial" : "", ylabel = "")
        axes_grid[row, 3] = Axis(fig[plot_row, 3], title = row == 1 ? "Moving Avg Reward" : "",
                                 xlabel = row == n_agents ? "Trial" : "", ylabel = "")
        axes_grid[row, 4] = Axis(fig[plot_row, 4], title = row == 1 ? "Cumulative Regret" : "",
                                 xlabel = row == n_agents ? "Trial" : "", ylabel = "")
        axes_grid[row, 5] = Axis(fig[plot_row, 5], title = row == 1 ? "Rewards Over Time" : "",
                                 xlabel = row == n_agents ? "Trial" : "", ylabel = "")

        # Add row label (model/policy name) at column 0
        Label(fig[plot_row, 0], text = labels[row],
              fontsize = 12, rotation = π/2, halign = :center, valign = :center,
              tellwidth = true, tellheight = false)
    end

    # Set equal heights for all plot rows and equal widths for all columns
    for row in 1:n_agents
        rowsize!(fig.layout, row + 2, Relative(1/n_agents))
    end
    for col in 1:5
        colsize!(fig.layout, col, Relative(1/5))
    end

    # Color palette for different agents
    if n_agents <= 7
        agent_colors = Makie.wong_colors()[1:n_agents]
    else
        agent_colors = Makie.resample_cmap(:Set1_9, n_agents)
    end

    # Plot for each agent
    for (agent_idx, history) in enumerate(histories)
        # Calculate performance metrics
        cumulative_rewards = cumsum(history.rewards)
        average_rewards = cumulative_rewards ./ trials
        moving_avg_rewards = zeros(n_trials)
        for t in 1:n_trials
            start_idx = max(1, t - moving_avg_window + 1)
            moving_avg_rewards[t] = Statistics.mean(history.rewards[start_idx:t])
        end

        # Calculate regret
        regret = optimal_cumulative .- cumulative_rewards

        # Get axes for this row
        ax_cum = axes_grid[agent_idx, 1]
        ax_avg = axes_grid[agent_idx, 2]
        ax_mov = axes_grid[agent_idx, 3]
        ax_reg = axes_grid[agent_idx, 4]
        ax_rew = axes_grid[agent_idx, 5]

        color = agent_colors[agent_idx]
        label = labels[agent_idx]

        # Plot cumulative reward
        lines!(ax_cum, trials, cumulative_rewards, color = color, linewidth = 2, label = label)
        lines!(ax_cum, trials, optimal_cumulative, color = :red, linestyle = :dash, linewidth = 2, label = "Optimal result")

        # Plot average reward
        lines!(ax_avg, trials, average_rewards, color = color, linewidth = 2, label = label)
        hlines!(ax_avg, optimal_reward, color = :red, linestyle = :dash, linewidth = 2, label = "Optimal result")
        ylims!(ax_avg, 0, 1)

        # Plot moving average reward
        lines!(ax_mov, trials, moving_avg_rewards, color = color, linewidth = 2, label = label)
        hlines!(ax_mov, optimal_reward, color = :red, linestyle = :dash, linewidth = 2, label = "Optimal result")
        ylims!(ax_mov, 0, 1)

        # Plot cumulative regret
        lines!(ax_reg, trials, regret, color = color, linewidth = 2, label = label)
        hlines!(ax_reg, 0.0, color = :red, linestyle = :dash, linewidth = 2, label = "Optimal result")

        # Plot rewards over time (scatter with low alpha for density)
        scatter!(ax_rew, trials, history.rewards,
                 color = color, markersize = 1.5, alpha = 0.2, label = label)
        lines!(ax_rew, trials, moving_avg_rewards,
               color = color, linewidth = 2)
        hlines!(ax_rew, optimal_reward, color = :red, linestyle = :dash, linewidth = 2, label = "Optimal result")
        ylims!(ax_rew, 0, 1)
    end

    # fig[2]: Legend
    legend_elements = [
        [LineElement(color = agent_colors[i], linewidth = 2) for i in 1:n_agents]...,
        LineElement(color = :red, linestyle = :dash, linewidth = 2)
    ]
    legend_labels = [labels..., "Optimal result"]
    Legend(fig[2, :],
           legend_elements,
           legend_labels,
           orientation = :horizontal, tellwidth = false, tellheight = true,
           framevisible = false, nbanks = 1)

    fig
end


include("utils/sampling.jl")
include("utils/sleeping_arms.jl")

"""
TODO: calculation of performance metrics integrated on DataFrame (BDF format)
calculate the average reward for each trial over the histories
"""
function calculate_average_reward(systems::Vector{System}, trials::Int)::Vector{Float64}
    # make a matrix of systems.history.rewards
    all_rewards = [sys.history.rewards for sys in systems]
    # println("all_rewards: ", all_rewards)
    average_reward = zeros(trials)
    n_systems = length(systems)
    for t in 1:trials
        total_reward = 0.0
        for i in 1:length(systems)
            total_reward += all_rewards[i][t]
        end
        average_reward[t] = total_reward / n_systems
    end
    average_reward
end

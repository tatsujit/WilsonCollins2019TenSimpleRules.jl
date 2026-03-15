"""
    prob_stay(actions, rewards) -> Vector{Float64}

Calculate the probability of staying on the same arm given the actions and rewards.

# Arguments
- `actions::Vector{Int}`: actions
- `rewards::Vector{Float64}`: rewards

# Returns
- `Vector{Float64}`: probability of staying on the same arm given the actions and rewards
"""
function prob_stay(actions, rewards)
    counts = [0, 0] # count for staying when reward is 0, and 1
    indices = [findall(rewards .== 0.0), findall(rewards .== 1.0)]
    for i in 1:2 # reward is 0 or 1
        for idx in indices[i]
            if idx < length(actions)
                if actions[idx] == actions[idx+1]
                    counts[i] += 1
                end
            end
        end
    end
    return counts ./ length.(indices)
end
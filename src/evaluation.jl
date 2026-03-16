"""
    prob_stay(actions, rewards) -> Vector{Float64}

Calculate the probability of staying on the same arm given the actions and rewards.
Note: Bernoulli reward assumed. 

# Arguments
- `actions::Vector{Int}`: actions
- `rewards::Vector{Float64}`: rewards

# Returns
- `Vector{Float64}`: probability of staying on the same arm given the actions and rewards

  [probability of staying at the same action when reward is 0.0, 
   probability of staying at the same action when reward is 1.0, 
  ]
"""
function prob_stay(actions, rewards)
    stay_counts = [0, 0] # count for staying when reward is 0, and 1
    indices = [findall(rewards .== 0.0), findall(rewards .== 1.0)]

    # the last action in the sequence has to be excluded from the calculation
    last_action = actions[end]
    indices_length = length.(indices)
    indices_length[last_action] = indices_length[last_action] - 1
    
    for i in 1:2 # reward is 0 or 1
        for idx in indices[i]
            if idx < length(actions) # last action excluded
                if actions[idx] == actions[idx+1]
                    stay_counts[i] += 1
                end
            end
        end
    end
    return @. stay_counts / indices_length
    # return (stay_counts, indices_length, stay_counts ./ indices_length)
end
function estimator_history_validity(eh::EstimatorHistory)
    # Check if the history is valid
    if eh.n_arms <= 0
        return false, "Invalid number of arms: $(eh.n_arms)"
    end
    if eh.n_avail_arms <= 0
        return false, "Invalid number of available arms: $(eh.n_avail_arms)"
    end
    if length(eh.actions) == 0
        return false, "No actions recorded in history"
    end
    if length(eh.expectations) != length(eh.actions)
        return false, "Expectations length does not match actions length"
    end
    if length(eh.rewards) != length(eh.actions)
        return false, "Rewards length does not match actions length"
    end
    params = eh.params
    (eh.Css .== params.C0,
     eh.Qss[1,:] .== params.Q0,
        eh.Wss .== params.β * eh.Vss .+ params.φ * eh.Css,
        eh.Qss .== eh.Vss) |> (all, all, all) || return false, "Inconsistent values in history",



    return true, "History is valid"
end

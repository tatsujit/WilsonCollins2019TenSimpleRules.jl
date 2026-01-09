"""
There are only History and LAHistory (no NHistory nor NLAHistory)
because they are not necessary.
"""
abstract type AbstractHistory end

mutable struct History <: AbstractHistory
    n_arms::Int
    trials::Int
    actions::Vector{Int}
    expectations::Vector{Vector{Float64}} # TODO 非定常環境を想定して、毎trialの期待値を保存する？　
    # TODO しかし、等平均で異分散の状況を扱いたくないか？　とくにリスク態度に関して
    rewards::Vector{Float64}
    function History(trials::Int, n_arms::Int)
        actions = zeros(Int64, trials)
        expectations = [zeros(n_arms) for _ in 1:trials]
        rewards = zeros(Float64, trials)
        new(n_arms, trials, actions, expectations, rewards)
    end
end
function dataframe(history::AbstractHistory)
    DataFrame(Trial = 1:history.trials,
              Action = history.actions,
              Expectations = history.expectations,
              Reward = history.rewards)
end
function dataframe(history::AbstractHistory, case::Int)
    DataFrame(Case = case,
              Trial = 1:history.trials,
              Action = history.actions,
              Expectations = history.expectations,
              Reward = history.rewards)
end
function dataframe(histories::Vector{AbstractHistory})
    n_histories = length(histories)
    df = DataFrame()
    for i in 1:n_histories
        df = vcat(df,
                  DataFrame(Case = i,
                            Action = histories[i].actions,
                            Expectations = histories[i].expectations,
                            Reward = histories[i].rewards))
    end
    return df
end


function print(history::History, verbose::Bool=false)
    trials = length(history.actions)
    if verbose
        println("n_arms: $(n_arms)")
        println("trials: $(trials)")
        println("History: actions=$(history.actions)")
        println("expectations=$(history.expectations)")
        println("rewards=$(history.rewards)")
    else
        println("trials: $(trials)")
    end
end
function record!(history::History, trial::Int, action::Int, expectations::Vector{Float64}, reward::Float64)
    n_arms = history.n_arms # length(expectations) #
    history.actions[trial] = action
    for i in 1:n_arms
        history.expectations[trial][i] = expectations[i]
    end
    # history.expectations[trial] = expectations
    history.rewards[trial] = reward
end

# Overload to accept estimator argument (ignored for History type)
function record!(history::History, trial::Int, action::Int, expectations::Vector{Float64}, reward::Float64, estimator::AbstractEstimator)
    record!(history, trial, action, expectations, reward)
end
mutable struct EstimatorHistory <: AbstractHistory
    n_arms::Int
    actions::Vector{Int}
    expectations::Vector{Vector{Float64}}
    rewards::Vector{Float64}
    params::NamedTuple # 追加: パラメータの記録
    Qss::Matrix{Float64} # 追加: 各 trial の Q 値の記録
    Vss::Matrix{Float64} # 追加: 各 trial の V 値の記録
    Wss::Matrix{Float64} # 追加: 各 trial の W 値の記録
    Css::Matrix{Float64} # 追加: 各 trial の C 値の記録
    Nss::Matrix{Int64} # 追加: 各 trial の N 値の記録
    Pss::Matrix{Float64} # selection_probabilities
    function EstimatorHistory(trials::Int, n_arms::Int, est::Estimator)
        actions = zeros(Int64, trials)
        expectations = [zeros(n_arms) for _ in 1:trials]
        rewards = zeros(Float64, trials)
        params = parameters(est) # 追加: パラメータの記録
        Qss = zeros(Float64, trials, n_arms)
        Vss = zeros(Float64, trials, n_arms)
        Css = zeros(Float64, trials, n_arms)
        Wss = zeros(Float64, trials, n_arms)
        Nss = zeros(Int64, trials, n_arms)
        Pss = zeros(Float64, trials, n_arms)
        new(n_arms, actions, expectations, rewards, params,
            Qss, Vss, Css, Wss, Nss, Pss)
    end
end

function record!(history::EstimatorHistory, trial::Int, action::Int,
                 expectations::Vector{Float64}, reward::Float64,
                 est::Estimator=nothing)
    n_arms = history.n_arms # length(expectations) #
    history.actions[trial] = action
    # @ic model
    sel_probs = selection_probabilities(est) # with Softmax(β=1.0)
    for i in 1:n_arms
        history.expectations[trial][i] = expectations[i]
        history.Qss[trial, i] = est.Q[i]
        history.Vss[trial, i] = est.V[i]
        history.Css[trial, i] = est.C[i]
        history.Wss[trial, i] = est.W[i]
        history.Nss[trial, i] = est.N[i]
        history.Pss[trial, i] = sel_probs[i]
    end
    # history.expectations[trial] = expectations
    history.rewards[trial] = reward
end

struct LAHistory <: AbstractHistory
    n_arms::Int
    n_avail_arms::Int
    actions::Vector{Int}
    available_arms::Vector{Vector{Int}} # 追加: 利用可能な腕の記録
    expectationss::Vector{Vector{Float64}}
    rewards::Vector{Float64}
end
function LAHistory(n_arms::Int, n_avail_arms::Int, expected_trials::Int=0)
    actions =       Vector{Int}(undef, 0)
    available_arms =Vector{Vector{Float64}}()
    expectationss = Vector{Vector{Float64}}()
    rewards =       Vector{Float64}(undef, 0)
    if expected_trials > 0
        sizehint!(actions, expected_trials)
        sizehint!(expectationss, expected_trials)
        sizehint!(rewards, expected_trials)
        sizehint!(available_arms, expected_trials)
    end
    LAHistory(n_arms, n_avail_arms, actions, available_arms, expectationss, rewards)
end
function print(history::LAHistory)
    trials = length(history.actions)
    println("n_arms: $(n_arms)")
    println("n_avail_arms: $(n_avail_arms)")
    println("trials: $(trials)")
    println("History: ")
    println("actions=$(history.actions)")
    println("available_arms=$(history.available_arms)")
    println("expectationss=$(history.expectationss)")
    println("rewards=$(history.rewards)")
end
# function record!(history::HistoryO, trial::Int, action::Int, expectations::Vector{Float64}, reward::Float64, Qs::Vector{Float64})
#     push!(history.actions, action)
#     push!(history.expectations, expectations)
#     push!(history.rewards, reward)
# end
function record!(history::LAHistory, trial::Int, action::Int, available_arms::Vector{Int},
                 expectations::Vector{Float64}, reward::Float64, verbose::Bool=false)
    if verbose
        println("Qs: $(Qs)")
        println("history.Qss: $(history.Qss)")
    end
    push!(history.actions, action)
    push!(history.available_arms, copy(available_arms))
    push!(history.rewards, reward)
    push!(history.expectationss, copy(expectations))
end


abstract type AbstractSystem end

struct System <: AbstractSystem
    agent::Agent
    env::AbstractEnvironment
    history::AbstractHistory
    rng::AbstractRNG  # Add RNG to System
end

"""Run the Simulation for the system for the given number of trials"""
function run!(system::System, trials::Int, verbose::Bool=false)
    # @ic system.rng
    for trial in 1:trials
        step!(system, trial, verbose)
    end
end

"""The step function (t → t+1) for the System"""
function step!(system::System, trial::Int, verbose::Bool=false)
    # variable localization
    env = system.env
    agent = system.agent
    history = system.history
    estimator = agent.estimator
    policy = agent.policy
    rng = system.rng
    n_arms = system.env.n_arms
    expectations = mean(env) # calculate the expectations for the current environment

    action = select_action(policy, estimator; rng=rng)
    reward = sample_reward(env, action; rng=rng)
    record!(history, trial, action, expectations, reward, estimator)
    update!(estimator, action, reward)

    ################################################################
    # *for sleeping environments*
    ################################################################
    # if typeof(env) != LAEnvironment && typeof(env) != NLAEnvironment
    #     1
    # else # for regular (all actions always available) environments
    #     sample_available_arms!(system) # limited available arms をここで決める
    #     # shuffle_available_arms!(system)
    #     available_arms = env.available_arms
    #     action = select_action(policy, estimator, available_arms; rng)
    #     reward = sample_reward(env, action, rng=rng)
    #     record!(history, trial, action, available_arms, expectations, reward)
    #     update!(estimator, action, reward) # ここでも utility_function を入れるか
    # end
end


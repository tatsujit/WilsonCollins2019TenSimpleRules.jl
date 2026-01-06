abstract type AbstractEnvironment end

using Random, Distributions, StatsBase

"""
Environment: stationary environment
"""
struct Environment <: AbstractEnvironment
    n_arms::Int
    distributions::AbstractVector{UnivariateDistribution}
    function Environment(distributions)#::AbstractVector{Distribution{Univariate}})
        n_arms = length(distributions)
        new(n_arms, distributions)
    end
    function Environment(rewardProbs::Vector{Float64})
        n_arms = length(rewardProbs)
        distributions = UnivariateDistribution[Bernoulli(p) for p in rewardProbs]
        new(n_arms, distributions)
    end
    """
    when given only n_arms::Int, Bernoullis with uniformly partitioning (0, 1) into (n_arms+1) intervals
    e.g. n_arms=3 ならば、[0.25, 0.5, 0.75] となる
    """
    function Environment(n_arms::Int)
        ps = [x / (n_arms + 1) for x in 1:n_arms]
        Environment(ps)
        # distributions = [Bernoulli(x / (n_arms+1)) for x in 1:n_arms]
        # new(distributions)
    end
end

"""as a Bernoulli returns a Boolean"""
function bool_to_float(x::Number)::Float64 # as Bool <: Number
    if isa(x, Bool)
        return convert(Float64, x)
    elseif isa(x, Float64)
        return x
    else
        throw(ArgumentError("Input must be Bool or Float64"))
    end
end

function sample_reward(env::Environment, arm::Int; rng::AbstractRNG=Random.default_rng())
    return bool_to_float(rand(rng, env.distributions[arm]))
end

# """
# arm ∈ available_arms であることを前提にし、ここでは単純に処理している
# """
# function sample_reward(env::LAEnvironment, arm::Int; rng::AbstractRNG=Random.default_rng())
#     return bool_to_float(rand(rng, env.distributions[arm]))
# end

# """
# non-stationary なので、毎 trial 異なる distributions を参照しうる。
# mutable にして、 trial 情報を持っておいて、 sample_reward! する度に increment する。
# trial += 1 することにより、破壊的になるので sample_reward ではなく sample_reward! としている

# """
# function sample_reward(env::NEnvironment, arm::Int; rng::AbstractRNG=Random.default_rng())
#     reward = bool_to_float(rand(rng, env.distributionsS[env.t][arm]))
#     env.t += 1
#     return reward
# end

# """
# non-stationary なので、毎 trial 異なる distributions を参照しうる。
# mutable にして、 trial 情報を持っておいて、 sample_reward! する度に increment する。
# trial += 1 することにより、破壊的になるので sample_reward ではなく sample_reward! としている
# """
# function sample_reward(env::NLAEnvironment, arm::Int; rng::AbstractRNG=Random.default_rng())
#     reward = bool_to_float(rand(rng, env.distributionsS[env.t][arm]))
#     env.t += 1
#     return reward
# end

function mean(b::Bernoulli{Float64})
    return b.p  # Bernoulli distributionの平均は成功確率
end
"""
mean for various distributions
"""
function mean(env::Environment)
    return mean.(env.distributions)  # broadcast版
end

"""
Non-stationary environment with changing reward distributions
"""
mutable struct NonStationaryEnvironment <: AbstractEnvironment
    n_arms::Int
    current_trial::Int
    change_points::Vector{Int}
    distribution_phases::Vector{Vector{UnivariateDistribution}}
    
    function NonStationaryEnvironment(initial_probs::Vector{Float64}, 
                                    change_points::Vector{Int},
                                    phase_probs::Vector{Vector{Float64}})
        n_arms = length(initial_probs)
        
        # Create distribution phases
        distribution_phases = Vector{Vector{UnivariateDistribution}}()
        
        # Initial phase
        push!(distribution_phases, [Bernoulli(p) for p in initial_probs])
        
        # Subsequent phases
        for probs in phase_probs
            push!(distribution_phases, [Bernoulli(p) for p in probs])
        end
        
        new(n_arms, 1, change_points, distribution_phases)
    end
end

function get_current_distributions(env::NonStationaryEnvironment)
    # Determine which phase we're in based on current trial
    phase_index = 1
    for (i, change_point) in enumerate(env.change_points)
        if env.current_trial >= change_point
            phase_index = i + 1
        end
    end
    
    return env.distribution_phases[min(phase_index, length(env.distribution_phases))]
end

function sample_reward(env::NonStationaryEnvironment, arm::Int; rng::AbstractRNG=Random.default_rng())
    current_dists = get_current_distributions(env)
    reward = bool_to_float(rand(rng, current_dists[arm]))
    env.current_trial += 1
    return reward
end

function mean(env::NonStationaryEnvironment)
    current_dists = get_current_distributions(env)
    return [d.p for d in current_dists]
end

function reset_trial_counter!(env::NonStationaryEnvironment)
    env.current_trial = 1
end

# """
# ERROR: This doesn't work. Refer to
# - bandit1/test/distributions_test.jl
# - ~/Dropbox/org/_roam/20250629112859-julia_distributions_jl_配列の非変性_invariance.org
# """
# function mean(env::AbstractEnvironment)
#     if env.distributions isa AbstractVector{Bernoulli{Float64}}
#         # expectations are the probabilities of success for each arm
#         expectations = [env.distributions[i].p for i in 1:n_arms]
#     elseif env.distributions isa Vector{Normal{Float64}}
#         # expectations are the means of the normal distributions for each arm
#         expectations = [env.distributions[i].μ for i in 1:n_arms]
#     elseif env.distributions isa Vector{Uniform{Float64}}
#         expectations = [mean(env.distributions[i]) for i in 1:n_arms]
#     else
#         throw(ArgumentError("Unsupported distribution type in environment"))
#     end
#     return expectations
# end

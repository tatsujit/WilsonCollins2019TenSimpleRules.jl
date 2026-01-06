struct RandomResponding <: AbstractPolicy
    probs::Vector{Float64}  # 各アームの選択確率
    function RandomResponding(n_arms::Int) # デフォルトは均等分布
        return new(ones(Float64, n_arms) ./ n_arms)
    end
end

function selection_probabilities(policy::RandomResponding, estimator::AbstractActionValueEstimator)
    return policy.probs
end

function select_action(policy::RandomResponding, estimator::AbstractActionValueEstimator; rng::AbstractRNG=Random.default_rng())
    n_arms = length(policy.probs)
    return sample(rng, 1:n_arms, Weights(policy.probs))
end

rr1 = RandomResponding(5)
selection_probabilities(rr1, Estimator(5))

using StatsBase
cs = [select_action(s1, RandomResponding(5)) for _ in 1:1000]
countmap(cs)

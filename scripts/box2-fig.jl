using DrWatson
@quickactivate "WilsonCollins2019TenSimpleRules.jl"

# Here you may include files from the source directory
using MultiBandits 
using Random, DataFrames
include(srcdir("plots.jl"))

#include(srcdir("_setup.jl"))
rewardProbs = [0.2, 0.8]
n_arms = length(rewardProbs)
trials = 1_000
rseed = 2345

# common 
env = Environment(rewardProbs)

# Create 3 different RandomResponding agents with different parameters
probs_list = [
    [0.5, 0.5],      # uniform random
    rewardProbs,     # matching reward probabilities
    [0.0, 1.0]       # always choose arm 2 (optimal)
]
n_agents = length(probs_list)
labels = ["Uniform Random [0.5, 0.5]", "Matching Reward Probs [0.2, 0.8]", "Optimal [0.0, 1.0]"]

histories = Vector{History}(undef, n_agents)
for (i, probs) in enumerate(probs_list)
    pol = RandomResponding(probs)
    est = EmptyEstimator()
    agent = Agent(pol, est)
    history = History(n_arms, trials)
    rng = Random.MersenneTwister(rseed + i)  # Different seed for each agent
    system = System(agent, env, history; rng=rng)
    run!(system, trials)
    histories[i] = history
end

# 比較プロットを作成 (3x5)
fig_comparison = plot_history_comparison(histories, labels, 
                                        figure_title = "Random Responding Comparison")
display(fig_comparison)

# est = Estimator(n_arms)
# pol = SoftmaxPolicy()
# agent = Agent(pol, est)
# env = Environment(n_arms)
# history = EstimatorHistory(trials, n_arms, est)
# rng = Random.MersenneTwister(42)
# system = System(agent, env, history, rng)

# println(history)

# # 包括的なプロットを作成
# fig = plot_estimator_history(history)
# display(fig)



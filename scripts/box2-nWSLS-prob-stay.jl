using DrWatson
@quickactivate "WilsonCollins2019TenSimpleRules.jl"

# Here you may include files from the source directory
using MultiBandits 
using Random, DataFrames, Statistics
include(srcdir("plots.jl"))
include(srcdir("evaluation.jl"))

#include(srcdir("_setup.jl"))
rewardProbs = [0.2, 0.8]
n_arms = length(rewardProbs)
trials = 1_000
rseed = 2345

# common 
env = Environment(rewardProbs)

# Create 3 different RandomResponding agents with different parameters
ϵs = [
    0.0, 
    0.1, 
    0.5, 
    1.0, 
]
n_agents = length(ϵs)
labels = ["ϵ=$ϵ" for ϵ in ϵs]
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

# compute p(stay) from a history
his = histories[1]
as = his.actions
rs = his.rewards

p_stay = prob_stay(his.actions, his.rewards)
p_stays = [prob_stay(his.actions, his.rewards) for his in histories]

fig = Figure(size=(400, 400))
ax = Axis(fig[1, 1])
box2_fig1A!(ax, histories, labels)
# 比較プロットを作成 (3x5)
fig
fig

safesave(plotsdir("box2-fig1A-prob-stay.png"), fig)

#= fig_comparison = plot_history_comparison(histories, labels, 
                                        figure_title = "Random Responding Comparison")
display(fig_comparison)
 =#
# 包括的なプロットを作成
# fig = plot_estimator_history(histories[1])
# display(fig)



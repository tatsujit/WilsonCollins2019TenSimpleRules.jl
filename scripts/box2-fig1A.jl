using DrWatson
@quickactivate "WilsonCollins2019TenSimpleRules.jl"

# Here you may include files from the source directory
using MultiBandits 
using Random, DataFrames
using LaTeXStrings
include(srcdir("plots.jl"))

#include(srcdir("_setup.jl"))
sims = 10
rewardProbs = [0.2, 0.8]
n_arms = length(rewardProbs)
trials = 1_000
rseed = 2345

# previous reward によって、その腕をもう一度選ぶかどうかの確率がどうなるのかのグラフである。
# これを計算するためには、 actions, rewards があればよい。
# それらは history にある。

config_seed = Dict(
    :model => ["random", "WSLS", "RW", "CK", "RW+CK"], 
    :rseed => [rseed + i for i in 1:sims], 
    :rewardProbs => Tuple(rewardProbs), # made a tuple to prevent vector expansion
    :n_arms => n_arms,
    :trials => trials,
)



# common 
env = Environment(rewardProbs)

# Create a RandomResponding agent with different parameters
probs = [0.2, 0.8]
pol = RandomResponding(probs)
est = EmptyEstimator()
agent = Agent(pol, est)
history = History(n_arms, trials)
system = System(agent, env, history; rng=Random.MersenneTwister(rseed))
run!(system, trials)
using DrWatson
@quickactivate "WilsonCollins2019TenSimpleRules.jl"

# Here you may include files from the source directory
include(srcdir("_setup.jl"))

n_arms = 10
trials = 100

est = Estimator(n_arms)
pol = SoftmaxPolicy()
agent = Agent(pol, est)
env = Environment(n_arms)
history = EstimatorHistory(trials, n_arms, est)
rng = Random.MersenneTwister(42)
system = System(agent, env, history, rng)

run!(system, trials)
println(history)

# 包括的なプロットを作成
fig = plot_estimator_history(history)
display(fig)



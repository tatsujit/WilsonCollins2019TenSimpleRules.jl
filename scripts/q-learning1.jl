using DrWatson
@quickactivate "WilsonCollins2019TenSimpleRules.jl"

# Here you may include files from the source directory
include(srcdir("_setup.jl"))

n_arms = 10
trials = 100
typeof(n_arms)
est = Estimator(n_arms)
pol = SoftmaxPolicy()
agent = Agent(pol, est)
env = Environment(n_arms)
history = History(n_arms, trials)
rng = Random.MersenneTwister(42)
system = System(agent, env, history, rng)

run!(system, trials)

println(history)



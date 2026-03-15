################################################################
# initial setting
################################################################
# setting the DrWatson.jl simulation managament library
using DrWatson
# activate this project 
@quickactivate "GoalDirectedTS.jl"
# define this variable so that the data directory has its name 
# TODO: make it automatic, the filename substituted to `program_name` as on Emacs
program_name = "STEP_test_low_reward_probs"

################################################################
# parameters
################################################################
sims = 10
trials = 100
rseed = 1234
n_arms = 9

# parameter set
config_seed = Dict(
    :n_arms => [n_arms],
    :trials => trials,
    :rseed => [rseed + i for i in 1:sims], 
    :aspiration => collect(0.0:0.01:1.0), 
    :reward_probs_uniform => false, 
)

# all the combinations of the parameter set 
configs = dict_list(config_seed)


################################################################
# the main computation defined
################################################################
"""
    the main compute function: 
    result = simulate(config::Dict)::Dict
"""
function simulate(config::Dict)
    @unpack n_arms, trials, rseed, aspiration, reward_probs_uniform = config
    # ...
    # simulation run 
    run!(system, trials, true)
    @unpack n_arms, trials, actions, expectations, rewards = history
    return @strdict n_arms trials rseed actions expectations rewards aspiration
end

################################################################
# actual computation 
################################################################
# 各シミュレーションを処理するループ    
Threads.@threads for config in configs
    produce_or_load(simulate,
                    config,
                    datadir(program_name),
                    ;
                    #                force = true, # 「もう計算してあっても計算し直して上書きするか」
                    )
end
# (value = nothing, time = 43.905049334, bytes = 408139560, gctime = 0.0, gcstats = Base.GC_Diff(408139560, 280, 0, 7115591, 237, 0, 0, 0, 0))


################################################################
# data integration into a dataframe
################################################################
data = collect_results!( # only available when using DataFrames
    scriptsdir(program_name * ".jld2"), # where to save the collected results
    datadir(program_name); # where to collect the individual results
    update = true,
    verbose = true,
)

################################################################
# data analysis
################################################################

# ...
# 各 row から n_arms, TS_sample_size, regret（累積リグレットの時系列ベクトル）を格納
result_df = DataFrame(
    n_arms = Int[],
    # ...
)

for row in eachrow(data)
    cr = cumulative_regret(row.actions, row.expectations)
    # ...
    push!(result_df, (row.n_arms, row.trials, cr, row.aspiration, rd, ard))
end

sort!(result_df, [:n_arms, :aspiration])

################################################################
# plot
################################################################
# TS_sample_size ごとの regret 時間発展を n_arms 別にプロット
using CairoMakie

# ...

# savepath = plotsdir(program_name * ".pdf")
# safesave(savepath, fig)


# using Pkg
# Pkg.add(["Random", "DataFrames", "CSV", "Distributions", "StatsBase", "Statistics", 
#          "Optim", "Distributed", "IceCream", "ProgressMeter", "YAML", "SpecialFunctions"])

using Random
using DataFrames
using CSV
using Distributions
using StatsBase
using Optim
using Distributed
using IceCream
# path = "src/"
path = "core/"
include(path * "actionValueEstimator.jl")
include(path * "policy.jl")
include(path * "agent.jl")
include(path * "environment.jl")
include(path * "history.jl")
include(path * "system.jl")
include(path * "estimator.jl")
# include(path * "evaluation.jl")
# include(path * "heatmap.jl")
# include(path * "plot.jl")
# include(path * "_utils.jl")
include("_utils.jl")
# include(path * "parameter-estimation.jl")
include("estimate_parameters_mle.jl")
include("plot.jl")

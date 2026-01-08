# using Pkg
# Pkg.add("Random")
# Pkg.add("DataFrames")
# Pkg.add("CSV")
# Pkg.add("Distributions")
# Pkg.add("StatsBase")
# Pkg.add("Optim")
# Pkg.add("Distributed")
# Pkg.add("IceCream")
# Pkg.add("ProgressMeter")
# Pkg.add("YAML")
# Pkg.add("SpecialFunctions")
# add Random,  DataFrames,  Distributions,  StatsBase,  Optim,  Distributed,  IceCream

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

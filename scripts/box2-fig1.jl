using DrWatson
@quickactivate "WilsonCollins2019TenSimpleRules.jl"

# Here you may include files from the source directory
using MultiBandits 
using Random, DataFrames
using LaTeXStrings
include(srcdir("plots.jl"))

#include(srcdir("_setup.jl"))
rewardProbs = [0.2, 0.8]
n_arms = length(rewardProbs)
trials = 1_000
rseed = 2345

(titlesize, xlabelsize, ylabelsize) = (32, 24, 24)


fig = Figure(size=(1100, 400))

axA = Axis(fig[1, 1], 
    title = "stay behavior",
    xlabel = "previous reward",
    ylabel = L"p(\text{stay})",
    titlesize = titlesize,
    xlabelsize = xlabelsize,
    ylabelsize = ylabelsize,
)

axB1 = Axis(fig[1, 2], 
    title = "early trials",
    xlabel = "learning rate, previous reward",
    ylabel = L"p(\text{switch})",
    titlesize = titlesize,
    xlabelsize = xlabelsize,
    ylabelsize = ylabelsize,
)
axB2 = Axis(fig[1, 3])

# パネルラベル
Label(fig[1, 1, TopLeft()], "A",
    fontsize = 32,
    font = :bold,
    padding = (0, 10, 10, 0),  # (left, right, top, bottom)
    halign = :left
)

Label(fig[1, 2, TopLeft()], "B",
    fontsize = 32,
    font = :bold,
    padding = (0, 10, 10, 0),
    halign = :left
)

lines!(ax, 1:trials, 1:trials)
fig |> display


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





# font size for titles, xlabels, ylabels
(titlesize, xlabelsize, ylabelsize) = (32, 24, 24)

fig = Figure(size=(1100, 400))

axA = Axis(fig[1, 1], 
    title = "stay behavior",
    xlabel = L"\text{previous reward}",
    ylabel = L"p(\text{stay})",
    titlesize = titlesize,
    xlabelsize = xlabelsize,
    ylabelsize = ylabelsize,
)

axB1 = Axis(fig[1, 2], 
    title = "early trials",
    xlabel = L"\text{learning rate, }\alpha",
    ylabel = L"p(\text{correct})",
    titlesize = titlesize,
    xlabelsize = xlabelsize,
    ylabelsize = ylabelsize,
)

axB2 = Axis(fig[1, 3],
    title = "late trials",
    xlabel = L"\text{learning rate, }\alpha",
    titlesize = titlesize,
    xlabelsize = xlabelsize,
    ylabelsize = ylabelsize,
)

panel_label_size = 48
# パネルラベル
Label(fig[1, 1, TopLeft()], "A",
    fontsize = panel_label_size,
    font = :bold,
    padding = (0, 10, 10, 0),  # (left, right, top, bottom)
    halign = :left
)

Label(fig[1, 2, TopLeft()], "B",
    fontsize = panel_label_size,
    font = :bold,
    padding = (0, 10, 10, 0),
    halign = :left
)

# A と B の間を広げる
colgap!(fig.layout, 1, 50)

lines!(axA, 1:trials, 1:trials)

# save and display
fig |> display


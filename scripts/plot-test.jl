using CairoMakie

xs = 1:10
ys = xs .^ 2

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, xs, ys)
fig |> display


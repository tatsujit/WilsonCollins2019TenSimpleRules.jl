
using CairoMakie, DataFrames
"""
TODO: (nullでない) パラメータのリストを NamedTuple として持っておくと便利
TODO: パラメータ組みについて適切に表示できるようにする！
"""
function plot_parameter_comparison(dataframes::Vector{DataFrame}; marker_size=8)
    # 2x5のサブプロット配置
    fig = Figure(size=(1200, 600))

    n_dfs = length(dataframes)
    n_rows = 2
    n_cols = 5

    # 各DataFrameに対してサブプロットを作成
    for i in 1:min(n_dfs, n_rows * n_cols)
        row = div(i - 1, n_cols) + 1
        col = mod(i - 1, n_cols) + 1

        ax = Axis(fig[row, col],
                  title="DataFrame $i",
                  xlabel="Sample",
                  ylabel="Value")

        df = dataframes[i]
        x = 1:nrow(df)

        # 真値（横線）
        hlines!(ax, df.α[1], color=:red, linestyle=:solid, linewidth=2, label="α (true)")
        hlines!(ax, df.β[1], color=:blue, linestyle=:solid, linewidth=2, label="β (true)")
        hlines!(ax, df.αf[1], color=:green, linestyle=:solid, linewidth=2, label="αf (true)")

        # 推定値（折れ線 + マーカー）
        scatterlines!(ax, x, df.α_est, color=:red, linestyle=:dash, linewidth=2,
                     marker=:circle, markersize=marker_size, label="α_est")
        scatterlines!(ax, x, df.β_est, color=:blue, linestyle=:dash, linewidth=2,
                     marker=:rect, markersize=marker_size, label="β_est")
        scatterlines!(ax, x, df.αf_est, color=:green, linestyle=:dash, linewidth=2,
                     marker=:diamond, markersize=marker_size, label="αf_est")

        # y軸の範囲を適切に設定
        all_values = vcat(df.α, df.β, df.αf, df.α_est, df.β_est, df.αf_est)
        y_min = minimum(all_values) - 0.1 * abs(minimum(all_values))
        y_max = maximum(all_values) + 0.1 * abs(maximum(all_values))
        ylims!(ax, y_min, y_max)

        # 最初のサブプロットにのみ凡例を表示
        if i == 1
            axislegend(ax, position=:rt, backgroundcolor=:transparent)
        end
    end

    return fig
end

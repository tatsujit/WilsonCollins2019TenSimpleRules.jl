using CSV
using DataFrames
using CairoMakie
using StatsBase

output_dir = "../../_output/"
# output_dir = "../../../_output/"

model_name = "DFQ"
n_opt = 1
exp_str = "exp-takarada-20250110"
# data_filename = output_dir * exp_str * "-$(model_name)-parameter-estimation-MLE-n_opt=$(n_opt).csv"
data_filename = "/Users/tatsujit/git/multibanditjulia1/_output/exp-takarada-20250110-DFQ-parameter-estimation-MLE-n_opt=1.csv"
groups_dscr = ["high", "low"]
fn = exp_str * "-$(model_name)-parameter-estimation"

# groups_dscr = ["Appr. goal", "Low goal", "No goal"]
# exp_str = "exp-20241226-123"
# fn = "exp-20241226-123-UQ-parameter-estimation"
# n_opt = 20
# data_filename = output_dir * exp_str * "-UQ-parameter-estimation-MLE-n_opt=$(n_opt).csv"


# fn = "exp7-UQ-parameter-estimation"
# groups_dscr = ["Appr. goal", "Low goal", "No goal", "No goal (no bar)"]
# # n_opt = 20
# # exp_str = "exp7-20250522"
# exp_str = "exp7"
# data_filename = output_dir * exp_str * "-UQ-parameter-estimation-MLE-n_opt=$(n_opt).csv"
# fn = "exp6-UQ-parameter-estimation"
# groups_dscr = ["Appr. goal", "No goal"]

# fn = "exp20250307-UQ-parameter-estimation"
# groups_dscr = ["Appr. goal", "Low goal", "No goal"]
# n_opt = 20
# exp_str = "exp20250307"
# data_filename = output_dir * exp_str * "-UQ-parameter-estimation-MLE-n_opt=$(n_opt).csv"


# est_params_df = CSV.read(output_dir * fn * ".csv", DataFrame)
est_params_df = CSV.read(data_filename, DataFrame)
df = est_params_df
n_groups = length(groups_dscr)
colors = cgrad(:turbo, n_groups, categorical=true)


size_unit = 300
fig = Figure(size = (4size_unit, 1size_unit))


# 3つの指標のraincloud plots
metrics = [
    (:α, "a"),
    # (:αm, "am"),
    (:αf, "af"),
    (:β, "b"),
    (:nll, "nll"),
    (:aic, "AIC"),
    (:aicc, "AICc"),
    (:bic, "BIC"),
    # (:η, "eta"),
    # (:ν, "nu"),
]
n_metrics = length(metrics)
# グリッドレイアウトを作成
gl = fig[1, 1] = GridLayout(1, n_metrics)  # 4行3列のグリッド

axs = [Axis(gl[1, i], title = string(var)) for (i, (var, title)) in enumerate(metrics)]

gids = sort(unique(est_params_df.gid))

for (j, (metric, title)) in enumerate(metrics)
    ax = axs[j] # metric ごとに異なる軸を使用
    ax.xticks = (gids, groups_dscr)
    (min_gid, max_gid) = (minimum(gids), maximum(gids))
    @show (min_gid, max_gid)
    # limits!(ax, 0.5, n_groups+0.5, 0, 1) # 表示範囲を設定
    limits!(ax, min_gid-0.5, max_gid+0.5, 0, 1) # 表示範囲を設定
    for (i, gid) in enumerate(gids)
        group_data = filter(row -> row.gid == gid, df)
        n = nrow(group_data)
        color = colors[i] # colors[mod1(i, length(colors))]
        data = group_data[!, metric]
        data_max = maximum(data)
        # limits!(ax, 0.5, n_groups+0.5, 0, data_max > 1.0 ? data_max : 1.0) # 表示範囲を設定
        limits!(ax, min_gid - 0.5, max_gid+0.5, 0, data_max > 1.0 ? data_max + 1.0 : 1.0) # 表示範囲を設定

        ################################################################
        # raincloud plot
        ################################################################
        # Violin, scatter, boxplotの処理
        # Violin part (cloud)
        violin!(ax, fill(gid, n),
                data,
                side = :left, color = (color, 0.6), show_median = false)
        # Scatter part (drops)
        offset = 0.2 # violin が左なのに対して、scatterを右にずらす
        scatter!(ax, fill(gid, n) .+ rand(n) .* 0.1 .- 0.05 .+ offset,
                 data,
                 color = color, markersize = 5, alpha = 0.7)
        # Boxplot part (rain)
        boxplot!(ax, fill(gid, n),
                 data,
                 show_notch = true, color = (color, 0.8), width = 0.2)

        ################################################################
        # 統計値プロット
        ################################################################
        μ = mean(data)

        σ = std(data)
        med = median(data)
        # 平均値 ± 標準偏差（errorbars! + scatter!）
        # errorbars!(ax, [gid + offset], [μ], yerr = ([σ], [σ]), color = :black)
        errorbars!(ax, [gid + offset*2], [μ], [σ],
                   color = color, alpha = 0.5, linewidth = 3, whiskerwidth = 8)
        # ダイヤモンド（平均値）: 鋭角的な形状が平均値の「計算された」性質を表現
        scatter!(ax, [gid + offset*2], [μ], color = :red, marker = :diamond, markersize = 10)
        # 円（中央値）: 滑らかな形状が中央値の「安定した」性質を表現
        scatter!(ax, [gid + offset*2], [med], color = :blue, marker = :circle, markersize = 15)
    end
end



fig |> display
# save(output_dir * fn * "-raincloud.png", fig)
save(output_dir * fn * "-raincloud.pdf", fig)

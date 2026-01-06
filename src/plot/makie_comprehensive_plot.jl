using CSV, DataFrames, CairoMakie
# import Pkg; Pkg.add("ColorSchemes")
using ColorSchemes


parameter = "ρ"
# 解像度はバランスによって調整、このままスライドとするので、4:3の比率を維持
res = 350
fig = Figure(size=(res * 4, res * 3))

# データ読み込み
fn = total_filename # csvファイル名
title = output_filename # タイトルをここではファイル名(拡張子なし)に
# フォントサイズの設定
# title_fontsize = 28
# ylabel_fontsize = 24  # Y軸ラベルのフォントサイズを大きく設定
# xlabel_fontsize = 24
# tick_fontsize = 20

# データフレームの読み込み
df = CSV.read(fn, DataFrame)
# プロット対象の列名を取得 (3列目以降)
metrics = names(df)[3:end]
# 長形式（ロングフォーマット）に変換
long_df = DataFrame()
for metric in metrics
    temp = select(df, :Parameter, :Trial, metric => :Value)
    temp.Metric = fill(metric, nrow(temp))
    append!(long_df, temp)
end
# メトリクスごとにY軸のスケーリングを調整するため、範囲を計算
metric_ranges = Dict()
for metric in metrics
    data = filter(row -> row.Metric == metric, long_df)
    metric_ranges[metric] = (minimum(data.Value), maximum(data.Value))
end
metric_ranges["AverageReward"] = (0.4, 0.8)
metric_ranges["MeanAccuracy"] = (0.25, 1.0)
metric_ranges["MeanRelativeAccuracy"] = (0.25, 1.0)
# metric_ranges["MeanRelativeAccuracy"] = (0.0, 1.0)

# 参照値のユニークな値を取得
parameters = unique(df.Parameter) # |> sort

# 色のリストを作成
# colors = ColorSchemes.tab10.colors       # 定番の10色パレット
# colors = ColorSchemes.tableau_20.colors #
# colors = get(ColorSchemes.viridis, range(0, stop=1, length=length(ab_pairs)))
colors = get(ColorSchemes.turbo, range(0, stop=1, length=length(ab_pairs)))
# colors = [:red, :blue, :green, :orange, :purple, :brown, :black, :cyan]

################################################################
## レイアウト
################################################################
# まず大まかに左右に分ける
l_gl = GridLayout() # 左側のグリッドレイアウト
r_gl = GridLayout() # 右側のグリッドレイアウト
fig[1, 1] = l_gl
fig[1, 2] = r_gl

# ax111の代わりに、直接テーブル用のGridLayoutを配置
ax111 = l_gl[1, 1] = GridLayout()
ax211 = l_gl[2, 1] = GridLayout()
# ax112 = l_gl[1:2, 2] = Axis(l_gl[1, 2], title="legend")
# ax121 = Axis(l_gl[2, 1:2], title="comments / notes")
# ax211 = Axis(r_gl[1, 1:2], title="mean relative accuracy")
# ax221 = Axis(r_gl[2, 1:2], title="mean regret")
rl_gl = r_gl[3, 1] = GridLayout()
# ax2211 = Axis(r_gl[3, 1], title="value function")
# ax2212 = Axis(r_gl[3, 2], title="action selection probability")

# ax121をスライドでコメントを書くための空白スペースに設定
# hidedecorations!(ax121)  # 軸の装飾を非表示
# hidespines!(ax121)       # 軸の枠線を非表示

# プロット用の辞書（行と列のインデックスをキーとして使用）
axes_dict = Dict()
# 表示するのはこれらのメトリクスだけ
metrics_to_show = ["MeanRelativeAccuracy", "MeanRegret"]
metrics_to_show_labels = ["relative accuracy", "regret"]
# 各メトリクスに対してサブプロットを作成（メトリクスの数に応じて）
for (i, metric) in enumerate(metrics_to_show)
    ax = Axis(r_gl[i, 1:2],
              title = metrics_to_show_labels[i],
              xlabel = "Trial",
              ylabel = metric)
    # Y軸の範囲を調整（少し余白を持たせる）
    min_val, max_val = metric_ranges[metric]
    padding = (max_val - min_val) * 0.1
    ylims!(ax, min_val - padding, max_val + padding)
    # 各参照値ごとにラインプロット
    for (j, ref) in enumerate(parameters)
        # その参照値のデータを抽出
        subset = filter(row -> row.Parameter == ref, df)
        # プロット
        lines!(ax, subset.Trial, subset[:, metric],
              label = "Ref = $(ref)",
              color = colors[mod(j-1, length(colors))+1],
              linewidth = 2)
    end
end




################################################################
#
# 左側のテーブル二つ
#
################################################################



################################################################
# テーブルのタイトル
################################################################
# Label(ax111[0, :], "parameters 1-1-1", fontsize=18)
# display_params(fig, 1, 1, params_df)
# display_params(fig, 2, 1, ab_params_df)
Label(ax111[0, :], "Simulation settings", fontsize=18)
display_params(ax111, 1, 1, params_df)
Label(ax211[0, :], "Beta parameters", fontsize=18)
display_params(ax211, 1, 1, ab_params_df)



################################################################
#
# 右側のプロット各種
#
################################################################


# 凡例を(3,2)の位置に配置
elements = [LineElement(color = colors[mod(j-1, length(colors))+1], linewidth = 10)
            for j in 1:length(parameters)]
labels = ["$(parameter) = $(ref)" for ref in parameters]
# Legend(l_gl[1,2], elements, labels, "パラメータ値",
Legend(r_gl[3,2], elements, labels, "パラメータ値",
       framevisible = true,
       margin = (10, 10, 10, 10),
       padding = (10, 10, 10, 10),
       labelsize = 12)
# l_gl[1,2]
hidedecorations!(ax112)  # 軸の装飾を非表示
hidespines!(ax112)       # 軸の枠線を非表示

# グリッドの行と列の間隔を設定
rowgap!(ax111, 0)
colgap!(ax111, 5)

################################################################
## value function
################################################################
# y値を計算（デフォルトのパラメータを使用）
# x_values = -10.0:0.1:10.0
# x_values = -1.0:0.01:1.0
# (vα, vβ, vλ) = params_df.Value[8]
# (dμ, dλ) = params_df.Value[9]
# # TODO: ほんとは実際にシミュレーションで動いている agent の estimator からの値を使うべき
# # 関数を渡せば良いだけだから、やると良い
# println("(vα, vβ, vλ, dμ, dλ): ", (vα, vβ, vλ, dμ, dλ))
# # y_values = [prospectValue(x, vα, vβ, vλ, dμ, dλ) for x in x_values]
# # y_values = [sigmoidValue(x, vβ) for x in x_values]
# y_values = [symlogValue(x, vα, vβ, vλ, dμ, dλ) for x in x_values]
# y_values = [beta_cdf(x, a, b) for x in x_values]
ax2211 = Axis(r_gl[3, 1],
              xlabel = "Action value (Q)",
              ylabel = "Utility (V)",
              title="utility function")
# # 線をプロット
# x_values = -0.5:0.01:1.5
x_values = -0.5:0.01:1.5
ys_values = Vector{Vector{Float64}}()
for ab in ab_pairs
    a, b = ab
    uf = beta_cdf_curried(a, b)
    ys = [uf(x) for x in x_values]
    push!(ys_values, ys)
end
for i in 1:length(ab_pairs)
    lines!(ax2211, x_values, ys_values[i], linewidth = 2, color = colors[i])
end

# # 原点に縦線と横線を追加
# vlines!(ax2211, 0, color = :black, linestyle = :dash, alpha = 0.5)
# hlines!(ax2211, 0, color = :black, linestyle = :dash, alpha = 0.5)

# ################################################################
# ################ 行動選択確率の計算
# ################################################################
# x_values = -2.5:0.01:2.5
# # Qs = [2.0, 1.2]
# # Qs = [20.0, 5.0]
# # Qs = [1.1, -1.1]
# # Qs = [0.8, 0.4]
# Qs = [0.6, 0.4]
# n_arm = length(Qs)
# βs = 0.0:2.0:10.0
# # ρs = -1.8:0.1:3.0
# ρs = -1.0:0.01:2.0
# # データの生成
# agent = Agent(
#     # ProspectValueEstimator(n_arm, ρ, vα, vβ, vλ),
#     # ProspectSampleAverageEstimator(n_arm, ρ, vα, vβ, vλ),
#     # estimator(n_arm, ρ, vα, vβ, vλ, dμ, dλ),
#     # estimator(n_arm, ρs[1], vα, vβ, vλ, dμ, dλ), # discontinuous
#     estimator(n_arm, α, a, b), # betaCDF
#     # estimator(n_arm, ρs[1], vβ),
#     policy(0.0)
#     # ProspectSampleAverageEstimator(n_arm, ρ, vα, vβ, vλ),
# #     SoftmaxPolicy(0.0)
# )
# beta_rho_prob = action_selection_probability(agent, βs, ρs, Qs) # softmax.jl にあり

# ax2212 = Axis(r_gl[3, 2],
#               title="action selection probability",
#               # title = "Pとρの関係（βグループ別）",
#               xlabel = "Rho (ρ)",
#               ylabel = "P(a=1) つまり価値 $(Qs[1]) の行動を選ぶ確率",
#               subtitle = "values = $(string(Qs))"#", vα=$vα, vβ=$vβ, vλ=$vλ"
#               )
# # 読み込む
# df = beta_rho_prob
# # ユニークなBeta値を取得
# # beta_groups = unique(df.Beta)
# beta_groups = reverse(unique(df.Beta)) # 凡例の表示順番を逆にするため

# # 各Betaグループごとにプロット
# for (i, beta) in enumerate(beta_groups)
#     group_data = df[df.Beta .== beta, :]
#     # 線をプロット - 各βに対して異なる色を使用
#     lines!(ax2212, group_data.Rho, group_data.P,
#            label = "β = $beta",
#            linewidth = 2)
# end

# # ρ=rewardProbabilitiesの垂直線を描く
# for q in Qs
#     lines!(ax2212, [q, q], [0.5, 1.0],
#            color = :black, linestyle = :dash, alpha = 0.5)
# end

# # 凡例を図の右下に配置
# # Legend(fig, ax, halign = :right, valign = :bottom)
# # axislegend(ax, position = :rb)
# axislegend(ax2212 # reverse=true,
#            , position = :rb
#            , margin = (10, 10, 15, 10)  # (左, 右, 下, 上)のマージン
#            # , padding = (10, 10, 10, 10)
#            , padding = (5, 5, 5, 5)
#            # , alpha = 0.7
#            )  # 内部パディング



# 表示
display(fig)

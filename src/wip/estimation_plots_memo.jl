using CairoMakie, DataFrames


# 使用例（サンプルデータでのテスト）
function create_sample_data()
    # サンプルデータの生成（実際のデータに置き換えてください）
    dfs = DataFrame[]

    for i in 1:10
        df = DataFrame(
            α = fill(0.1, 10),
            β = fill(6.0, 10),
            αf = fill(0.9, 10),
            α_est = 0.1 .+ 0.2 * randn(10),
            β_est = 6.0 .+ 2.0 * randn(10),
            αf_est = max.(0, min.(1, 0.9 .+ 0.3 * randn(10)))
        )
        push!(dfs, df)
    end

    return dfs
end

# プロット実行例
# sample_dfs = create_sample_data()
# fig = plot_parameter_comparison(sample_dfs)
# display(fig)

# 実際のデータを使用する場合：
# your_dataframes = [df[1], df[2], df[3], df[4], df[5], df[6], df[7], df[8], df[9], df[10]]
# fig = plot_parameter_comparison(your_dataframes)
# display(fig)

# より詳細なカスタマイズ版
function plot_parameter_comparison_detailed(dataframes::Vector{DataFrame};
                                          figsize=(1400, 700),
                                          title_fontsize=12,
                                          label_fontsize=10)
    fig = Figure(size=figsize)

    n_dfs = length(dataframes)
    n_rows = 2
    n_cols = 5

    # カラーパレット
    colors = Dict(
        :α => :red,
        :β => :blue,
        :αf => :green
    )

    for i in 1:min(n_dfs, n_rows * n_cols)
        row = div(i - 1, n_cols) + 1
        col = mod(i - 1, n_cols) + 1

        ax = Axis(fig[row, col],
                  title="Simulation $i",
                  titlesize=title_fontsize,
                  xlabel="Sample Index",
                  ylabel="Parameter Value",
                  xlabelsize=label_fontsize,
                  ylabelsize=label_fontsize)

        df = dataframes[i]
        x = 1:nrow(df)

        # 真値（水平線）
        for (param, color) in colors
            true_val = df[1, param]
            hlines!(ax, true_val,
                   color=color,
                   linestyle=:solid,
                   linewidth=3,
                   alpha=0.7,
                   label="$param (true)")
        end

        # 推定値（折れ線グラフ + マーカー）
        scatterlines!(ax, x, df.α_est,
               color=colors[:α],
               linestyle=:dash,
               linewidth=2,
               marker=:circle,
               markersize=4,
               label="α_est")

        scatterlines!(ax, x, df.β_est,
               color=colors[:β],
               linestyle=:dash,
               linewidth=2,
               marker=:rect,
               markersize=4,
               label="β_est")

        scatterlines!(ax, x, df.αf_est,
               color=colors[:αf],
               linestyle=:dash,
               linewidth=2,
               marker=:diamond,
               markersize=4,
               label="αf_est")

        # y軸範囲の調整
        all_values = vcat(df.α, df.β, df.αf, df.α_est, df.β_est, df.αf_est)
        # 異常値を除外して範囲設定
        valid_values = filter(x -> !isnan(x) && isfinite(x), all_values)

        if !isempty(valid_values)
            y_min = quantile(valid_values, 0.05) - 0.5
            y_max = quantile(valid_values, 0.95) + 0.5
            ylims!(ax, y_min, y_max)
        end

        # グリッド追加
        ax.xgridvisible = true
        ax.ygridvisible = true
        ax.xgridcolor = (:gray, 0.3)
        ax.ygridcolor = (:gray, 0.3)
    end

    # 全体的な凡例を右上に配置
    Legend(fig[1, 6],
           [LineElement(color=colors[:α], linestyle=:solid, linewidth=3),
            LineElement(color=colors[:β], linestyle=:solid, linewidth=3),
            LineElement(color=colors[:αf], linestyle=:solid, linewidth=3),
            LineElement(color=colors[:α], linestyle=:dash, linewidth=2),
            LineElement(color=colors[:β], linestyle=:dash, linewidth=2),
            LineElement(color=colors[:αf], linestyle=:dash, linewidth=2)],
           ["α (true)", "β (true)", "αf (true)", "α_est", "β_est", "αf_est"],
           "Legend")

    return fig
end

# 使用方法の例
"""
# あなたのデータを使用する場合：
df_vector = [df[1], df[2], df[3], df[4], df[5], df[6], df[7], df[8], df[9], df[10]]
fig = plot_parameter_comparison_detailed(df_vector)
display(fig)

# または保存する場合：
save("parameter_comparison.png", fig)
save("parameter_comparison.pdf", fig)
"""
df_vector = [df[1,:], df[2,:], df[3,:], df[4,:], df[5,:], df[6,:], df[7,:], df[8,:], df[9,:], df[10,:]]
fig = plot_parameter_comparison_detailed(df_vector)
display(fig)

df

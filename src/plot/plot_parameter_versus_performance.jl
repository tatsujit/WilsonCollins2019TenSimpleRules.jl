using CairoMakie
using DataFrames

"""
    plot_parameter_versus_performance!(ax, sub, param_col, perf_col; ...)

パラメータ（横軸）とパフォーマンス（縦軸）の関係を Axis `ax` に描画する。
グループ（例: モデル）ごとに線とスキャッターを描き、必要に応じて縦の参照線や y 軸スケールを指定できる。

# Arguments
- `ax`: 描画先の Makie Axis
- `sub`: 少なくとも `param_col`, `perf_col`, `group_col` 列を含む DataFrame
- `param_col`: パラメータ（横軸）の列名（Symbol）
- `perf_col`: パフォーマンス（縦軸）の列名（Symbol）

# Keyword arguments
- `group_col`: グループ分けに使う列名（例: :model）. デフォルト :model
- `model_colors`: グループ値 => 色 の Dict
- `model_markers`: グループ値 => マーカー の Dict
- `vline_x`: 縦の参照線を引く x の値のベクトル（省略可）
- `vline_labels`: 参照線の凡例ラベル（vline_x と同じ長さ）. 省略可
- `vline_colors`: 参照線の色（vline_x と同じ長さ）. 省略時は :red, :orange を繰り返し
- `yscale`: :linear, :log10, :log10_offset のいずれか. デフォルト :linear
- `y_offset`: yscale == :log10_offset のときに log10(y + y_offset) に使うオフセット. デフォルト 1.0
- `legend_position`: 凡例の位置. デフォルト :lt
"""
function plot_parameter_versus_performance!(ax, sub, param_col::Symbol, perf_col::Symbol;
    group_col::Symbol = :model,
    model_colors = Dict(),
    model_markers = Dict(),
    vline_x = nothing,
    vline_labels = nothing,
    vline_colors = nothing,
    yscale = :linear,
    y_offset = 1.0,
    legend_position = :lt,
)
    groups = sort(unique(sub[!, group_col]))
    default_vline_colors = [:red, :orange]

    for g in groups
        rows = sub[sub[!, group_col] .== g, :]
        rows = sort(rows, param_col)
        x = rows[!, param_col]
        y = copy(rows[!, perf_col])
        if yscale == :log10_offset
            y = y .+ y_offset
        end
        color = get(model_colors, g, :blue)
        marker = get(model_markers, g, :circle)
        lines!(ax, x, y; color = color, linewidth = 2, label = string(g))
        scatter!(ax, x, y; color = color, marker = marker, markersize = 8)
    end

    if vline_x !== nothing && !isempty(vline_x)
        for (i, xv) in enumerate(vline_x)
            lbl = (vline_labels !== nothing && i <= length(vline_labels)) ? vline_labels[i] : ""
            clr = (vline_colors !== nothing && i <= length(vline_colors)) ? vline_colors[i] : default_vline_colors[mod1(i, length(default_vline_colors))]
            vlines!(ax, [xv]; color = clr, linestyle = :dash, linewidth = 1.5, label = lbl)
        end
    end

    if yscale == :log10 || yscale == :log10_offset
        ax.yscale = log10
    end

    axislegend(ax; position = legend_position)
    return ax
end

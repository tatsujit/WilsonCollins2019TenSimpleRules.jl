# プロット関数整理の設計ガイド

バンディット問題でのシミュレーション、データ分析、パラメータ推定やパラメータリカバリ、モデル選択などの結果をプロットするための設計指針。

## 1. 3層構造で整理する

```
plots/
├── core/           # 低レベル: 基本プリミティブ
│   ├── themes.jl   # 色、フォント、スタイル設定
│   └── utils.jl    # 共通ユーティリティ
│
├── components/     # 中レベル: 再利用可能な部品
│   ├── raincloud.jl
│   ├── timeseries.jl
│   └── legend.jl
│
└── templates/      # 高レベル: 用途別テンプレート
    ├── simulation_results.jl
    ├── parameter_recovery.jl
    └── model_comparison.jl
```

## 2. バンディット研究で必要な定番プロット

| 用途 | プロットタイプ | 関数名例 |
|------|---------------|----------|
| シミュレーション | 累積regret曲線、選択確率推移 | `plot_regret()`, `plot_choice_probability()` |
| パラメータ推定 | raincloud plot (群間比較) | `plot_parameter_distribution()` |
| パラメータリカバリ | 真値 vs 推定値 scatter + 対角線 | `plot_recovery()` |
| モデル選択 | AIC/BIC/WAIC バープロット | `plot_model_comparison()` |
| 価値関数 | utility function 曲線 | `plot_utility_function()` |

## 3. 設計の核心: 「データ構造」と「見た目」を分離

```julia
# 悪い例: すべてが混在
function my_plot(df, param, colors, fontsize, ...)
    # データ処理 + スタイル + プロット が一緒
end

# 良い例: 分離する
struct PlotConfig
    colors::Vector
    fontsize::Int
    # ... スタイル設定
end

function plot_recovery(
    true_values::Vector,
    estimated_values::Vector;
    config::PlotConfig = default_config()
)
    # プロットロジックのみ
end
```

## 4. 「育てていく」ための現実的なアプローチ

**問題点**: プロットは常に微調整が必要なので、完璧な汎用関数は作れない

**解決策**: **70-30ルール**
- 70%: 関数が自動で処理（デフォルトで十分なケース）
- 30%: 戻り値の `Figure` / `Axis` を返して、呼び出し側で微調整可能にする

```julia
function plot_raincloud(df, metric; kwargs...)
    fig = Figure()
    ax = Axis(fig[1,1])

    # 基本プロット
    # ...

    return (fig=fig, ax=ax)  # 戻り値を返す！
end

# 使う側
result = plot_raincloud(df, :α)
# 必要なら微調整
xlims!(result.ax, 0, 2)
Label(result.fig[0, 1], "追加タイトル")
save("output.pdf", result.fig)
```

## 5. 具体的な実装案: 最小限から始める

まず `MultiBandits.jl` に入れるべき核となる関数:

```julia
module Plots

export plot_recovery, plot_raincloud_comparison, plot_regret_curves

# テーマ設定
const DEFAULT_COLORS = cgrad(:turbo, 8, categorical=true)

"""
パラメータリカバリプロット（真値 vs 推定値）
"""
function plot_recovery(true_vals, est_vals;
                       xlabel="True", ylabel="Estimated",
                       identity_line=true)
    fig = Figure()
    ax = Axis(fig[1,1], xlabel=xlabel, ylabel=ylabel)
    scatter!(ax, true_vals, est_vals)
    if identity_line
        lims = extrema(vcat(true_vals, est_vals))
        lines!(ax, lims, lims, color=:gray, linestyle=:dash)
    end
    return (fig=fig, ax=ax)
end

"""
群間比較用 raincloud プロット
"""
function plot_raincloud_comparison(df, metric::Symbol, group_col::Symbol;
                                   group_labels=nothing)
    # ... 実装
    return (fig=fig, axes=axes)
end

end # module
```

## 6. 実践的なステップ

1. **今すぐ**: よく使う2-3個のプロットだけ関数化（`plot_recovery`, `plot_raincloud_comparison`）
2. **常に**: `(fig, ax)` を返す設計にして、後から調整可能に
3. **徐々に**: 使いながら共通パターンを `core/` に抽出
4. **避ける**: 最初から完璧な汎用化を目指さない（YAGNI原則）

## 優先度の高いプロット関数

現在のコードを見ると、以下が優先度高い:
1. `plot_raincloud_comparison` - パラメータ推定結果の群間比較
2. `plot_recovery` - パラメータリカバリの検証
3. `plot_regret_curves` - シミュレーション結果の可視化

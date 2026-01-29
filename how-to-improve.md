# WilsonCollins2019TenSimpleRules.jl パッケージ化改善計画

## 目次

1. [現状分析](#1-現状分析)
2. [課題の特定](#2-課題の特定)
3. [改善案](#3-改善案)
4. [推奨パッケージ構造](#4-推奨パッケージ構造)
5. [実装ガイドライン](#5-実装ガイドライン)
6. [アクションプラン](#6-アクションプラン)

---

## 1. 現状分析

### 1.1 プロジェクト概要

本リポジトリは、Wilson & Collins (2019) "Ten Simple Rules for the Computational Modeling of Behavioral Data" に基づく強化学習バンディットシミュレーションと、パラメータ推定を実装した研究用コードベースである。

**主要機能**:
- Q学習ベースのエージェントによるマルチアームバンディットシミュレーション
- 効用関数（Utility Function）と粘着性（Stickiness）を含む拡張モデル
- 最尤推定（MLE）によるパラメータ推定
- CairoMakie/GLMakieによる可視化

### 1.2 ディレクトリ構造

```
src/
├── _setup.jl                          # メインエントリーポイント（include集約）
├── _utils.jl                          # ユーティリティ関数
├── core/                              # コアシミュレーション（11ファイル）
│   ├── actionValueEstimator.jl        # 抽象型定義
│   ├── agent.jl                       # エージェント定義
│   ├── environment.jl                 # 環境（バンディット）定義
│   ├── estimator.jl                   # Q学習推定器（主要ファイル）
│   ├── history.jl                     # 履歴記録
│   ├── policy.jl                      # 方策の抽象型
│   ├── random_responding.jl           # ランダム方策
│   ├── softmax.jl                     # ソフトマックス方策
│   ├── system.jl                      # シミュレーションループ
│   ├── utility_functions.jl           # 効用関数群
│   └── misc/                          # その他のプロット関連
│       ├── utility_functions_plot-in1-.jl
│       └── utility_functions_plot.jl
├── parameter-handling/                # 設定管理（2ファイル）
│   ├── config.jl                      # YAML/JSON設定
│   └── estimator-params.jl            # パラメータ定義
├── plot/                              # プロット機能（3ファイル）
│   ├── makie_comprehensive_plot.jl
│   ├── parameter-estimation-exp-plot.jl
│   └── raincloud-simple.jl
├── utils/                             # ユーティリティ（2ファイル）
│   ├── sampling.jl                    # サンプリング関数
│   └── sleeping_arms.jl               # Sleeping arms実装
├── wip/                               # 作業中（1ファイル）
│   └── estimation_plots_memo.jl
├── estimate_parameters_mle.jl         # MLE推定（簡易版）
├── estimation_plots.jl                # 推定結果プロット
├── parameter-estimation.jl            # MLE推定（拡張版）
├── parameters.jl                      # パラメータ定義・Enum
├── plot.jl                            # メインプロット関数
├── plot-memo.jl                       # プロットメモ
└── validity.jl                        # バリデーション関数
```

**ファイル数**: 34ファイル（srcディレクトリ内）

### 1.3 主要コンポーネント分析

#### 1.3.1 Estimator（推定器）

**ファイル**: `src/core/estimator.jl`

Q学習ベースの行動価値推定器。22個のパラメータを持つ複雑な構造体。

```julia
struct Estimator <: AbstractEstimator
    # 学習パラメータ
    Q0::Float64      # 初期Q値
    α::Float64       # 学習率（報酬獲得時）
    αm::Float64      # 学習率（報酬なし時）
    αf::Float64      # 未選択armの減衰率
    μ::Float64       # 減衰先の値

    # 行動選択パラメータ
    β::Float64       # 逆温度（ソフトマックス）

    # 効用関数パラメータ
    η::Float64       # Beta分布形状パラメータ1
    ν::Float64       # Beta分布形状パラメータ2

    # 粘着性パラメータ
    τ::Float64       # 粘着性の強さ
    φ::Float64       # 粘着性の減衰率
    C0::Float64      # 初期粘着性

    # 状態ベクトル
    Q::Vector{Float64}   # 行動価値
    V::Vector{Float64}   # 効用値
    C::Vector{Float64}   # 粘着性値
    W::Vector{Float64}   # 統合価値（Q + τ*C）
    N::Vector{Int}       # 選択回数

    # その他
    n_arms::Int
    utility_function::Int
end
```

**主要関数**:
- `Estimator(n_arms; kwargs...)`: コンストラクタ
- `update!(e::Estimator, action, reward)`: 学習更新
- `selection_probabilities(W, available_arms)`: ソフトマックス選択確率
- `beta_cdf(x, η, ν)`: 効用関数計算
- `count_params(e)`: 自由パラメータ数カウント

#### 1.3.2 Environment（環境）

**ファイル**: `src/core/environment.jl`

2種類の環境を定義：

```julia
# 定常環境
struct Environment <: AbstractEnvironment
    distributions::Vector{Distribution}
    n_arms::Int
end

# 非定常環境
struct NonStationaryEnvironment <: AbstractEnvironment
    distributions_sequence::Vector{Vector{Distribution}}
    change_points::Vector{Int}
    n_arms::Int
    current_trial::Ref{Int}
end
```

**主要関数**:
- `Environment(probs::Vector{Float64})`: Bernoulliバンディット作成
- `Environment(dists::Vector{Distribution})`: 任意分布バンディット
- `sample_reward(env, arm)`: 報酬サンプリング
- `mean(env)`: 期待報酬取得

#### 1.3.3 Policy（方策）

**ファイル**: `src/core/policy.jl`, `softmax.jl`, `random_responding.jl`

```julia
abstract type AbstractPolicy end

struct SoftmaxPolicy <: AbstractPolicy
    β::Float64  # 逆温度パラメータ
end

struct RandomResponding <: AbstractPolicy
    probs::Vector{Float64}  # 固定選択確率
end
```

#### 1.3.4 Agent（エージェント）

**ファイル**: `src/core/agent.jl`

```julia
struct Agent
    policy::AbstractPolicy
    estimator::AbstractEstimator
end
```

**主要関数**:
- `select_action(agent, available_arms)`: 行動選択

#### 1.3.5 System（シミュレーションシステム）

**ファイル**: `src/core/system.jl`

```julia
struct System <: AbstractSystem
    agent::Agent
    environment::AbstractEnvironment
    history::AbstractHistory
    rng::AbstractRNG
end
```

**主要関数**:
- `run!(system, n_trials)`: シミュレーション実行
- `step!(system, trial)`: 1試行実行

#### 1.3.6 History（履歴）

**ファイル**: `src/core/history.jl`

3種類の履歴構造体：

```julia
# 基本履歴
struct History <: AbstractHistory
    actions::Vector{Int}
    rewards::Vector{Float64}
end

# 推定器状態を含む詳細履歴
struct EstimatorHistory <: AbstractHistory
    actions::Vector{Int}
    rewards::Vector{Float64}
    Qss::Matrix{Float64}    # Q値の履歴（trial × arms）
    Vss::Matrix{Float64}    # 効用値の履歴
    Css::Matrix{Float64}    # 粘着性の履歴
    Wss::Matrix{Float64}    # 統合価値の履歴
    Nss::Matrix{Int}        # 選択回数の履歴
    Pss::Matrix{Float64}    # 選択確率の履歴
end

# Limited Arms用履歴
struct LAHistory <: AbstractHistory
    # ...
end
```

#### 1.3.7 パラメータ推定

**ファイル**: `src/estimate_parameters_mle.jl`, `src/parameter-estimation.jl`

```julia
function estimate_parameters_mle(
    bandit_data,           # 行動・報酬データ
    true_params_df,        # 真のパラメータ（検証用）
    parameters,            # 推定対象パラメータ名
    estimation_ranges,     # パラメータ範囲
    n_opt                  # 最適化試行回数
)
```

Optim.jlを使用したMLE推定。複数の初期値からの最適化を実行。

### 1.4 依存パッケージ

**Project.toml記載**（17パッケージ）:

| パッケージ | 用途 | 必須度 |
|-----------|------|--------|
| Distributions | 確率分布 | 必須 |
| Random | 乱数生成 | 必須 |
| Statistics | 統計計算 | 必須 |
| SpecialFunctions | Beta関数など | 必須 |
| StatsBase | 統計ユーティリティ | 必須 |
| LinearAlgebra | 線形代数 | 必須 |
| DataFrames | データ操作 | 高 |
| CSV | CSV読み書き | 中 |
| Optim | 最適化 | 高（推定用） |
| CairoMakie | プロット（静的） | 中（可視化用） |
| GLMakie | プロット（インタラクティブ） | 低 |
| Makie | プロット基盤 | 中 |
| YAML | 設定ファイル | 低 |
| ProgressMeter | 進捗表示 | 低 |
| DrWatson | 研究ワークフロー | 低 |
| Distributed | 並列処理 | 低 |
| IceCream | デバッグ | 開発用 |

### 1.5 ファイル間依存関係

```
_setup.jl（エントリーポイント）
├── core/actionValueEstimator.jl
│   └── core/estimator.jl
├── core/policy.jl
│   ├── core/softmax.jl
│   └── core/random_responding.jl
├── core/agent.jl
│   └── [policy.jl, estimator.jlに依存]
├── core/environment.jl
├── core/history.jl
│   └── [estimator.jlの型に依存]
├── core/system.jl
│   └── [agent.jl, environment.jl, history.jlに依存]
├── _utils.jl
│   ├── utils/sampling.jl
│   └── utils/sleeping_arms.jl
├── estimate_parameters_mle.jl
│   └── [estimator.jl, DataFrames, Optimに依存]
└── plot.jl
    └── [history.jl, CairoMakieに依存]
```

---

## 2. 課題の特定

### 2.1 構造的課題

#### 2.1.1 モジュール構造の欠如

**現状**: `_setup.jl`で全ファイルをフラットにincludeしている

```julia
# 現在の_setup.jl（概要）
include("core/actionValueEstimator.jl")
include("core/policy.jl")
include("core/agent.jl")
# ... 以下同様
```

**問題点**:
- 名前空間が分離されていない
- 全シンボルがグローバルスコープに展開される
- 依存関係が暗黙的
- 部分的なロードができない

#### 2.1.2 export定義の欠如

**現状**: どの関数・型が公開APIか不明確

**問題点**:
- ユーザーが使うべき関数が分からない
- 内部実装と公開APIの区別がない
- ドキュメント生成時に対象が不明確

#### 2.1.3 重複コード

**該当ファイル**:
- `parameters.jl` と `parameter-handling/estimator-params.jl`
- `estimate_parameters_mle.jl` と `parameter-estimation.jl`

両者で類似の構造体・関数が定義されており、どちらを使うべきか不明確。

### 2.2 設計上の課題

#### 2.2.1 パラメータの複雑さ

**現状**: `Estimator`に22個のパラメータが存在

**問題点**:
- コンストラクタ呼び出しが複雑
- パラメータの意味・グループが分かりにくい
- デフォルト値の変更が困難
- ドキュメントが膨大になる

#### 2.2.2 型階層の不整合

**現状**:
```julia
abstract type AbstractEstimator end
struct Estimator <: AbstractEstimator end
struct EmptyEstimator end  # AbstractEstimatorを継承していない
```

**問題点**:
- `EmptyEstimator`が型階層に含まれていない
- 多相ディスパッチが正しく機能しない可能性

#### 2.2.3 関数名の曖昧さ

**例**:
- `run!` - 一般的すぎる名前
- `update!` - 何をupdateするか不明確
- `record!` - 同上

他パッケージとの名前衝突リスクがある。

### 2.3 品質・保守性の課題

#### 2.3.1 テストの欠如

**現状**: `test/`ディレクトリが存在しない

**問題点**:
- コードの正確性が検証されていない
- リファクタリング時の回帰リスク
- CI/CD構築が困難

#### 2.3.2 ドキュメントの不足

**現状**: docstringがほとんど存在しない

**例**（現在のコード）:
```julia
function update!(e::Estimator, action, reward)
    # docstringなし
    # ...
end
```

**問題点**:
- 関数の使い方が分からない
- パラメータの意味が不明
- 自動ドキュメント生成ができない

#### 2.3.3 作業中ファイルの存在

**該当**:
- `wip/estimation_plots_memo.jl`
- `plot-memo.jl`
- `misc/`ディレクトリ

公開パッケージには不適切。

### 2.4 依存関係の課題

#### 2.4.1 依存パッケージの過多

17パッケージへの依存は、以下の問題を引き起こす：
- インストール時間の増加
- 依存関係の競合リスク
- メンテナンス負荷

#### 2.4.2 可視化依存の混在

コア機能（シミュレーション）と可視化機能が分離されていない。
CairoMakie/GLMakieは重量級パッケージであり、シミュレーションのみ使いたいユーザーには不要。

### 2.5 ユーザビリティの課題

#### 2.5.1 高レベルAPIの欠如

現状、シミュレーション実行には多くのステップが必要：

```julia
# 現在の使用例（推測）
include("src/_setup.jl")
env = Environment([0.2, 0.4, 0.6, 0.8])
est = Estimator(4, α=0.1, β=1.0)
policy = SoftmaxPolicy(1.0)
agent = Agent(policy, est)
history = EstimatorHistory(100, 4)
system = System(agent, env, history, Random.default_rng())
run!(system, 100)
```

初心者には複雑すぎる。

#### 2.5.2 エラーメッセージの不足

パラメータ検証やエラーハンドリングが不十分。

### 2.6 命名・規約の課題

#### 2.6.1 パッケージ名

`WilsonCollins2019TenSimpleRules.jl`は：
- 長すぎる（35文字）
- 内容が直感的に分からない
- 論文タイトルへの依存

#### 2.6.2 ファイル命名の不統一

- ハイフン使用: `estimate-parameters-mle.jl`
- アンダースコア使用: `_setup.jl`, `_utils.jl`
- キャメルケース: `actionValueEstimator.jl`

---

## 3. 改善案

### 3.1 モジュール構造の導入

#### 3.1.1 サブモジュール分割

```julia
# src/ReinforcementLearningBandits.jl
module ReinforcementLearningBandits

# 標準ライブラリ
using Random
using Statistics
using LinearAlgebra

# 外部依存
using Distributions
using SpecialFunctions

# サブモジュール定義
include("Core/Core.jl")
include("Estimation/Estimation.jl")
include("Utils/Utils.jl")

# 再エクスポート
using .Core
using .Estimation
using .Utils

# 公開API
export Estimator, EstimatorConfig
export Environment, NonStationaryEnvironment
export SoftmaxPolicy, RandomResponding
export Agent, System
export History, EstimatorHistory
export simulate!, update_estimator!
export estimate_parameters_mle

end # module
```

#### 3.1.2 Coreサブモジュール

```julia
# src/Core/Core.jl
module Core

using ..ReinforcementLearningBandits: Distributions, Random

# 抽象型
export AbstractEstimator, AbstractEnvironment, AbstractPolicy
export AbstractHistory, AbstractSystem

# 具象型
export Estimator, EmptyEstimator
export Environment, NonStationaryEnvironment
export SoftmaxPolicy, RandomResponding
export Agent
export History, EstimatorHistory, LAHistory
export System

# 関数
export simulate!, step!
export update_estimator!, select_action
export sample_reward, record!

# ファイルインクルード
include("types.jl")           # 抽象型定義
include("estimators.jl")      # Estimator実装
include("environments.jl")    # Environment実装
include("policies.jl")        # Policy実装
include("agents.jl")          # Agent実装
include("histories.jl")       # History実装
include("systems.jl")         # System実装
include("utility_functions.jl")

end # module
```

### 3.2 パラメータ構造の改善

#### 3.2.1 パラメータグループ化

```julia
"""
学習に関するパラメータ群

# Fields
- `Q0::Float64`: 初期Q値（デフォルト: 0.5）
- `α::Float64`: 報酬獲得時の学習率（デフォルト: 0.1）
- `αm::Float64`: 報酬なし時の学習率（デフォルト: α）
- `αf::Float64`: 未選択armの忘却率（デフォルト: 0.0）
- `μ::Float64`: 忘却先の値（デフォルト: Q0）
"""
Base.@kwdef struct LearningParams
    Q0::Float64 = 0.5
    α::Float64 = 0.1
    αm::Float64 = α
    αf::Float64 = 0.0
    μ::Float64 = Q0
end

"""
効用関数のパラメータ

# Fields
- `η::Float64`: Beta分布の形状パラメータ1（デフォルト: 1.0）
- `ν::Float64`: Beta分布の形状パラメータ2（デフォルト: 1.0）
- `type::Symbol`: 効用関数タイプ（デフォルト: :beta_cdf）
"""
Base.@kwdef struct UtilityParams
    η::Float64 = 1.0
    ν::Float64 = 1.0
    type::Symbol = :beta_cdf
end

"""
粘着性（Stickiness/Perseveration）のパラメータ

# Fields
- `τ::Float64`: 粘着性の強さ（デフォルト: 0.0）
- `φ::Float64`: 粘着性の減衰率（デフォルト: 0.0）
- `C0::Float64`: 初期粘着性（デフォルト: 0.0）
"""
Base.@kwdef struct StickinessParams
    τ::Float64 = 0.0
    φ::Float64 = 0.0
    C0::Float64 = 0.0
end

"""
行動選択のパラメータ

# Fields
- `β::Float64`: 逆温度パラメータ（デフォルト: 1.0）
"""
Base.@kwdef struct SelectionParams
    β::Float64 = 1.0
end

"""
Estimator全体の設定

# Example
```julia
config = EstimatorConfig(
    learning = LearningParams(α=0.2),
    selection = SelectionParams(β=2.0)
)
```
"""
Base.@kwdef struct EstimatorConfig
    learning::LearningParams = LearningParams()
    utility::UtilityParams = UtilityParams()
    stickiness::StickinessParams = StickinessParams()
    selection::SelectionParams = SelectionParams()
end
```

#### 3.2.2 プリセット設定の提供

```julia
"""一般的な設定プリセット"""
module Presets

using ..Core: EstimatorConfig, LearningParams, SelectionParams

"""標準的なQ学習設定"""
const STANDARD_QLEARNING = EstimatorConfig(
    learning = LearningParams(α=0.1, Q0=0.5),
    selection = SelectionParams(β=1.0)
)

"""Wilson & Collins (2019) Figure 2の設定"""
const WILSON_COLLINS_FIG2 = EstimatorConfig(
    learning = LearningParams(α=0.15, αm=0.1),
    utility = UtilityParams(η=1.5, ν=1.0),
    stickiness = StickinessParams(τ=0.2),
    selection = SelectionParams(β=3.0)
)

"""探索重視の設定"""
const EXPLORATORY = EstimatorConfig(
    selection = SelectionParams(β=0.5)
)

"""活用重視の設定"""
const EXPLOITATIVE = EstimatorConfig(
    selection = SelectionParams(β=10.0)
)

end # module Presets
```

### 3.3 型階層の整理

```julia
# src/Core/types.jl

"""推定器の抽象型"""
abstract type AbstractEstimator end

"""テーブル形式の推定器"""
abstract type AbstractTabularEstimator <: AbstractEstimator end

"""環境の抽象型"""
abstract type AbstractEnvironment end

"""バンディット環境"""
abstract type AbstractBanditEnvironment <: AbstractEnvironment end

"""定常バンディット"""
abstract type AbstractStationaryBandit <: AbstractBanditEnvironment end

"""非定常バンディット"""
abstract type AbstractNonStationaryBandit <: AbstractBanditEnvironment end

"""方策の抽象型"""
abstract type AbstractPolicy end

"""価値ベースの方策"""
abstract type AbstractValueBasedPolicy <: AbstractPolicy end

"""履歴の抽象型"""
abstract type AbstractHistory end

"""システムの抽象型"""
abstract type AbstractSystem end
```

### 3.4 API設計の改善

#### 3.4.1 明確な関数命名

```julia
# 曖昧な名前 → 明確な名前
run!          → simulate!
update!       → update_estimator!
record!       → record_trial!
select_action → select_arm

# または名前空間での区別
Bandits.simulate!(system, n_trials)
Bandits.Estimation.fit_mle(data, config)
```

#### 3.4.2 高レベルAPI

```julia
"""
シンプルなバンディットシミュレーションを実行

# Arguments
- `n_arms::Int`: アーム数
- `n_trials::Int`: 試行数
- `reward_probs::Vector{Float64}`: 各アームの報酬確率

# Keyword Arguments
- `config::EstimatorConfig`: 推定器設定
- `seed::Int`: 乱数シード
- `record_history::Bool`: 詳細履歴を記録するか

# Returns
- `SimulationResult`: シミュレーション結果

# Example
```julia
result = simulate_bandit(
    4, 100, [0.2, 0.4, 0.6, 0.8];
    config = EstimatorConfig(learning = LearningParams(α=0.1)),
    seed = 42
)
println(result.total_reward)
println(result.optimal_rate)
```
"""
function simulate_bandit(
    n_arms::Int,
    n_trials::Int,
    reward_probs::Vector{Float64};
    config::EstimatorConfig = EstimatorConfig(),
    seed::Union{Int, Nothing} = nothing,
    record_history::Bool = true
)
    # 実装
end

"""シミュレーション結果"""
struct SimulationResult
    history::AbstractHistory
    total_reward::Float64
    optimal_rate::Float64
    final_estimator::AbstractEstimator
    config::EstimatorConfig
    metadata::Dict{Symbol, Any}
end

# DataFrameへの変換
Base.convert(::Type{DataFrame}, r::SimulationResult) = DataFrame(r.history)
DataFrames.DataFrame(r::SimulationResult) = convert(DataFrame, r)
```

### 3.5 ドキュメンテーション

#### 3.5.1 docstring標準フォーマット

```julia
"""
    update_estimator!(estimator::Estimator, action::Int, reward::Float64)

選択した行動と得られた報酬に基づいて推定器を更新する。

Q値、効用値(V)、粘着性(C)、統合価値(W)、選択回数(N)を更新する。

# Arguments
- `estimator::Estimator`: 更新対象の推定器
- `action::Int`: 選択した行動（1-indexed）
- `reward::Float64`: 得られた報酬

# Effects
推定器の内部状態を直接変更する（破壊的操作）。

# Algorithm
1. 報酬に効用関数を適用: `u = utility(reward)`
2. Q値を更新: `Q[action] += α * (u - Q[action])`
3. 効用値を更新: `V[action] = utility(Q[action])`
4. 粘着性を更新: `C[action] = 1.0`, 他は減衰
5. 統合価値を計算: `W = V + τ * C`
6. 選択回数をインクリメント: `N[action] += 1`

# Example
```julia
est = Estimator(4)
update_estimator!(est, 1, 1.0)  # arm 1を選択し、報酬1.0を獲得
```

# See also
- [`Estimator`](@ref): 推定器の構造体
- [`select_arm`](@ref): 行動選択関数
"""
function update_estimator!(estimator::Estimator, action::Int, reward::Float64)
    # 実装
end
```

#### 3.5.2 Documenter.jlによるドキュメント生成

```julia
# docs/make.jl
using Documenter
using ReinforcementLearningBandits

makedocs(
    sitename = "ReinforcementLearningBandits.jl",
    modules = [ReinforcementLearningBandits],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Tutorials" => [
            "tutorials/basic_simulation.md",
            "tutorials/parameter_estimation.md",
            "tutorials/custom_environments.md",
        ],
        "API Reference" => [
            "api/core.md",
            "api/estimation.md",
            "api/plotting.md",
        ],
        "Theory" => "theory.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

deploydocs(
    repo = "github.com/username/ReinforcementLearningBandits.jl.git",
)
```

### 3.6 テスト戦略

#### 3.6.1 テスト構造

```
test/
├── runtests.jl              # テストエントリーポイント
├── test_estimators.jl       # Estimatorのテスト
├── test_environments.jl     # Environmentのテスト
├── test_policies.jl         # Policyのテスト
├── test_agents.jl           # Agentのテスト
├── test_systems.jl          # Systemのテスト
├── test_histories.jl        # Historyのテスト
├── test_estimation.jl       # パラメータ推定のテスト
├── test_integration.jl      # 統合テスト
└── fixtures/                # テストデータ
    └── sample_data.csv
```

#### 3.6.2 テスト例

```julia
# test/test_estimators.jl
using Test
using ReinforcementLearningBandits

@testset "Estimator" begin
    @testset "Construction" begin
        @testset "default parameters" begin
            est = Estimator(4)
            @test est.n_arms == 4
            @test length(est.Q) == 4
            @test all(est.Q .== 0.5)
            @test all(est.N .== 0)
        end

        @testset "custom parameters" begin
            config = EstimatorConfig(
                learning = LearningParams(Q0=0.3, α=0.2)
            )
            est = Estimator(4; config=config)
            @test all(est.Q .== 0.3)
        end

        @testset "invalid parameters" begin
            @test_throws ArgumentError Estimator(0)
            @test_throws ArgumentError Estimator(4;
                config=EstimatorConfig(learning=LearningParams(α=-0.1)))
        end
    end

    @testset "Update" begin
        @testset "reward increases Q" begin
            est = Estimator(4)
            initial_Q = copy(est.Q)
            update_estimator!(est, 1, 1.0)
            @test est.Q[1] > initial_Q[1]
            @test est.N[1] == 1
        end

        @testset "no reward decreases Q" begin
            est = Estimator(4)
            initial_Q = copy(est.Q)
            update_estimator!(est, 1, 0.0)
            @test est.Q[1] < initial_Q[1]
        end

        @testset "asymmetric learning rates" begin
            config = EstimatorConfig(
                learning = LearningParams(α=0.5, αm=0.1)
            )
            est = Estimator(4; config=config)

            # 報酬ありの場合は大きく更新
            est1 = deepcopy(est)
            update_estimator!(est1, 1, 1.0)

            # 報酬なしの場合は小さく更新
            est2 = deepcopy(est)
            update_estimator!(est2, 1, 0.0)

            δ_reward = abs(est1.Q[1] - 0.5)
            δ_no_reward = abs(est2.Q[1] - 0.5)
            @test δ_reward > δ_no_reward
        end
    end

    @testset "Selection Probabilities" begin
        est = Estimator(4)
        est.W .= [0.2, 0.4, 0.6, 0.8]

        probs = selection_probabilities(est, 1:4)

        @test length(probs) == 4
        @test sum(probs) ≈ 1.0
        @test probs[4] > probs[3] > probs[2] > probs[1]
    end
end

@testset "Environment" begin
    @testset "Bernoulli Bandit" begin
        env = Environment([0.2, 0.5, 0.8])
        @test env.n_arms == 3

        # 多数回サンプリングして期待値を検証
        rewards = [sample_reward(env, 2) for _ in 1:10000]
        @test mean(rewards) ≈ 0.5 atol=0.05
    end

    @testset "Non-stationary Environment" begin
        env = NonStationaryEnvironment(
            [[Bernoulli(0.2), Bernoulli(0.8)],
             [Bernoulli(0.8), Bernoulli(0.2)]],
            [50]
        )

        # 変化点前
        @test mean(env)[1] ≈ 0.2

        # 試行を進める
        for _ in 1:50
            sample_reward(env, 1)
        end

        # 変化点後
        @test mean(env)[1] ≈ 0.8
    end
end
```

### 3.7 依存関係の整理

#### 3.7.1 コア依存とオプション依存の分離

```toml
# Project.toml
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[weakdeps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"

[extensions]
CairoMakieExt = "CairoMakie"
DataFramesExt = "DataFrames"
OptimExt = "Optim"

[compat]
julia = "1.9"
Distributions = "0.25"
SpecialFunctions = "2"
CairoMakie = "0.10, 0.11, 0.12"
DataFrames = "1"
Optim = "1"
```

#### 3.7.2 Package Extension実装

```julia
# src/extensions/CairoMakieExt.jl
module CairoMakieExt

using ReinforcementLearningBandits
using CairoMakie

# プロット関数の実装
function ReinforcementLearningBandits.plot_history(history::EstimatorHistory; kwargs...)
    # CairoMakieを使用した実装
end

function ReinforcementLearningBandits.plot_estimator_evolution(history::EstimatorHistory; kwargs...)
    # 実装
end

end # module
```

### 3.8 エラーハンドリングとバリデーション

```julia
"""パラメータバリデーション"""
function validate_config(config::EstimatorConfig)
    lp = config.learning
    sp = config.selection

    # 学習率の検証
    0.0 ≤ lp.α ≤ 1.0 || throw(ArgumentError(
        "Learning rate α must be in [0, 1], got $(lp.α)"))
    0.0 ≤ lp.αm ≤ 1.0 || throw(ArgumentError(
        "Learning rate αm must be in [0, 1], got $(lp.αm)"))
    0.0 ≤ lp.αf ≤ 1.0 || throw(ArgumentError(
        "Forgetting rate αf must be in [0, 1], got $(lp.αf)"))

    # 初期値の検証
    0.0 ≤ lp.Q0 ≤ 1.0 || throw(ArgumentError(
        "Initial Q value Q0 must be in [0, 1], got $(lp.Q0)"))

    # 逆温度の検証
    sp.β ≥ 0.0 || throw(ArgumentError(
        "Inverse temperature β must be non-negative, got $(sp.β)"))

    return true
end

"""カスタム例外型"""
struct InvalidParameterError <: Exception
    parameter::Symbol
    value::Any
    message::String
end

function Base.showerror(io::IO, e::InvalidParameterError)
    print(io, "InvalidParameterError: $(e.parameter) = $(e.value) - $(e.message)")
end
```

---

## 4. 推奨パッケージ構造

### 4.1 ディレクトリ構造

```
ReinforcementLearningBandits.jl/
├── .github/
│   └── workflows/
│       ├── CI.yml                    # 継続的インテグレーション
│       └── Documentation.yml         # ドキュメント自動デプロイ
├── docs/
│   ├── src/
│   │   ├── index.md                  # ホームページ
│   │   ├── getting_started.md        # 入門ガイド
│   │   ├── tutorials/
│   │   │   ├── basic_simulation.md
│   │   │   ├── parameter_estimation.md
│   │   │   └── custom_environments.md
│   │   ├── api/
│   │   │   ├── core.md
│   │   │   ├── estimation.md
│   │   │   └── plotting.md
│   │   └── theory.md                 # 理論的背景
│   └── make.jl
├── examples/
│   ├── basic_simulation.jl
│   ├── parameter_recovery.jl
│   └── nonstationary_bandit.jl
├── src/
│   ├── ReinforcementLearningBandits.jl  # メインモジュール
│   ├── Core/
│   │   ├── Core.jl                      # サブモジュール定義
│   │   ├── types.jl                     # 抽象型
│   │   ├── estimators.jl                # 推定器
│   │   ├── environments.jl              # 環境
│   │   ├── policies.jl                  # 方策
│   │   ├── agents.jl                    # エージェント
│   │   ├── histories.jl                 # 履歴
│   │   ├── systems.jl                   # システム
│   │   └── utility_functions.jl         # 効用関数
│   ├── Estimation/
│   │   ├── Estimation.jl
│   │   └── mle.jl
│   ├── Utils/
│   │   ├── Utils.jl
│   │   ├── sampling.jl
│   │   └── validation.jl
│   ├── Presets/
│   │   └── Presets.jl                   # 設定プリセット
│   └── extensions/
│       ├── CairoMakieExt.jl
│       ├── DataFramesExt.jl
│       └── OptimExt.jl
├── test/
│   ├── runtests.jl
│   ├── test_estimators.jl
│   ├── test_environments.jl
│   ├── test_policies.jl
│   ├── test_agents.jl
│   ├── test_systems.jl
│   ├── test_histories.jl
│   ├── test_estimation.jl
│   └── test_integration.jl
├── Project.toml
├── README.md
├── LICENSE
└── CHANGELOG.md
```

### 4.2 メインモジュール

```julia
# src/ReinforcementLearningBandits.jl
module ReinforcementLearningBandits

# バージョン
const VERSION = v"0.1.0"

# 必須依存
using Random
using Statistics
using LinearAlgebra
using Distributions
using SpecialFunctions

# サブモジュール
include("Core/Core.jl")
include("Estimation/Estimation.jl")
include("Utils/Utils.jl")
include("Presets/Presets.jl")

# サブモジュールのシンボルを再エクスポート
using .Core
using .Estimation
using .Utils

# 公開シンボルのエクスポート
# 型
export AbstractEstimator, AbstractEnvironment, AbstractPolicy
export AbstractHistory, AbstractSystem
export Estimator, EmptyEstimator
export Environment, NonStationaryEnvironment
export SoftmaxPolicy, RandomResponding
export Agent
export History, EstimatorHistory
export System
export SimulationResult

# 設定
export EstimatorConfig, LearningParams, UtilityParams
export StickinessParams, SelectionParams

# 関数
export simulate!, simulate_bandit
export update_estimator!, select_arm
export sample_reward, record_trial!
export estimate_parameters_mle

# プリセット（サブモジュールとして）
export Presets

# Package Extensionで提供される関数のスタブ
function plot_history end
function plot_estimator_evolution end

end # module
```

---

## 5. 実装ガイドライン

### 5.1 コーディング規約

#### 5.1.1 命名規則

| 種類 | 規則 | 例 |
|------|------|-----|
| モジュール | PascalCase | `Core`, `Estimation` |
| 型 | PascalCase | `Estimator`, `SimulationResult` |
| 関数 | snake_case | `simulate_bandit`, `update_estimator!` |
| 定数 | SCREAMING_SNAKE_CASE | `DEFAULT_LEARNING_RATE` |
| 変数 | snake_case | `n_arms`, `reward_probs` |
| 破壊的関数 | 末尾に`!` | `update!`, `record!` |

#### 5.1.2 ファイル構成規則

- 1ファイル1主要型（関連するヘルパー関数は同居可）
- ファイル名は含まれる主要型/機能を反映
- サブモジュールディレクトリはPascalCase

### 5.2 バージョニング

Semantic Versioning (SemVer) に従う：

- **MAJOR**: 後方互換性のない変更
- **MINOR**: 後方互換性のある機能追加
- **PATCH**: 後方互換性のあるバグ修正

### 5.3 Git ワークフロー

```
main           # 安定版リリース
├── develop    # 開発統合ブランチ
│   ├── feature/xxx    # 機能開発
│   ├── fix/xxx        # バグ修正
│   └── refactor/xxx   # リファクタリング
└── release/vX.Y.Z     # リリース準備
```

### 5.4 CI/CD設定

```yaml
# .github/workflows/CI.yml
name: CI
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        julia-version: ['1.9', '1.10', '1']
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v1
        with:
          version: ${{ matrix.julia-version }}
      - uses: julia-actions/cache@v1
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
      - uses: julia-actions/julia-processcoverage@v1
      - uses: codecov/codecov-action@v3

  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v1
        with:
          version: '1'
      - run: |
          julia --project=docs -e '
            using Pkg
            Pkg.develop(PackageSpec(path=pwd()))
            Pkg.instantiate()'
      - run: julia --project=docs docs/make.jl
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          DOCUMENTER_KEY: ${{ secrets.DOCUMENTER_KEY }}
```

---

## 6. アクションプラン

### 6.1 フェーズ1: 基盤整備（1-2週間）

**目標**: パッケージとして動作する最小構成を作成

| # | タスク | 優先度 | 見積時間 |
|---|--------|--------|----------|
| 1.1 | 新パッケージ名の決定 | 高 | 1日 |
| 1.2 | 基本ディレクトリ構造の作成 | 高 | 1日 |
| 1.3 | Project.toml/Manifest.tomlの整備 | 高 | 1日 |
| 1.4 | メインモジュール構造の実装 | 高 | 2日 |
| 1.5 | 既存コードのモジュールへの移行 | 高 | 3日 |
| 1.6 | export定義の追加 | 高 | 1日 |

### 6.2 フェーズ2: 品質向上（2-3週間）

**目標**: テストとドキュメントの整備

| # | タスク | 優先度 | 見積時間 |
|---|--------|--------|----------|
| 2.1 | 基本テストの作成（Core） | 高 | 3日 |
| 2.2 | 統合テストの作成 | 高 | 2日 |
| 2.3 | 主要関数へのdocstring追加 | 高 | 3日 |
| 2.4 | README.mdの作成 | 高 | 1日 |
| 2.5 | Getting Startedガイドの作成 | 中 | 2日 |
| 2.6 | CI/CD設定 | 中 | 1日 |

### 6.3 フェーズ3: API改善（1-2週間）

**目標**: 使いやすいAPIの提供

| # | タスク | 優先度 | 見積時間 |
|---|--------|--------|----------|
| 3.1 | パラメータ構造体の再設計 | 中 | 2日 |
| 3.2 | 高レベルAPI（simulate_bandit等）の実装 | 中 | 2日 |
| 3.3 | プリセット設定の実装 | 低 | 1日 |
| 3.4 | エラーハンドリング強化 | 中 | 2日 |

### 6.4 フェーズ4: 拡張機能（2-3週間）

**目標**: オプション機能の整備

| # | タスク | 優先度 | 見積時間 |
|---|--------|--------|----------|
| 4.1 | Package Extensionの実装 | 中 | 3日 |
| 4.2 | プロット機能の分離・整備 | 低 | 3日 |
| 4.3 | DataFrames統合の改善 | 中 | 2日 |
| 4.4 | 詳細ドキュメント（API Reference） | 中 | 3日 |
| 4.5 | チュートリアルの作成 | 低 | 3日 |

### 6.5 フェーズ5: 公開準備（1週間）

**目標**: GitHub公開とJulia General Registryへの登録

| # | タスク | 優先度 | 見積時間 |
|---|--------|--------|----------|
| 5.1 | LICENSEファイルの追加 | 高 | 1日 |
| 5.2 | CHANGELOG.mdの作成 | 中 | 1日 |
| 5.3 | GitHubリポジトリの作成・設定 | 高 | 1日 |
| 5.4 | General Registryへの登録申請 | 中 | 1日 |
| 5.5 | ドキュメントサイトのデプロイ | 中 | 1日 |

---

## 付録

### A. 参考リソース

- [Pkg.jl Documentation](https://pkgdocs.julialang.org/v1/)
- [Julia Package Development Guide](https://julialang.github.io/Pkg.jl/v1/creating-packages/)
- [Documenter.jl](https://documenter.juliadocs.org/stable/)
- [JuliaRegistries/General](https://github.com/JuliaRegistries/General)
- [Wilson & Collins (2019)](https://elifesciences.org/articles/49547)

### B. 推奨パッケージ名候補

| 名前 | 長所 | 短所 |
|------|------|------|
| `BanditRL.jl` | 短い、明確 | 一般的すぎる可能性 |
| `ReinforcementLearningBandits.jl` | 説明的、検索しやすい | やや長い |
| `CognitiveRL.jl` | 認知科学的側面を強調 | 範囲が広すぎる |
| `BehavioralBandits.jl` | 行動モデリングを強調 | ニッチすぎる可能性 |
| `QLearningBandits.jl` | 技術的に正確 | Q学習以外も含む場合は不適切 |

### C. 類似パッケージ

- [ReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl)
- [Bandits.jl](https://github.com/rawls238/Bandits.jl)
- [MultiArmedBandits.jl](https://github.com/UnofficialJuliaMirror/MultiArmedBandits.jl)

これらとの差別化ポイント：
- 認知科学・心理学向けのモデリング
- パラメータ推定機能の統合
- Wilson & Collins (2019) の実装

---

*最終更新: 2026-01-30*

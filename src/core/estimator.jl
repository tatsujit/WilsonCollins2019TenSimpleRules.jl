# import Pkg; Pkg.add("SpecialFunctions")
using SpecialFunctions

abstract type AbstractEstimator end

struct EmptyEstimator <: AbstractEstimator end # for random responding, choice kernel (only), etc. 

"""
TODO: (nullでない) パラメータのリストを NamedTuple として持っておくと便利
"""
mutable struct Estimator <: AbstractEstimator

    # 22 parameters as of 2026-01-06
    
    # params 1-5
    Q::Vector{Float64} # 価値（通常のQ値や平均報酬など）
    V::Vector{Float64} # 効用 (utility function でQ値を変調したもの；これにβをかける)
    C::Vector{Float64} # 各腕に対する「固執性」
    W::Vector{Float64} # 総合的な「行動価値」 W = βV + φC でありこれを Policy で行動選択に使う
    N::Vector{Int64} # 行動選択の回数（頻度）
    # params 6-10
    Q0::Float64 # Q値の初期値、0.0だと普通のQ値
    α::Float64 # 学習率 (if α=-1.0 Qは報酬平均 else RPE正の場合の学習率)
    αm::Float64 # RPE負の場合の学習率 (-1.0なら vanish)
    β::Float64 # 行動価値に対する逆温度 (policy の方のβは 1.0 とすることを想定) (1.0ならvanish)
    αf::Float64 # 忘却率 (0.0なら vanish)
    # params 11-15
    μ::Float64 # 忘却デフォルト値 (0.0 なら vanish)
    τ::Float64 # stickiness 固執性 (0.0 なら基本 vanish)
    φ::Float64 # 固執性の逆温度 (0.0 なら τ も含めて vanish)
    C0::Float64 # 固執性の初期値 (0.0 なら vanish)
    η::Float64 # ベータ分布のパラメータ (η=α/(α+β); expectation, 0.5 なら ν = 2.0 のときvanish)
    # params 16-20
    ν::Float64 # ベータ分布のパラメータ (ν=α+β; precision, 2.0 なら η = 0.5 のときvanish)
    γ::Float64 # ハイブリッドの場合の満足化の重み (0.0 ならvanish, 1.0 なら満足化のみ)
    n_params::Int64 # （実効的な）モデルのパラメータ数
    utility_function::Function
    betaCDF::Bool
    # params 21-22
    hybrid::Bool # hybrid ? W=βQ+γV : W=βV+φC
    params::Dict{Symbol, Float64} # パラメータの名前と値の辞書

    function Estimator(n_arms::Int;
                       Q0::Float64=0.0,
                       α::Float64=-1.0, αm::Float64=-1.0,
                       β::Float64=1.0,
                       αf::Float64=0.0, μ::Float64=0.0,
                       τ::Float64=0.0, φ::Float64=0.0, C0::Float64=0.0,
                       η::Float64=0.5, ν::Float64=2.0,
                       γ::Float64=0.0, # hybrid の場合の γ
                       utility_function::Function = x->x,
                       # utility_function::Function,
                       betaCDF::Bool=true, # βCDFを使うかどうか
                       hybrid::Bool=false # hybrid かどうか
                       )
        Q = fill(Q0, n_arms) # 行動価値
        N = zeros(n_arms) # 行動の選択頻度
        C = fill(C0, n_arms) # 固執性
        params = (Q0=Q0, α=α, αm=αm, β=β, # 学習系
            αf=αf, μ=μ, # 忘却系
            τ=τ, φ=φ, C0=C0, # 固執性系
            η=η, ν=ν) # 効用関数系
        # n_params = count_params(Q0, α, αm, β, αf, μ, τ, φ, C0, η, ν)
        n_params = count_params(params...)
        # (η, ν) を使って BetaCDF utility function にするかどうか
        if betaCDF
            if η != 0.5 || ν != 2.0
                uf = x -> beta_cdf(x, η, ν)
            else
                uf = x -> x # identity function
            end
        else
            # betaCDF以外には sigmoid やステップ関数などを想定。その場合ηが原点、νがゲイン
            uf = x -> utility_function(x, η, ν)
        end
        V = uf.(Q) # 行動選択価値（効用）
        if hybrid
            W = β * ((1-γ) * Q + γ * V) # 最適化 (maxQ) と満足化 (V) のハイブリッドというイメージ
        else
            W = β*V + φ*C # 固執性ありの総合的行動価値
        end
        new(
            Q, V, C, W, N, # 1-5: data vectors 
            Q0, α, αm, β, αf, μ, # 6-11: Q, D, and F params
            τ, φ, C0, η, ν, γ, n_params, # 12-18: C, U params
            uf, # 19: utility_function
            betaCDF, hybrid, # 20-21: whether to use betaCDF utility function and hybrid
            Dict(pairs(params)) # 22: params dict
        )
    end
end
"""
all the parameters of the model as a NamedTuple
"""
function parameters(est::Estimator)
    return (Q0=est.Q0, α=est.α, αm=est.αm, β=est.β, # 学習系
            αf=est.αf, μ=est.μ, # 忘却系
            τ=est.τ, φ=est.φ, C0=est.C0, # 固執性系
            η=est.η, ν=est.ν, γ=est.γ) # 効用関数系
end
"""
algorithm_parameters_default_values
"""
function algo_params_default_vals(m::Estimator)
    est = Estimator(2)
    (Q0=est.Q0, α=est.α, αm=est.αm, β=est.β, # 学習系
     αf=est.αf, μ=est.μ, # 忘却系
     τ=est.τ, φ=est.φ, C0=est.C0, # 固執性系
     η=est.η, ν=est.ν, γ=est.γ) # 効用関数系
end
"""
The number of free parameters in the model.
"""
function count_params(Q0::Float64, α::Float64, αm::Float64,
               β::Float64, αf::Float64, μ::Float64,
               τ::Float64, φ::Float64, C0::Float64,
               η::Float64, ν::Float64)
    singles = count([Q0≠0, α≠-1, αm≠-1, β≠1, αf≠0, μ≠0, C0≠0])
    pairs = 2count([τ≠0||φ≠0, η≠0.5||ν≠2])
    return singles + pairs
end

function toString(e::Estimator)
    str1 = "Estimator(Q0=$(e.Q0), α=$(e.α), αm=$(e.αm), β=$(e.β), αf=$(e.αf), "
    str2 = "μ=$(e.μ), τ=$(e.τ), φ=$(e.φ), C0=$(e.C0), "
    str3 = "η=$(e.η), ν=$(e.ν), n_params=$(e.n_params))"
    return str1 * str2 * str3
end
"""
display() したときの表示
"""
function Base.show(io::IO, ::MIME"text/plain", obj::Estimator)
    toString(obj)
end

# 高速なベータ分布CDF計算
function beta_cdf(x, η, ν)
    if x <= 0.0
        return 0.0
    elseif x >= 1.0
        return 1.0
    else
        a = beta_a(η, ν)
        b = beta_b(η, ν)
        # if a <= 0.0 || b <= 0.0 || η <= 0.0 || ν <= 0.0
        #     @warn "a or b is negative or zero: a=$(a), b=$(b), η=$(η), ν=$(ν)"
        #     return 0.0 # または 1.0 とするか？
        # end
        return beta_inc(a, b, x)[1] # 正規化不完全ベータ関数を使用
    end
end

function beta_cdf_curried(η, ν)
    x -> beta_cdf(x, η, ν)
end

"""
from a/(a+b) and (a+b) to a = a/(a+b) * (a+b)
"""
beta_a(η, ν) = η * ν
"""
from a/(a+b) and (a+b) to b = (1-a/(a+b)) * (a+b)
"""
beta_b(η, ν) = (1-η)*ν
beta_mean(a, b) = a / (a+b)
beta_η(a, b) = a / (a+b)
beta_ν(a, b) = a+b
beta_median(a, b) = (a-1) / (a+b-2)

# EmptyEstimator does nothing on update
function update!(e::EmptyEstimator, action::Int, reward::Float64)
    # EmptyEstimator doesn't maintain any state, so nothing to update
    return nothing
end

function update!(e::Estimator, action::Int, reward::Float64)
    n_arms = length(e.Q)
    e.N[action] += 1
    # 行動価値Q (報酬推定値) の更新
    if e.α == -1.0
        # 報酬平均を価値Qにする
        e.Q[action] = (e.Q[action] * (e.N[action] - 1) + reward) / e.N[action]
    else # 固定学習率を持つ
        # RPE (reward prediction error) を計算
        δ = reward - e.Q[action]
        if e.αm == -1.0
            # 固定学習率 α での対称な更新
            e.Q[action] += e.α * δ
        else # e.αm != -1.0
            # RPE正負非対称学習率での更新
            if δ >= 0.0
                # RPE正の場合の学習率
                e.Q[action] += e.α * δ
            else
                # RPE負の場合の学習率
                e.Q[action] += e.αm * δ
            end
        end
    end
    # TODO 忘却
    if e.αf != 0.0
        for a in 1:n_arms
            # 忘却率 αf による更新
            # 忘却は選択していない行動の行動価値にのみ適用
            if a != action
                # μ=0 の場合は、 Q = (1-αf) * Q
                e.Q[a] = e.Q[a] + e.αf * (e.μ - e.Q[a])
                #        = (1 - e.αf) * e.Q[a] + e.αf * e.μ
                # e.Q[a] += e.αf * (e.μ - e.Q[a])
                # e.Q[a] = (1 - e.αf) * e.Q[a] # これはμが0のときの特殊形だが試してみる
                # e.Q[a] -= e.αf * e.Q[a] # これはμが0のときの特殊形だが試してみる
            end
        end
    end
    # 効用関数による変調 (Q → V=u(Q))
    if e.betaCDF
        if e.η != 0.5 || e.ν != 2.0 # if e.a != 1.0 || e.b != 1.0
            e.V[action] = e.utility_function(e.Q[action])
            # 選択してフィードバックを受けて価値を更新する腕以外も効用を
            # アップデートしなくてはならないのは、忘却が入っている場合
            # TODO: 忘却のみか？　他では必要ないか？
        else
            e.V[action] = e.Q[action] # 効用関数が identity な場合はそのまま
        end
        # 忘却するなら他のすべての腕の価値も効用に変換する
        if e.αf != 0.0
            for a in 1:n_arms
                if a != action
                    e.V[a] = e.utility_function(e.Q[a])
                end
            end
        end
    else # betaCDF 以外の効用関数が指定されている場合
        e.V[action] = e.utility_function(e.Q[action], e.η, e.ν)
        if e.αf != 0.0
            for a in 1:n_arms
                if a != action
                    e.V[a] = e.utility_function(e.Q[a], e.η, e.ν)
                end
            end
        end
    end
    # 固執性 C の更新
    # NOTE: e.τ == 0.0 でも、 e.C0 != 0.0 ならば、 e.φ * e.C[a] = e.φ * e.C0 が入っている
    if e.τ != 0.0 && e.φ != 0.0 # e.τ==0.0ならばC[]の更新不要、e.φ==0.0ならばC[]は行動選択に影響なし
        e.C[action] += e.τ * (1 - e.C[action])
        for a in 1:n_arms
            if a != action
                e.C[a] += - e.τ * e.C[a]
            end
        end
    end
    # 総合的な行動価値 W=βV+φC の更新
    for a in 1:n_arms
        e.W[a] = e.β * e.V[a] + e.φ * e.C[a]
    end
end


"""
W は K 行動それぞれの（総合的）行動価値ベクトル [W[1], W[2], ..., W[K]] == [1.2, 0.5, ..., 1.0, ]
available_arms はその trial で選択可能だった腕（インデックス）のベクトル、例: [9, 2, 5, 1, 4]
総合的、というのは、 W = βV + φC であり、V は効用関数で変調された（かもしれない） Q 値、 C は固執性を表すベクトル

戻り値については、例えば確率ベクトル：
[ 0.1,  0.2,  0.0,  0.1,  0.3,  0.0,  0.0,  0.0,  0.3]
を返す。これは、 [9, 2, 5, 1, 4]に対応するブールベクトル
[true, true,false, true, true,false,false,false, true]
に対応している。

こうすることでデバッグも可能である。つまり、尤度が0.0になるようであれば、
available_arms 以外の腕が選ばれているということになる。
"""
function selection_probabilities(W::Vector{Float64}, available_arms::Vector{Int})
    n_arms = length(W)
    # 全ての腕に対する確率ベクトルを初期化（全て0.0）
    probs = zeros(Float64, n_arms)
    # 利用可能な腕のみで softmax を計算
    available_exp_values = [exp(W[arm]) for arm in available_arms]
    normalization_factor = sum(available_exp_values)
    # 利用可能な腕に確率を割り当て
    for (i, arm) in enumerate(available_arms)
        probs[arm] = available_exp_values[i] / normalization_factor
    end
    return probs
end

function selection_probabilities(W::Vector{Float64})
    n_arms = length(W)
    proto_probs = [exp(w) for w in W]
    return proto_probs / sum(proto_probs)
end

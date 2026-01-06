# all the (currently) 11 parameters of the "Estimator"

struct EstimatorData
    Q::Vector{Float64} # 価値（通常のQ値や平均報酬など）
    V::Vector{Float64} # 効用 (utility function でQ値を変調したもの；これにβをかける)
    C::Vector{Float64} # 各腕に対する「固執性」
    W::Vector{Float64} # 総合的な「行動価値」 W = βV + φC でありこれを Policy で行動選択に使う
    N::Vector{Int64} # 行動選択の回数（頻度）
end

struct EstimatorParameters
    Q0::Float64 # Q値の初期値、0.0だと普通のQ値
    α::Float64 # 学習率 (if α=-1.0 Qは報酬平均 else RPE正の場合の学習率)
    αm::Float64 # RPE負の場合の学習率 (-1.0なら vanish)
    β::Float64 # 行動価値に対する逆温度 (policy の方のβは 1.0 とすることを想定) (1.0ならvanish)
    αf::Float64 # 忘却率 (0.0なら vanish)
    μ::Float64 # 忘却デフォルト値 (0.0 なら vanish)
    τ::Float64 # stickiness 固執性 (0.0 なら基本 vanish)
    φ::Float64 # 固執性の逆温度 (0.0 なら τ も含めて vanish)
    C0::Float64 # 固執性の初期値 (0.0 なら vanish)
    η::Float64 # ベータ分布のパラメータ (η=α/(α+β); expectation, 0.5 なら ν = 2.0 のときvanish)
    ν::Float64 # ベータ分布のパラメータ (ν=α+β; precision, 2.0 なら η = 0.5 のときvanish)
    utility_function::Function
    betaCDF::Bool
end

struct EstimatorMembers
    ep::EstimatorParameters # EstimatorParameters のインスタンス
    ed::EstimatorData # EstimatorData のインスタンス
end

defaultEstimatorParameters = Dict(
    :n_arms => 1, # simulation with n_arms = 1 can show the development of Q values and others
    :Q0 => 0.0,
    :α => -1.0,
    :αm => -1.0,
    :β => 1.0,
    :αf => 0.0,
    :μ => 0.0,
    :τ => 0.0,
    :φ => 0.0,
    :C0 => 0.0,
    :η => 0.5,
    :ν => 2.0,
    :utility_function => x -> x,
    :betaCDF => true
)

defaultEstimatorParameters_nt = (
    n_arms = 1,
    Q0 = 0.0,
    α = -1.0,
    αm = -1.0,
    β = 1.0,
    αf = 0.0,
    μ = 0.0,
    τ = 0.0,
    φ = 0.0,
    C0 = 0.0,
    η = 0.5,
    ν = 2.0,
    utility_function = x -> x,
    betaCDF = true
)



# All the 11 parameters
# parameters = [:Q0, :α, :αm, :β, :αf, :μ, :τ, :φ, :C0, :η, :ν]

parameters = [
    # :Q0,
    :α
    # , :αm
    , :β
    , :αf#, :μ
    # , :τ, :φ
    # , :C0
    # ,:η, :ν
]
@ic parameters

sampling_dists = Dict(
    :Q0 => Beta(1.1, 1.1),
    :α => Uniform(0.0, 1.0),
    # :α => Beta(1.1, 1.1),
    :αm => Beta(1.1, 1.1),
    # :β => Gamma(0.5, 50.0),
    # :β => Exponential(10.0),
    :β => Uniform(0.01, 10.0),
    :αf => Uniform(0.0, 1.0),
    # :αf => Beta(1.1, 1.1),
    :μ => Beta(1.1, 1.1),
    :τ => Beta(1.1, 1.1),
    :φ => Normal(0.0, 10.0),
    :C0 => Beta(1.1, 1.1),
    :η => Beta(1.1, 1.1),
    :ν => Gamma(0.5, 50.0)
)

sampling_ranges = Dict(
    :Q0 => [0.0, 1.0],
    :α => [0.0, 1.0],
    :αm => [0.0, 1.0],
    :β => [0.01, 10.0],
    :αf => [0.0, 1.0],
    :μ => [0.0, 1.0],
    :τ => [0.0, 1.0],
    :ϕ => [-20.0, 20.0],
    :C0 => [0.0, 1.0],
    :η => [0.0, 1.0],
    :ν => [1.0, 100.0]
)

unit_ranged_parameters = [:Q0, :α, :αm, :αf, :μ, :τ, :C0]
non_unit_ranged_parameters = [:β, :φ, :ν]

# TODO: estimation_ranges != sampling_ranges としたい場合もあるかも？
estimation_ranges = Dict(
    :Q0 => [0.0, 1.0],
    :α => [0.0, 1.0],
    :αm => [0.0, 1.0],
    :β => [0.01, 20.0], # ここだけ10.0ではなく20.0にしている
    :αf => [0.0, 1.0],
    :μ => [0.0, 1.0],
    :τ => [0.0, 1.0],
    :ϕ => [-20.0, 20.0],
    :C0 => [0.0, 1.0],
    :η => [0.0, 1.0],
    :ν => [1.0, 100.0]
)
# ranges は parameters を keys として、estimation_ranges の値を持つ辞書
ranges = OrderedDict(zip(parameters, [estimation_ranges[param] for param in parameters]))

prior_dists = Dict(
    :Q0 => Beta(1.1, 1.1), # Normal(0.5, 0.1),
    :α => Beta(1.1, 1.1),
    :αm => Beta(1.1, 1.1),
    :β => Gamma(0.5, 50.0),
    :αf => Beta(1.1, 1.1),
    :μ => Uniform(0.0, 1.0),
    :τ => Uniform(0.0, 1.0),
    :φ => Uniform(-10.0, 10.0),
    :C0 => Uniform(0.0, 1.0),
    :η => Beta(1.1, 1.1),
    :ν => Gamma(5.0, 10.0)
)


"""
all the parameters of the model as a NamedTuple
"""
function parameters(est::Estimator)
    return (Q0=est.Q0, α=est.α, αm=est.αm, β=est.β, # 学習系
            αf=est.αf, μ=est.μ, # 忘却系
            τ=est.τ, φ=est.φ, C0=est.C0, # 固執性系
            η=est.η, ν=est.ν) # 効用関数系
end

"""
algorithm_parameters_default_values
"""
function algo_params_default_vals(m::Estimator)
    est = Estimator(1)
    (Q0=est.Q0, α=est.α, αm=est.αm, β=est.β, # 学習系
     αf=est.αf, μ=est.μ, # 忘却系
     τ=est.τ, φ=est.φ, C0=est.C0, # 固執性系
     η=est.η, ν=est.ν) # 効用関数系
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
    str1 = "Estimator(Q0=$(e.Q0), α=$(e.α), αm=$(e.αm), β=$(e.β), αf=$(e.αf),\n"
    str2 = "          μ=$(e.μ), τ=$(e.τ), φ=$(e.φ), C0=$(e.C0),\n"
    str3 = "          η=$(e.η), ν=$(e.ν), n_params=$(e.n_params))"
    return str1 * str2 * str3
end

"""
display() したときの表示
"""
function Base.show(io::IO, ::MIME"text/plain", obj::Estimator)
    toString(obj)
end


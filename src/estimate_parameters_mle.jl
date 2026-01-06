import ProgressMeter: Progress, next!, @timed
using Optim
using IceCream

function estimate_parameters_mle(bandit_data::DataFrame, true_params_df,
                                 parameters, estimation_ranges, n_opt::Int; verbose::Bool=true)
    n_arms = length(bandit_data.Expectations[1])
    n_params = length(parameters)
    trials = size(bandit_data, 1)
    # @assert bandit_data[:Trial] == 1:trials "Trial column must be 1:trials"

    # スレッド数を確認
    if verbose
        @info "Number of threads: $(Threads.nthreads())"
    end

    # 推定結果の容れ物を、拡張により準備
    est_params_df = similar(true_params_df, 0)  # 0行の空のDataFrameを作成、列は true_params_df と同じ
    rename!(est_params_df, [name => Symbol(string(name) * "_est") for name in names(est_params_df)]) # :α を :α_est のように変更

    # 最適化の設定（最適化範囲）
    lower, upper = first.(estimation_ranges), last.(estimation_ranges)

    ################################################################
    # target function definition
    ################################################################
    function negative_log_likelihood(params)
        actions = bandit_data[!, :Action]
        rewards = bandit_data[!, :Reward]
        wValuesArray = zeros(trials, n_arms) # W = βV +φC 「総合的行動価値」
        named_params = NamedTuple(zip(parameters, params))
        est = Estimator(n_arms; named_params...)

        for trial in 1:trials
            wValuesArray[trial, :] = est.W
            action = actions[trial]
            reward = rewards[trial]
            update!(est, action, reward)
        end

        probabilities = [selection_probabilities(wValuesArray[trial, :])
                         for trial in 1:trials]
        log_probs = [log(probabilities[trial][actions[trial]]) for trial in 1:trials]
        log_likelihood = sum(log_probs)

        return -log_likelihood # 負の対数尤度を返す（最小化するため）
    end

    ################################################################
    # optimization
    ################################################################
    results = Vector{Any}(undef, n_opt) # 各ケースに対する最適化結果を格納
    opt_params = Vector{Any}(undef, n_opt) # minimizer (最適化されたパラメータ) を格納
    opt_param = Vector{Float64}(undef, n_params) # 最適化されたパラメータの一時的な格納

    timed_result = @timed begin

        Threads.@threads for i_opt in 1:n_opt # case in case_values の方がスッキリだけど、1からの連番とは限らない
            p = Progress(n_opt; desc="Processing: ", dt=0.5) # スレッドセーフなプログレスバーを作成
            # randomize initial guess for parameters
            initial_guess = [uniform_or_constant(lower[k], upper[k]) for k in 1:length(lower)]
            # check an initial guess
            # if flag2 @ic initial_guess; flag2 = false; end
            # optimize the negative log likelihood function
            results[i_opt] = optimize(negative_log_likelihood,
                                      lower, upper, initial_guess)
            opt_params[i_opt] = results[i_opt].minimizer

            # Progress の進捗を安全に更新
            next!(p)
        end

        min_index = argmin([r.minimum for r in results])
        opt_param = opt_params[min_index]
        push!(est_params_df, opt_param)
    end
    # for opt_param in opt_params
    #     # @ic opt_param
    #     # @ic est_params_df
    #     push!(est_params_df, opt_param)
    # end

    # @ic true_params_df
    # @ic est_params_df

    estimation_df = hcat(true_params_df, est_params_df)
    if verbose
        @info "Estimated parameters: $(estimation_df)"
    end

    return (estimation_df, timed_result)
end
"""
calculate correlations between :param (true) and :param_est (estimated)
"""
function correlation_true_est(estimation_df)
    df = estimation_df
    # Exclude the :Agent column if there exists
    if :Agent in names(df)
        df = select(df, Not(:Agent))
    end

    # Identify true parameter columns
    true_cols = [col for col in names(df) if !occursin("_est", string(col))]
    valid_pairs = [(Symbol(col), Symbol(string(col) * "_est")) for col in true_cols]

    # Calculate correlations and store results
    rs = Float64[]
    for (true_col, est_col) in valid_pairs
        push!(rs, cor(df[!, true_col], df[!, est_col]))
    end

    return rs
end

import ProgressMeter: Progress, next!, @timed
using Optim
using IceCream
"""
    Maximum likelihood estimation function for bandit parameters

    Arguments:
    - data: DataFrame containing simulation/experiment data (plural cases/samples)
    - true_params_df: DataFrame with true parameters (used for structure)
    - estimator: Estimator
    - policy: Policy (currently unused)
    - parameters: Vector of parameter symbols
    - ranges: Vector of parameter ranges for optimization
    - n_opt: Number of optimization attempts per simulation

    Returns:
    - estimation_df: DataFrame with true (e.g., :α) and estimated parameters (e.g., :α_est)
    - timed_result: Timing result of the estimation process

- パラメータリカバリならば、true_params_df がある
- パラメータ推定ならば、true_params_df はない
- 推定結果は、:α に対して :α_est のように列名が変更される
- :Case 列は実験データも人工データも同じように扱う (実験データの :id は :Case に rename することにしよう)
"""
flag = true # for debugging
# function estimate_parameters_mle(data, true_params_df, estimator, policy, parameters, ranges, n_opt)
function estimate_parameters_mle(n_arms, data, true_params_df, estimator, policy, parameters, ranges, n_opt)
    n_params = length(parameters)

    let case_values = sort(unique(data[!, :Case])) # :id
        n_cases = length(case_values)
        println("cases: ", n_cases)

        # スレッド数を確認
        println("Running with $(Threads.nthreads()) threads")

        # est_params_df: 推定結果を格納するDataFrame (:α に対して :α_est が推定値)
        # @ic true_params_df
        # est_params_df = select(similar(true_params_df, 0), Not(:Agent))  # 0行の空のDataFrameを作成、列は true_params_df と同じ
        est_params_df = similar(true_params_df, 0)  # 0行の空のDataFrameを作成、列は true_params_df と同じ
        rename!(est_params_df, [name => Symbol(string(name) * "_est") for name in names(est_params_df)]) # :α を :α_est のように変更

        # results が、 each :Case に対する best params (候補) を入れるので、 n_cases * n_opt 個の要素を持つ
        # results = Vector{Any}(undef, n_opt)
        # results = fill(nothing, n_cases, n_opt)
        results = Matrix{Any}(undef, n_cases, n_opt)
        # results = Dict{Tuple{Int, Int}, Any}() # (case, i_opt) => result

        # flag = true

        flag1 = true
        flag2 = true


        function negative_log_likelihood(params, data, n_arms)
            # if true @ic typeof(params); end #=> Vector{Float64}
            trials = size(data, 1)
            actions = data[!, :Action]
            rewards = data[!, :Reward]
            # available_arms_array = string_to_vector.(data[!, :AvailableArms])
            # シミュレートされた wValuesArray を実際の `actions` と `rewards` から生成

            # wValues とは W = βV +φC で、これを総合的行動価値として扱う
            wValuesArray = zeros(trials, n_arms)
            # TODO ここで入れ替わっちゃったりしていないか？　順番が crucial
            # named_params = NamedTuple(Dict(zip(parameters, params)))
            named_params = NamedTuple(zip(parameters, params))
            # if flag
            #     @ic named_params
            #     flag = false
            # end

            est = estimator(n_arms; named_params...)

            for trial in 1:trials
                wValuesArray[trial, :] = est.W
                action = actions[trial]
                reward = rewards[trial]
                update!(est, action, reward)
            end

            # probabilities = [selection_probabilities(wValuesArray[i, :], available_arms_array[i])
            #                  for i in 1:size(wValuesArray, 1)]
            probabilities = [selection_probabilities(wValuesArray[trial, :])
                             for trial in 1:trials]

            # log_probs = [log(probabilities[trial][actions[trial]]) for trial in 1:trials]
            log_probs = [log(probabilities[trial][actions[trial]]) for trial in 1:trials]
            log_likelihood = sum(log_probs)

            return -log_likelihood # 負の対数尤度を返す（最小化するため）
        end
        ################################################################
        # optimization
        ################################################################
        # opt_params = Vector{Vector{Float64}}(undef, n_cases, n_params)
        opt_params = Matrix{Float64}(undef, n_cases, n_opt)
        # opt_params = Vector{Float64}(undef, n_cases)

        timed_result = @timed begin
            # スレッドセーフなプログレスバーを作成
            p = Progress(n_cases; desc="Processing: ", dt=0.5)
            # 各シミュレーションを処理するループ
            Threads.@threads for i in 1:n_cases # case in case_values の方がスッキリだけど、1からの連番とは限らないので
                case = case_values[i]
                # 現在の case 値に対するデータをフィルタリング
                case_data = filter(row -> row[:Case] == case, data)
                # TODO
                # n_arms = getMaxPNumber(case_data)
                lower, upper = first.(ranges), last.(ranges)
                if flag1 @ic lower, upper; flag1 = false; end

                # TODO: そうだ、最適化の回数と、最適化の回数を変数にしたものと、両方のループを考えていたのだな
                # TODO: こうすると何かと複雑になりすぎるので、単純な部品から作った方が良いのではないだろうか？
                for i_opt in 1:n_opt

                    # randomize initial guess for parameters
                    initial_guess = [uniform_or_constant(lower[k], upper[k]) for k in 1:length(lower)]
                    # check an initial guess
                    if flag2 @ic initial_guess; flag2 = false; end
                    # optimize the negative log likelihood function
                    results[i, i_opt] = optimize(params -> negative_log_likelihood(params, case_data, n_arms),
                                                 lower, upper, initial_guess)
                end

                min_index = argmin([r.minimum for r in results[i, :]])
                opt_param = results[i, min_index].minimizer
                opt_params[i] = opt_param

                # case が 1:10 とかみたいなのじゃなくて、 id とかみたいに乱数だったり、 sim で 101, 205, 300-310 とかって
                # 場合も考えるとどうするか、という問題から、
                # results = Dict{Tuple{Int, Int}, Any}() # (case, i_opt) => result
                # という実装を考えた。しかし問題は、 Dict r の要素が (i,j) => V とかって場合に、
                # j に関する min とかをどうとるのか。
                # j_min = minimum([r.minimum for r in results[case, :]])
                # case => matrix のようにするてもある
                # dict[case][i, i_opt] とかね
                # それで、結果を

                # Progress の進捗を安全に更新
                next!(p)
            end
        end
        @ic results
        # メインスレッドで一括してDataFrameに追加
        for opt_param in opt_params
            # @ic opt_param
            # @ic est_params_df
            push!(est_params_df, opt_param)
        end

        estimation_df = hcat(true_params_df, est_params_df)
        println("Estimation complete.")
    end
    return (estimation_df, timed_result)
end

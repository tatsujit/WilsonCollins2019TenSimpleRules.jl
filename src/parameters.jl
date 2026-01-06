using Configurations
using Distributions
using Random
using SHA

const Option{T} = Union{T, Nothing}

@enum EstimatorType begin
    Q
    DQ
    FQ
    DFQ
    CQ
    CDQ
    CFQ
    CDFQ
    ThompsonSampling
    UCB1
    UCB1Tuned
    RS
    Softsatisficing
    EpsilonGreedy
    RSTS
end

@enum PolicyType begin
    Softmax
    EpsilonConst
    EpsilonFirst
    NWSLS
    Bias
end

@enum BanditType begin
    Bernoulli
    Normal
    Uniform
    Mixture
end

@enum EstimationMethod begin
    MLE
    MAP
    NM
    NMC # naive Monte Carlo
    # IS
    MCMC
    Bootstrap
end

@enum MetricType begin
    Regret
    SleepRegret
    Accuracy
    RelativeAccuracy
    Coverage
    Entropy
end

@enum StatisticalTest begin
    TTest
    Wilcoxon
end

@option struct AlgorithmConfig
    estimator::EstimatorType = Q
    estimator_params::NamedTuple = (β=3.0,)
    policy::PolicyType = Softmax
    policy_params::NamedTuple = (β=1.0,)
    
    function AlgorithmConfig(estimator, estimator_params, policy, policy_params)
        validate_estimator_params(estimator, estimator_params)
        validate_policy_params(policy, policy_params)
        new(estimator, estimator_params, policy, policy_params)
    end
end

@option mutable struct TaskConfig
    n_arms::Int = 2
    n_available_arms::Option{Int} = nothing
    n_rounds::Int = 50
    type::Option{BanditType} = nothing
    stationary::Bool = true
    sleeping::Bool = false
    change_points::Option{Vector{Int}} = nothing
    reward_distributions::Vector{<:UnivariateDistribution} = [Bernoulli(0.4), Bernoulli(0.6)]
    reward_means::Option{Vector{Float64}} = nothing
    reward_variances::Option{Vector{Float64}} = nothing
    reward_ranges::Option{Vector{Tuple{Float64, Float64}}} = nothing # assuming that the dist is Uniform
    available_arms::Option{Matrix{Int}} = nothing
    reward_distribution_matrix::Option{Matrix{<:UnivariateDistribution}} = nothing # for non-stationary bandits
    
    function TaskConfig(n_arms, n_available_arms, n_rounds, type, stationary, sleeping,
                       change_points, reward_distributions, reward_means, reward_variances,
                       reward_ranges, available_arms, reward_distribution_matrix)
        n_arms > 1 || throw(ArgumentError("n_arms must be greater than 1"))
        n_rounds > 1 || throw(ArgumentError("n_rounds must be greater than 1"))
        
        if n_available_arms !== nothing
            1 < n_available_arms <= n_arms || 
                throw(ArgumentError("n_available_arms must be between 1 and n_arms"))
        end
        
        if sleeping
            n_available_arms !== nothing && 1 < n_available_arms < n_arms || # これでいいのかな？
                throw(ArgumentError("For sleeping bandits, n_available_arms must be set and between 1 and n_arms"))
        end
        
        if !stationary
            (change_points !== nothing || reward_distribution_matrix !== nothing) ||
                throw(ArgumentError("Non-stationary bandits require either change_points or reward_distribution_matrix"))
        end
        
        if change_points !== nothing
            all(x -> 1 < x < n_rounds, change_points) ||
                throw(ArgumentError("All change_points must be between 1 and n_rounds"))
        end
        
        if sleeping && available_arms !== nothing
            size(available_arms) == (n_available_arms, n_rounds) ||
                throw(ArgumentError("available_arms must have shape (n_available_arms, n_rounds)"))
        end
        
        length(reward_distributions) == n_arms ||
            throw(ArgumentError("reward_distributions must have length equal to n_arms"))
        
        new(n_arms, n_available_arms, n_rounds, type, stationary, sleeping,
            change_points, reward_distributions, reward_means, reward_variances,
            reward_ranges, available_arms, reward_distribution_matrix)
    end
end

@option struct EstimationConfig
    method::EstimationMethod = MLE
    n_samples::Option{Int} = nothing
    estimation_ranges::Option{Vector{Tuple{Float64, Float64}}} = nothing # TODO: これは NMC にも必要では？
    initial_guess_dists::Option{Vector{<:Distribution}} = nothing
    n_opts::Int = 5
    priors::Option{Vector{<:Distribution}} = nothing
    mcmc_samples::Option{Int} = nothing
    mcmc_warmup::Option{Int} = nothing
    n_bootstrap_samples::Option{Int} = nothing
    
    function EstimationConfig(method, n_samples, estimation_ranges, initial_guess_dists,
                             n_opts, priors, mcmc_samples, mcmc_warmup, n_bootstrap_samples)
        n_opts > 0 || throw(ArgumentError("n_opts must be positive"))
        
        if method == NMC
            n_samples !== nothing || throw(ArgumentError("n_samples required for IS method"))
        end
        
        if method ∈ [MLE, MAP]
            estimation_ranges !== nothing || throw(ArgumentError("estimation_ranges required for MLE/MAP methods"))
        end
        
        if method == MAP
            priors !== nothing || throw(ArgumentError("priors required for MAP method"))
        end
        
        if method == MCMC
            mcmc_samples !== nothing || throw(ArgumentError("mcmc_samples required for MCMC method"))
            mcmc_warmup !== nothing || throw(ArgumentError("mcmc_warmup required for MCMC method"))
        end
        
        if method == Bootstrap
            n_bootstrap_samples !== nothing || throw(ArgumentError("n_bootstrap_samples required for Bootstrap method"))
        end
        
        new(method, n_samples, estimation_ranges, initial_guess_dists,
            n_opts, priors, mcmc_samples, mcmc_warmup, n_bootstrap_samples)
    end
end

@option struct EvaluationConfig
    metrics::Vector{MetricType} = [Regret]
    statistical_tests::Option{Vector{StatisticalTest}} = nothing
    
    function EvaluationConfig(metrics, statistical_tests)
        !isempty(metrics) || throw(ArgumentError("At least one metric must be specified"))
        new(metrics, statistical_tests)
    end
end

@option struct ExperimentConfig
    random_seed::Int = 12345
    setting_string::String = ""
    experiment_id::String = ""
    description::String = ""
    tags::Vector{String} = String[]
    task::TaskConfig = TaskConfig()
    algorithms::Vector{AlgorithmConfig} = [AlgorithmConfig()]
    estimation::Option{EstimationConfig} = nothing
    evaluation::Option{EvaluationConfig} = nothing
    n_replications::Option{Int} = nothing
    output_dir::String = "results"
    save_trajectories::Bool = false
    save_intermediate_results::Bool = false
    parallel_execution::Bool = true
    max_threads::Int = Threads.nthreads()
    timeout_minutes::Option{Int} = nothing
    
    function ExperimentConfig(random_seed, setting_string, experiment_id, description, tags,
                             task, algorithms, estimation, evaluation, n_replications,
                             output_dir, save_trajectories, save_intermediate_results,
                             parallel_execution, max_threads, timeout_minutes)
        !isempty(experiment_id) || throw(ArgumentError("experiment_id cannot be empty"))
        !isempty(algorithms) || throw(ArgumentError("At least one algorithm must be specified"))
        
        max_threads > 0 || throw(ArgumentError("max_threads must be positive"))
        max_threads <= Threads.nthreads() || 
            throw(ArgumentError("max_threads cannot exceed available threads"))
        
        if timeout_minutes !== nothing
            timeout_minutes > 0 || throw(ArgumentError("timeout_minutes must be positive"))
        end
        
        if n_replications !== nothing
            n_replications > 0 || throw(ArgumentError("n_replications must be positive"))
        end
        
        new(random_seed, setting_string, experiment_id, description, tags,
            task, algorithms, estimation, evaluation, n_replications,
            output_dir, save_trajectories, save_intermediate_results,
            parallel_execution, max_threads, timeout_minutes)
    end
end

@option struct AnalysisConfig
    random_seed::Int = 123456
    data_path::String = ""
    experiment_name::String = ""
    description::String = ""
    algorithms::Option{Vector{AlgorithmConfig}} = nothing
    estimation::Option{Vector{EstimationConfig}} = nothing
    evaluation::Option{Vector{EvaluationConfig}} = nothing
    
    function AnalysisConfig(random_seed, data_path, experiment_name, description,
                           algorithms, estimation, evaluation)
        !isempty(data_path) || throw(ArgumentError("data_path cannot be empty"))
        isfile(data_path) || throw(ArgumentError("data_path must point to an existing file"))
        
        new(random_seed, data_path, experiment_name, description,
            algorithms, estimation, evaluation)
    end
end

@option struct SimulationConfig
    experiment::ExperimentConfig
    
    function SimulationConfig(experiment)
        experiment.estimation === nothing ||
            throw(ArgumentError("Simulation does not require estimation config"))
        new(experiment)
    end
end

@option struct ParameterRecoveryConfig
    experiment::ExperimentConfig
    
    function ParameterRecoveryConfig(experiment)
        experiment.estimation !== nothing ||
            throw(ArgumentError("Parameter recovery requires estimation config"))
        new(experiment)
    end
end

function validate_estimator_params(estimator::EstimatorType, params::NamedTuple)
    if estimator ∈ [Q, DQ, FQ, DFQ, CQ, CDQ, CFQ, CDFQ]
        if haskey(params, :β)
            params.β > 0 || throw(ArgumentError("β must be positive for Q-learning estimators"))
        end
    elseif estimator == UCB1 || estimator == UCB1Tuned
        if haskey(params, :c)
            params.c > 0 || throw(ArgumentError("c must be positive for UCB"))
        end
    elseif estimator == ThompsonSampling
        if haskey(params, :α)
            params.α > 0 || throw(ArgumentError("α must be positive for Thompson Sampling"))
        end
        if haskey(params, :β)
            params.β > 0 || throw(ArgumentError("β must be positive for Thompson Sampling"))
        end
    end
end

function validate_policy_params(policy::PolicyType, params::NamedTuple)
    if policy == Softmax
        if haskey(params, :β)
            params.β !== nothing || throw(ArgumentError("β required for Softmax policy"))
        end
    elseif policy ∈ [EpsilonConst, EpsilonFirst]
        if haskey(params, :ε)
            0 ≤ params.ε ≤ 1 || throw(ArgumentError("ε must be in [0, 1]"))
        end
        if policy == EpsilonFirst && haskey(params, :exploration_rounds)
            params.exploration_rounds > 0 || 
                throw(ArgumentError("exploration_rounds must be positive"))
        end
    elseif policy == NWSLS
        if haskey(params, :ε)
            0 ≤ params.ε ≤ 1 || throw(ArgumentError("ε must be in [0, 1] for n-WSLS"))
        end
    end
end

function generate_experiment_id(config::ExperimentConfig)
    config_str = string(config)
    bytes2hex(sha256(config_str))
end

function create_reward_distributions(type::BanditType, n_arms::Int; kwargs...)
    if type == Bernoulli
        probs = get(kwargs, :probs, sort(rand(n_arms)))
        return [Bernoulli(p) for p in probs]
    elseif type == Normal
        means = get(kwargs, :means, randn(n_arms))
        variance = get(kwargs, :variance, 1.0)
        return [Normal(μ, sqrt(variance)) for μ in means]
    elseif type == Uniform
        ranges = get(kwargs, :ranges, [(0.0, rand()) for _ in 1:n_arms])
        return [Uniform(a, b) for (a, b) in ranges]
    elseif type == Mixture
        throw(ArgumentError("Mixture type requires custom distribution specification"))
    end
end

export AlgorithmConfig, TaskConfig, EstimationConfig, EvaluationConfig
export ExperimentConfig, AnalysisConfig, SimulationConfig, ParameterRecoveryConfig
export EstimatorType, PolicyType, BanditType, EstimationMethod, MetricType, StatisticalTest
export generate_experiment_id, create_reward_distributions
export Option

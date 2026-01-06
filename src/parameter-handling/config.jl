using YAML, JSON3, Parameters, Distributions, Dates

abstract type AbstractConfig end

"""
Estimator configuration struct containing all 11 parameters
Supports YAML/JSON serialization and type-safe runtime usage
"""
@with_kw struct EstimatorConfig <: AbstractConfig
    # Learning parameters
    Q0::Float64 = 0.0      # Initial Q-value
    α::Float64 = -1.0      # Learning rate (positive RPE)
    αm::Float64 = -1.0     # Learning rate (negative RPE, -1.0 = vanish)
    β::Float64 = 1.0       # Inverse temperature for action values
    
    # Forgetting parameters  
    αf::Float64 = 0.0      # Forgetting rate (0.0 = vanish)
    μ::Float64 = 0.0       # Forgetting default value (0.0 = vanish)
    
    # Stickiness parameters
    τ::Float64 = 0.0       # Stickiness parameter (0.0 = vanish) 
    φ::Float64 = 0.0       # Stickiness inverse temperature (0.0 = vanish)
    C0::Float64 = 0.0      # Initial stickiness value (0.0 = vanish)
    
    # Utility function parameters
    η::Float64 = 0.5       # Beta distribution parameter (expectation)
    ν::Float64 = 2.0       # Beta distribution parameter (precision)
    
    # Utility function settings
    utility_function::String = "identity"  # "identity", "betaCDF", or "custom"
    betaCDF::Bool = true   # Whether to use betaCDF utility function
end

"""
Environment configuration for bandit experiments
Supports stationary, non-stationary, and limited-arm scenarios
"""
@with_kw struct EnvironmentConfig <: AbstractConfig
    # Basic environment settings
    n_arms::Int = 2
    environment_type::String = "stationary"  # "stationary", "non_stationary", "limited_arms"
    
    # Reward distribution settings
    distribution_type::String = "bernoulli"  # "bernoulli", "normal", "uniform"
    reward_params::Vector{Vector{Float64}} = [[0.3], [0.7]]  # Parameters for each arm
    
    # Non-stationary settings
    n_trials::Int = 100
    change_points::Vector{Int} = Int[]  # Trial numbers where distributions change
    
    # Limited arms settings  
    available_arms_schedule::Vector{Vector{Int}} = Vector{Int}[]  # Arms available per trial
    
    # Random seed
    seed::Int = 42
end

"""
Experiment configuration combining estimator and environment settings
"""
@with_kw struct ExperimentConfig <: AbstractConfig
    name::String = "default_experiment"
    description::String = ""
    
    estimator::EstimatorConfig = EstimatorConfig()
    environment::EnvironmentConfig = EnvironmentConfig()
    
    # Simulation settings
    n_simulations::Int = 100
    n_trials_per_simulation::Int = 100
    
    # Output settings
    output_dir::String = "_output"
    save_results::Bool = true
    
    # Meta information
    created_at::String = string(now())
    version::String = "1.0"
end

# Note: These conversion functions require the Estimator and Environment structs
# to be loaded. Include estimator.jl and environment.jl before using these functions.

"""
Convert EstimatorConfig to Estimator struct for runtime use
"""
function to_estimator(config::EstimatorConfig, n_arms::Int)
    # Check if Estimator is defined
    if !@isdefined(Estimator)
        error("Estimator struct not defined. Include estimator.jl first.")
    end
    
    # For betaCDF utility function, we let the Estimator constructor handle it
    if config.utility_function == "betaCDF" || config.betaCDF
        return Estimator(
            n_arms;
            Q0 = config.Q0,
            α = config.α,
            αm = config.αm, 
            β = config.β,
            αf = config.αf,
            μ = config.μ,
            τ = config.τ,
            φ = config.φ,
            C0 = config.C0,
            η = config.η,
            ν = config.ν,
            betaCDF = config.betaCDF
        )
    elseif config.utility_function == "identity"
        return Estimator(
            n_arms;
            Q0 = config.Q0,
            α = config.α,
            αm = config.αm, 
            β = config.β,
            αf = config.αf,
            μ = config.μ,
            τ = config.τ,
            φ = config.φ,
            C0 = config.C0,
            η = config.η,
            ν = config.ν,
            utility_function = (x -> x),
            betaCDF = false
        )
    else
        error("Unsupported utility function: $(config.utility_function)")
    end
end

"""
Convert EnvironmentConfig to Environment struct for runtime use
"""
function to_environment(config::EnvironmentConfig)
    # Check if Environment is defined
    if !@isdefined(Environment)
        error("Environment struct not defined. Include environment.jl first.")
    end
    
    if config.environment_type == "stationary"
        if config.distribution_type == "bernoulli"
            # reward_params should be [[p1], [p2], ...] for Bernoulli
            probs = [params[1] for params in config.reward_params]
            return Environment(probs)
        elseif config.distribution_type == "normal"
            # reward_params should be [[μ1, σ1], [μ2, σ2], ...] for Normal
            dists = [Normal(params[1], params[2]) for params in config.reward_params]
            return Environment(dists)
        elseif config.distribution_type == "uniform"
            # reward_params should be [[a1, b1], [a2, b2], ...] for Uniform
            dists = [Uniform(params[1], params[2]) for params in config.reward_params]
            return Environment(dists)
        else
            error("Unsupported distribution type: $(config.distribution_type)")
        end
    elseif config.environment_type == "non_stationary"
        if config.distribution_type == "bernoulli"
            # Check if NonStationaryEnvironment is defined
            if !@isdefined(NonStationaryEnvironment)
                error("NonStationaryEnvironment struct not defined. Include environment.jl first.")
            end
            
            # reward_params contains initial probabilities
            initial_probs = [params[1] for params in config.reward_params]
            
            # For the specific case of reward reversal at trial 51
            # We'll create phase probabilities by reversing the initial probabilities
            if length(config.change_points) > 0
                # Reverse the probabilities for the second phase
                reversed_probs = reverse(initial_probs)
                phase_probs = [reversed_probs]
                
                return NonStationaryEnvironment(initial_probs, config.change_points, phase_probs)
            else
                error("Non-stationary environment requires change_points to be specified")
            end
        else
            error("Non-stationary environment currently only supports Bernoulli distributions")
        end
    else
        error("Unsupported environment type: $(config.environment_type)")
    end
end

# YAML serialization functions
"""
Save configuration to YAML file with comments
"""
function save_yaml(config::AbstractConfig, filepath::String; add_comments::Bool = true)
    # Convert struct to dictionary for YAML serialization
    config_dict = Dict{String, Any}()
    
    for field in fieldnames(typeof(config))
        value = getfield(config, field)
        config_dict[string(field)] = value
    end
    
    # Add header comment if requested
    yaml_content = ""
    if add_comments
        yaml_content *= "# Configuration file for bandit experiment\n"
        yaml_content *= "# Generated at: $(now())\n"
        yaml_content *= "# \n"
    end
    
    yaml_content *= YAML.write(config_dict)
    
    open(filepath, "w") do io
        write(io, yaml_content)
    end
end

"""
Load configuration from YAML file
"""
function load_yaml(::Type{T}, filepath::String) where {T <: AbstractConfig}
    config_dict = YAML.load_file(filepath)
    
    # Convert dictionary back to struct
    # Create keyword arguments for struct constructor
    kwargs = Dict{Symbol, Any}()
    for (key, value) in config_dict
        kwargs[Symbol(key)] = value
    end
    
    return T(; kwargs...)
end

# JSON serialization functions  
"""
Save configuration to JSON file (lightweight for batch experiments)
"""
function save_json(config::AbstractConfig, filepath::String)
    config_dict = Dict{String, Any}()
    
    for field in fieldnames(typeof(config))
        value = getfield(config, field)
        config_dict[string(field)] = value
    end
    
    open(filepath, "w") do io
        JSON3.pretty(io, config_dict)
    end
end

"""
Load configuration from JSON file
"""
function load_json(::Type{T}, filepath::String) where {T <: AbstractConfig}
    config_dict = JSON3.read(read(filepath, String), Dict{String, Any})
    
    # Convert dictionary back to struct
    kwargs = Dict{Symbol, Any}()
    for (key, value) in config_dict
        kwargs[Symbol(key)] = value
    end
    
    return T(; kwargs...)
end

# Validation functions
"""
Validate EstimatorConfig parameters
"""
function validate_config(config::EstimatorConfig)
    errors = String[]
    
    # Check parameter ranges
    if !(0.0 <= config.α <= 1.0) && config.α != -1.0
        push!(errors, "α must be in [0.0, 1.0] or -1.0 (for sample average)")
    end
    
    if !(0.0 <= config.αm <= 1.0) && config.αm != -1.0
        push!(errors, "αm must be in [0.0, 1.0] or -1.0 (vanish)")
    end
    
    if config.β <= 0.0
        push!(errors, "β must be positive")
    end
    
    if !(0.0 <= config.αf <= 1.0)
        push!(errors, "αf must be in [0.0, 1.0]")
    end
    
    if !(0.0 <= config.η <= 1.0)
        push!(errors, "η must be in [0.0, 1.0]")
    end
    
    if config.ν <= 0.0
        push!(errors, "ν must be positive")
    end
    
    if !isempty(errors)
        error("EstimatorConfig validation failed:\n" * join(errors, "\n"))
    end
    
    return true
end

"""
Validate EnvironmentConfig parameters
"""
function validate_config(config::EnvironmentConfig)
    errors = String[]
    
    if config.n_arms <= 0
        push!(errors, "n_arms must be positive")
    end
    
    if length(config.reward_params) != config.n_arms
        push!(errors, "reward_params length must match n_arms")
    end
    
    if config.distribution_type == "bernoulli"
        for (i, params) in enumerate(config.reward_params)
            if length(params) != 1 || !(0.0 <= params[1] <= 1.0)
                push!(errors, "Bernoulli arm $i: probability must be in [0.0, 1.0]")
            end
        end
    end
    
    if !isempty(errors)
        error("EnvironmentConfig validation failed:\n" * join(errors, "\n"))
    end
    
    return true
end

# Convenience functions
"""
Create default experiment configuration
"""
function default_experiment_config()
    return ExperimentConfig(
        name = "default_bandit_experiment",
        description = "Default two-armed bandit experiment",
        estimator = EstimatorConfig(),
        environment = EnvironmentConfig(
            n_arms = 2,
            reward_params = [[0.3], [0.7]]
        )
    )
end

"""
Generate batch experiment configurations with parameter sweeps
"""
function generate_batch_configs(base_config::ExperimentConfig, 
                               param_sweeps::Dict{Symbol, Vector})
    configs = ExperimentConfig[]
    
    # Get all parameter combinations
    param_names = collect(keys(param_sweeps))
    param_values = collect(values(param_sweeps))
    
    # Generate all combinations using Iterators.product
    for combination in Iterators.product(param_values...)
        config = deepcopy(base_config)
        
        # Update parameters
        for (i, param_name) in enumerate(param_names)
            if hasproperty(config.estimator, param_name)
                setfield!(config.estimator, param_name, combination[i])
            elseif hasproperty(config.environment, param_name)  
                setfield!(config.environment, param_name, combination[i])
            else
                error("Parameter $param_name not found in estimator or environment config")
            end
        end
        
        # Update experiment name
        param_str = join(["$name=$(val)" for (name, val) in zip(param_names, combination)], "_")
        # Manually create new config with updated name
        config = ExperimentConfig(
            name = "$(base_config.name)_$(param_str)",
            description = config.description,
            estimator = config.estimator,
            environment = config.environment,
            n_simulations = config.n_simulations,
            n_trials_per_simulation = config.n_trials_per_simulation,
            output_dir = config.output_dir,
            save_results = config.save_results,
            created_at = config.created_at,
            version = config.version
        )
        
        push!(configs, config)
    end
    
    return configs
end
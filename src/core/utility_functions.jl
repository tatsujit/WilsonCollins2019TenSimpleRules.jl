################################################################
# anything related to utility functions
################################################################

function step_utility_function(x::Float64; origin::Float64=0.0, scale::Float64=1.0)
    return x >= origin ? scale : 0
end

function sigmoid_utility_function(x::Float64; origin::Float64=0.0, gain::Float64=1.0)
    return 1 / (1 + exp(-gain * (x - origin)))
end

function piecewise_linear_sigmoid_utility_function(x::Float64; low::Float64=0.0, high::Float64=1.0, scale::Float64=1.0)
    if x < low
        return 0.0
    elseif low ≤ x < high
        return scale * (x - low) / (high - low)
    else
        return scale
    end
end

"""
- 場合によっては、これについてはスケール (f(high) や f(low) の値) をパラメトライズしたバージョンも作れば良いだろう
- あと切片も (f(low) = 0.0 と限らないように)
"""
function piecewise_linear_utility_function(x::Float64; low::Float64=0.0, high::Float64=1.0,
                                           slope1::Float64=1.0, slope2::Float64=1.0, slope3::Float64=1.0)
    if x < low
        return slope1 * x
    elseif low ≤ x < high
        return slope2 * x
    else
        return slope2 * high + slope3 * (x-high)
    end
end

function discontinuous_linear_utility_function(x::Float64; origin::Float64=0.0,
                                               slope1::Float64=1.0, slope2::Float64=1.0)
    if x < origin
        return slope1 * x
    elseif origin ≤ x
        return slope2 * x
    end
end

# utility_functions = Dict(
utility_functions = ( # order preservation preferable, so not Dict (OrderedDict is ok)
    "step" => step_utility_function,
    "sigmoid" => sigmoid_utility_function,
    "piecewise_linear_sigmoid" => piecewise_linear_sigmoid_utility_function,
    "piecewise_linear" => piecewise_linear_utility_function,
    "discontinuous_linear" => discontinuous_linear_utility_function
)


################################################################
# plotting (three variants)
################################################################

include("misc/utility_functions_plot.jl")

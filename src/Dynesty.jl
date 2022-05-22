module Dynesty

# Write your package code here.
using PyCall
using StatsBase

export NestedSampler,
       DynamicNestedSampler,
       runplot,
       traceplot,
       cornerpoints,
       cornerplot,
       boundplot,
       cornerbound,
       sample, resample_equal

const dynesty = PyNULL()

"""
    `dyplot`
Object that holds the plotting submodule of the `dynesty` package
"""
const dyplot = PyNULL()

"""
    `NestedSampler`
Julia interface to the `dynesty` NestedSampler class. Note that we do
not pass the loglikelihood or prior transform function here. Instead this
is passed to the `sample`` call.

# Example
```julia
# define a distribution
d = MvNormal(ones(10))

loglikelihood(x) = logpdf(d, x)

prior_transform(p) = -10.0 .+ 20.0.*p

smplr = NestedSampler(10)

res = sample(loglikelihood, prior_transform, smplr; dlogz=0.5)
```
"""
Base.@kwdef struct NestedSampler
    ndim::Int
    nlive::Int = 500
    bound::String = "multi"
    sample::String = "auto"
    periodic = nothing
    reflective = nothing
    update_interval::Float64 = 1.5
    first_update = nothing
    gradient = nothing
    walks::Int = 25
    facc::Float64 = 0.5
    slices::Int = 5
    fmove::Float64 = 0.9
    max_move::Int = 100
end

function NestedSampler(ndim::Int; kwargs...)
    return NestedSampler(;ndim=ndim, kwargs...)
end


"""
    `DyamicNestedSampler`
Julia interface to the `dynesty` DynamicNestedSampler class. Note that we do
not pass the loglikelihood or prior transform function here. Instead this
is passed to the `sample`` call.

# Example
```julia
# define a distribution
d = MvNormal(ones(10))

loglikelihood(x) = logpdf(d, x)

prior_transform(p) = -10.0 .+ 20.0.*p

smplr = NestedSampler(10)

# sample using dynamic nested sampling with 500 initial live points
res = sample(loglikelihood, prior_transform, smplr; dlogz=0.5, nlive_init=500)
```
"""
Base.@kwdef struct DynamicNestedSampler
    ndim::Int
    bound::String = "multi"
    sample::String = "auto"
    periodic = nothing
    reflective = nothing
    update_interval::Float64 = 1.5
    first_update = nothing
    gradient = nothing
    walks::Int = 25
    facc::Float64 = 0.5
    slices::Int = 5
    fmove::Float64 = 0.9
    max_move::Int = 100
end

function DynamicNestedSampler(ndim::Int; kwargs...)
    return DynamicNestedSampler(;ndim=ndim, kwargs...)
end



"""
    DynestyOutput
A objects that holds the dynesty output as well as the sampler.
This object can be passed to the plotting functions to produce the usual
dynesty diagnostic plots.

# Notes
We have also implemented a limited number of Julia's `Dictionary` interface so you
can access the output in the usual Dynesty manner

# Example
```julia
# define a distribution
d = MvNormal(ones(10))

loglikelihood(x) = logpdf(d, x)

# We only look at finite region of parameter space
prior_transform(p) = -5.0 .+ 10.0.*p

smplr = NestedSampler(10)

# sample using dynamic nested sampling with 500 initial live points
res = sample(loglikelihood, prior_transform, smplr; dlogz=0.5, nlive_init=500)

# fetch the samples
res[:samples]

# print the keys
keys(res)
```

"""
struct DynestyOutput{D,S}
    dict::D
    sampler::S
end

Base.getindex(d::DynestyOutput, key) = getindex(d.dict, key)
Base.keys(d::DynestyOutput) = keys(d.dict)
Base.get(f::Function, d::DynestyOutput, key) = get(f, d.dict, key)
Base.values(d::DynestyOutput) = values(d.dict)

"""
    `sample(loglikelihood, prior_transform, s::NestedSampler; kwargs...)`

Runs dynesty's NestedSampler algorithm with the specified loglikelihood and prior_transform.
The loglikelihood and prior_transform are functions. For the specific relevant kwargs see the
dynesty documentation at [https://dynesty.readthedocs.io/]
"""
function StatsBase.sample(loglikelihood, prior_transform, s::NestedSampler; kwargs...)
    dysampler = dynesty.NestedSampler(loglikelihood,
                                      prior_transform,
                                      s.ndim,;
                                      nlive = s.nlive,
                                      bound = s.bound,
                                      sample = s.sample,
                                      periodic = s.periodic,
                                      reflective = s.reflective,
                                      update_interval = s.update_interval,
                                      first_update = s.first_update,
                                      gradient = s.gradient,
                                      walks = s.walks,
                                      facc = s.facc,
                                      slices = s.slices,
                                      fmove = s.fmove,
                                      max_move = s.max_move
                                    )
    dysampler.run_nested(;kwargs...)
    return DynestyOutput(dysampler.results, dysampler)
end

"""
    `sample(loglikelihood, prior_transform, s::DynamicNestedSampler; kwargs...)`

Runs dynesty's DynamicNestedSampler algorithm with the specified loglikelihood and prior_transform.
The loglikelihood and prior_transform are functions. For the specific relevant kwargs see the
dynesty documentation at [https://dynesty.readthedocs.io/]
"""
function StatsBase.sample(loglikelihood, prior_transform, s::DynamicNestedSampler; kwargs...)
    dysampler = dynesty.DynamicNestedSampler(loglikelihood,
                                      prior_transform,
                                      s.ndim,;
                                      bound = s.bound,
                                      sample = s.sample,
                                      periodic = s.periodic,
                                      reflective = s.reflective,
                                      update_interval = s.update_interval,
                                      first_update = s.first_update,
                                      gradient = s.gradient,
                                      walks = s.walks,
                                      facc = s.facc,
                                      slices = s.slices,
                                      fmove = s.fmove,
                                      max_move = s.max_move
                                    )
    dysampler.run_nested(;kwargs...)
    return DynestyOutput(dysampler.results, dysampler)
end

"""
    resample_equal(res::DynestyOutput, nsamples::Int)
Resample the `dynesty` nested sampling run so that the samples have equal weighting.
This uses the `StatsBase` algorithm under the hood.

The results are a vector of vectors where the inner vector corresponds to the samples.
"""
function resample_equal(res::DynestyOutput, nsamples::Int)
    samples, weights = res["samples"], exp.(res["logwt"] .- res["logz"][end])
    sample(collect(eachrow(samples)), Weights(weights), nsamples)
end

"""
    merge(args::DynestyOutput...; print_progres=true)
Runs dynesty's merge_runs to combine multiple separate dynesty runs.
"""
function Base.merge(args::DynestyOutput...; print_progress=true)
    runs = getproperty.([args...], :dict)
    return DynestyOutput(py"merge"(runs; print_progress), nothing)
end

# internal ess estimator
function ess(res::DynestyOutput)
    weights = exp.(res["logwt"] .- res["logz"][end])
    return inv(sum(abs2, weights))
end


# Make the dynesty plots with a bit of fun metaprogramming
PLOTS = [:runplot, :traceplot, :cornerpoints, :cornerplot, :boundplot, :cornerbound]
for P in PLOTS
    @eval begin
        """
            $($P)(d::DynestyOutput; kwargs...)
        Produces the $($P) plot from the dynesty.plotting module.
        For a list of possible kwargs see the dynesty documentation at
        [dynesty.readthedocs.io]
        """
        function ($P)(d::DynestyOutput, args...; kwargs...)
            f = getproperty(dyplot, Symbol($P))
            f(d.dict, args...; kwargs...)
        end
    end
end







function __init__()
    copy!(dynesty, pyimport_conda("dynesty","dynesty"))
    copy!(dyplot, pyimport("dynesty.plotting"))
    # Define a hack to get merge to work without doing silly dict conversion
    py"""
    import dynesty
    def merge(x, print_progress=True):
        xres = [dynesty.results.Results(r) for r in x]
        mruns = dynesty.utils.merge_runs(xres, print_progress)
        return mruns
    """
end

end

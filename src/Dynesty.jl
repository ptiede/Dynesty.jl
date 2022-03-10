module Dynesty

# Write your package code here.
using PyCall
using StatsBase

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
Base.@kwdef struct NestedSampler{P,R,D,G}
    ndim::Int
    nlive::Int = 500
    bound::String = "multi"
    sample::String = "auto"
    periodic::P = nothing
    reflective::R = nothing
    update_interval::Float64
    first_update::D = nothing
    gradient::G = nothing
    rwalks::Int = 25
    facc::Float64 = 0.5
    slices::Int = 5
    fmove::Float64 = 0.9
    max_move::Int = 100
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
Base.@kwdef struct DynamicNestedSampler{P,R,D,G}
    ndim::Int
    bound::String = "multi"
    sample::String = "auto"
    periodic::P = nothing
    reflective::R = nothing
    update_interval::Float64
    first_update::D = nothing
    gradient::G = nothing
    rwalks::Int = 25
    facc::Float64 = 0.5
    slices::Int = 5
    fmove::Float64 = 0.9
    max_move::Int = 100
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

function StatsBase.sample(loglikelihood, prior_transform, s::NestedSampler, kwargs...)
    dysampler = dynesty.NestedSampler(loglikelihood,
                                      prior_transform,
                                      s.ndim,;
                                      nlive = s.nlive,
                                      bound = s.bound,
                                      sample = s.sample,
                                      s.periodic = s.periodic,
                                      s.reflective = s.reflective,
                                      update_interval = s.update_interval,
                                      first_update = s.first_update,
                                      gradient = s.gradient,
                                      rwalks = s.rwalks,
                                      facc = s.facc,
                                      slices = s.slices,
                                      fmove = s.fmove,
                                      max_move = s.max_move
                                    )
    dysampler.run_nested(;kwargs...)
    return DynestyOutput(dysampler.results, dysampler)
end

function StatsBase.sample(loglikelihood, prior_transform, s::DynamicNestedSampler, kwargs...)
    dysampler = dynesty.NestedSampler(loglikelihood,
                                      prior_transform,
                                      s.ndim,;
                                      bound = s.bound,
                                      sample = s.sample,
                                      s.periodic = s.periodic,
                                      s.reflective = s.reflective,
                                      update_interval = s.update_interval,
                                      first_update = s.first_update,
                                      gradient = s.gradient,
                                      rwalks = s.rwalks,
                                      facc = s.facc,
                                      slices = s.slices,
                                      fmove = s.fmove,
                                      max_move = s.max_move
                                    )
    dysampler.run_nested(;kwargs...)
    return DynestyOutput(dysampler.results, dysampler)
end

function resample_equal(d::DynestyOutput, nsamples)
    res = d.dict
    samples, weights = res["samples"], exp.(res["logwt"] .- res["logz"][end])
    sample(samples, Weights(weights), nsamples)
end

function Base.merge(args::DynestyOutput..., print_progress=true)
    runs = collect(args)
    return dynesty.utils.merge_runs(runs; print_progress)
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
        function ($P)(d::DynestyOutput; kwargs...)
            dyplot.($P)(d.dict; kwargs...)
        end
    end
end







function __init__()
    copy!(dynesty, pyimport_conda("dynesty"))
    copy!(dyplot, pyimport("dynesty.plotting"))
end

end

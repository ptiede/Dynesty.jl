# Dynesty
<!---
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ptiede.github.io/Dynesty.jl/dev)
-->

[![Build Status](https://github.com/ptiede/Dynesty.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ptiede/Dynesty.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ptiede/Dynesty.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ptiede/Dynesty.jl)

A Julia interface to the python nested sampling library [dynesty](https://github.com/joshspeagle/dynesty)

This is built on PyCall and imports a lot of the functionality of dynesty. There are some differences in the interface to make the code more "Julian".

# Example

Here we will sample a 5 dimensional Gaussian restricted to the domain [-5,5]<sup>5</sup>

```julia
using Distributions
using Dynesty
# define a distribution
ndim = 5
d = MvNormal(ones(ndim))

loglikelihood(x) = logpdf(d, x)

prior_transform(p) = -10.0 .+ 20.0.*p

smplr = NestedSampler(ndim)

res = sample(loglikelihood, prior_transform, smplr; dlogz=0.5)

# plot the results
cornerplot(res)
```

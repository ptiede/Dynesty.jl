using Dynesty
using Test
using Distributions

@testset "Dynesty.jl" begin
    # Write your tests here.
    # define a distribution
    d = MvNormal(ones(10))

    loglikelihood(x) = logpdf(d, x)

    prior_transform(p) = -5.0 .+ 10.0.*p

    smplr = NestedSampler(10; nlive=2000)
    dysmplr = DynamicNestedSampler(10)
    # sample using dynamic nested sampling with 500 initial live points
    res1 = sample(loglikelihood, prior_transform, smplr; dlogz=0.5)
    res2 = sample(loglikelihood, prior_transform, smplr; dlogz=0.5)
    res3 = sample(loglikelihood, prior_transform, smplr; dlogz=0.5)

    mres = Dynesty.merge(res1, res2, res3)

    res = sample(loglikelihood, prior_transform, dysmplr; nlive_init=200)

    runplot(res1)
    cornerplot(res1)
    traceplot(res1)
    boundplot(res1, (1,2),idx=10)
    cornerbound(res1, it=100)
    cornerpoints(res1)

    esamples = resample_equal(res1, Int(ceil(2*Dynesty.ess(res1))))

    @test isapprox(mean(getindex.(esamples, 1)), mean(d)[1], atol=1e-2)

end

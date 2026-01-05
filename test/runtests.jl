using DrWatson, Test
@quickactivate "WilsonCollins2019TenSimpleRules.jl"

# Here you include files using `srcdir`
# include(srcdir("file.jl"))

# Run test suite
println("Starting tests")
ti = time()

@testset "WilsonCollins2019TenSimpleRules.jl tests" begin
    @test 1 == 1
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")

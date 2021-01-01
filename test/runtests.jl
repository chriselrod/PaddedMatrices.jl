using PaddedMatrices, LinearAlgebra, Aqua
using Test

@show PaddedMatrices.VectorizationBase.REGISTER_COUNT
const START_TIME = time()

@inferred PaddedMatrices.matmul_params(Float32)
@inferred PaddedMatrices.matmul_params(Float64)
@inferred PaddedMatrices.matmul_params(Int16)
@inferred PaddedMatrices.matmul_params(Int32)
@inferred PaddedMatrices.matmul_params(Int64)

@time @testset "PaddedMatrices.jl" begin
    Aqua.test_all(PaddedMatrices)
    # @test isempty(detect_unbound_args(PaddedMatrices))
    @time include("misc.jl")
    @time include("matmul_tests.jl")
    @time include("broadcast_tests.jl")
end

const ELAPSED_MINUTES = (time() - START_TIME)/60
@test ELAPSED_MINUTES < 120

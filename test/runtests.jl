using PaddedMatrices, LinearAlgebra
using Test

@show PaddedMatrices.VectorizationBase.REGISTER_COUNT
const START_TIME = time()

@inferred PaddedMatrices.matmul_params_static(Float32)
@inferred PaddedMatrices.matmul_params_static(Float64)
@inferred PaddedMatrices.matmul_params_static(Int16)
@inferred PaddedMatrices.matmul_params_static(Int32)
@inferred PaddedMatrices.matmul_params_static(Int64)

@time @testset "PaddedMatrices.jl" begin
    @test isempty(detect_unbound_args(PaddedMatrices))
    @time include("misc.jl")
    @time include("matmul_tests.jl")
    @time include("broadcast_tests.jl")
end

const ELAPSED_MINUTES = (time() - START_TIME)/60
@test ELAPSED_MINUTES < 120

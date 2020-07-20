using PaddedMatrices, LinearAlgebra
using Test

@inferred PaddedMatrices.matmul_params_val(Float32)
@inferred PaddedMatrices.matmul_params_val(Float64)
@inferred PaddedMatrices.matmul_params_val(Int16)
@inferred PaddedMatrices.matmul_params_val(Int32)
@inferred PaddedMatrices.matmul_params_val(Int64)

@testset "PaddedMatrices.jl" begin
    @test isempty(detect_unbound_args(PaddedMatrices))
    @time include("matmul_tests.jl")
    @time include("broadcast_tests.jl")
    @time include("misc.jl")
end

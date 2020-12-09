using PaddedMatrices, LinearAlgebra
using Test

@inferred PaddedMatrices.matmul_params_static(Float32)
@inferred PaddedMatrices.matmul_params_static(Float64)
@inferred PaddedMatrices.matmul_params_static(Int16)
@inferred PaddedMatrices.matmul_params_static(Int32)
@inferred PaddedMatrices.matmul_params_static(Int64)

@testset "PaddedMatrices.jl" begin
    @test isempty(detect_unbound_args(PaddedMatrices))
    @time include("misc.jl")
    @time include("matmul_tests.jl")
    @time include("broadcast_tests.jl")
end

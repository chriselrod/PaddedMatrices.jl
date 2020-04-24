using PaddedMatrices, LinearAlgebra
using Test

@testset "PaddedMatrices.jl" begin
    @time include("broadcast_tests.jl")
    @time include("matmul_tests.jl")
end

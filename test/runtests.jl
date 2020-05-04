using PaddedMatrices, LinearAlgebra
using Test

@testset "PaddedMatrices.jl" begin
    @time include("matmul_tests.jl")
    @time include("broadcast_tests.jl")
end

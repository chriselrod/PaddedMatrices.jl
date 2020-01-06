exit()

using PaddedMatrices
using Test

@testset "PaddedMatrices.jl" begin

    @testset "Broadcast" begin
        M, K, N = 47, 85, 74
        T = Float64
        for T ∈ (Float32, Float64)
            A = @Mutable rand($T, M, K);
            B = @Mutable rand($T, K, N);
            x = @Mutable rand($T, M);
            y = @Mutable rand($T, N);

            Aa = Array(A); Ba = Array(B); xa = Array(x); ya = Array(y);
            
            C = (A ∗ B) .* x .+ y' .- 43;
            Ca = (Aa * Ba) .* xa .+ ya' .- 43;
            @test C ≈ Ca
            
            
        end
    end
end

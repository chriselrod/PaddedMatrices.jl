using PaddedMatrices

@testset "Broadcast" begin
    M, K, N = 47, 85, 74
    # for T ∈ (Float32, Float64)
    A = @FixedSize randn(13,29);
    b = @FixedSize rand(13);
    c = @FixedSize rand(29);
    D = @. exp(A) + b * log(c');

    Aa = Array(A); ba = Array(b); ca = Array(c);
    Da = @. exp(Aa) + ba * log(ca');

    @test D ≈ Da
    # end
end




@testset "Miscellaneous" begin
    x = @StrideArray rand(127);
    @test maximum(abs, x) == maximum(abs, Array(x))
    y = @StrideArray rand(3);
    @test maximum(abs, y) == maximum(abs, Array(y))
end


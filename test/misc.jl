

@testset "Miscellaneous" begin
    x = @FixedSize rand(127);
    @test maximum(abs, x) == maximum(abs, Array(x))
    y = @FixedSize rand(3);
    @test maximum(abs, y) == maximum(abs, Array(y))
end


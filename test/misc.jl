
@noinline function dostuff(A; B = I)
    C = A * B
    s = zero(eltype(C))
    PaddedMatrices.@avx for i in eachindex(C)
        s += C[i]
    end
    s
end
function gc_preserve_test()
    A = @StrideArray rand(8,8);
    B = @StrideArray rand(8,8);
    @gc_preserve dostuff(A, B = B)
end

@testset "Miscellaneous" begin
    x = @StrideArray rand(127);
    @test maximum(abs, x) == maximum(abs, Array(x))
    y = @StrideArray rand(3);
    @test maximum(abs, y) == maximum(abs, Array(y))
    @test iszero(@allocated gc_preserve_test())
end


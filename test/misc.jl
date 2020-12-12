
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

    A = @StrideArray rand(10,8)
    A_u = view(A, StaticInt(1):StaticInt(6), :)
    A_l = view(A, StaticInt(7):StaticInt(10), :)
    @test A == @inferred(vcat(A_u, A_l))
    @test PaddedMatrices.size(A) === PaddedMatrices.size(vcat(A_u, A_l))

    @test A[1,1] == A[1]

    A_u = view(A, 1:StaticInt(6), :)
    A_l = view(A, StaticInt(7):StaticInt(10), :)
    @test A == @inferred(vcat(A_u, A_l))
    @test PaddedMatrices.size(A) == PaddedMatrices.size(vcat(A_u, A_l))
    @test PaddedMatrices.size(A) !== PaddedMatrices.size(vcat(A_u, A_l))

    Aa = Array(A)
    @test sum(A) ≈ sum(Aa)
    @test maximum(A) == maximum(Aa) == maximum(abs, @. -A)
    @test mapreduce(abs2, +, A) ≈ mapreduce(abs2, +, Aa)
end


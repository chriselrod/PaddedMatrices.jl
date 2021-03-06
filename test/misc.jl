
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

    A = @StrideArray rand(10,8);
    A_u = view(A, StaticInt(1):StaticInt(6), :)
    A_l = view(A, StaticInt(7):StaticInt(10), :)
    @test A == @inferred(vcat(A_u, A_l))
    @test PaddedMatrices.size(A) === PaddedMatrices.size(vcat(A_u, A_l))

    # On 1.5 tests fail if you don't do this first.
    @test pointer(view(A, 1:StaticInt(6), :)) == pointer(A)
    A_u = view(A, 1:StaticInt(6), :);
    A_l = view(A, StaticInt(7):StaticInt(10), :);
    @test A == @inferred(vcat(A_u, A_l))
    @test PaddedMatrices.size(A) == PaddedMatrices.size(vcat(A_u, A_l))
    @test PaddedMatrices.size(A) !== PaddedMatrices.size(vcat(A_u, A_l))

    Aa = Array(A)
    @test sum(A) ≈ sum(Aa)
    @test maximum(A) == maximum(Aa) == maximum(abs, @. -A)
    @test mapreduce(abs2, +, A) ≈ mapreduce(abs2, +, Aa)

    @test A[1,1] == A[1] == vec(A)[1]
    A[1,1] = 3
    @test A[1,1] == A[1] == vec(A)[1] == 3
    A[2] = 4
    @test A[2,1] == A[2] == vec(A)[2] == 4
    vec(A)[3] = 5
    @test A[3,1] == A[3] == vec(A)[3] == 5

    @test PaddedMatrices.stride_rank(PaddedMatrices.similar_layout(A')) === PaddedMatrices.stride_rank(PaddedMatrices.similar_layout(Aa')) === PaddedMatrices.StrideRank{(2,1)}()
    @test PaddedMatrices.ArrayInterface.contiguous_axis(PaddedMatrices.similar_layout(A')) === PaddedMatrices.ArrayInterface.contiguous_axis(PaddedMatrices.similar_layout(Aa')) === PaddedMatrices.ArrayInterface.Contiguous{2}()
    
    
end


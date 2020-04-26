
@testset "MatMul" begin
    @testset "FixedSize" begin
        for s in 2:32
            A = @FixedSize rand(s,s);
            B = @FixedSize rand(s,s);
            Aa = Array(A); Ba = Array(B);
            @test Aa * Ba ≈ A * B
        end
    end
    @testset "jmul" begin
        for T in (Float32, Float64, Int32, Int64)
            @show T, @__LINE__
            @time for i in 2:20#(T <: Integer ? 20 : 100)
                # s = round(Int, i^1.6505149978319904)
                s = i^3
            # @time for s in 2:200
                A = rand(T, s, s);
                B = rand(T, s, s);
                C1 = similar(A); C2 = similar(C1);
                blastime = @elapsed mul!(C1, A, B)
                pmtime = @elapsed PaddedMatrices.jmul!(C2, A, B)
                @show s, pmtime / blastime
                @test C1 ≈ C2
            end
        end
    end
end



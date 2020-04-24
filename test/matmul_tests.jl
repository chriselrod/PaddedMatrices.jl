
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
            @time for i in 2:(T <: Integer ? 20 : 100)
                s = round(Int, i^1.6505149978319904)
                A = rand(T, s, s);
                B = rand(T, s, s);
                C1 = similar(A); C2 = similar(C1);
                @test mul!(C1, A, B) ≈ PaddedMatrices.jmul!(C2, A, B)
            end
        end
    end
end



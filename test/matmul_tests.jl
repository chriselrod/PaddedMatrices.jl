
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
            @time for i in 2:(T <: Integer ? 10 : 15)
                # s = round(Int, i^1.6505149978319904)
                s = i^3
                # @time for s in 2:200
                gopc = 2e-9*s^3
                A = rand(T, s, s);
                B = rand(T, s, s);
                C1 = similar(A); C2 = similar(C1);
                blastime = @elapsed mul!(C1, A, B)
                bops = gopc / blastime
                @show s, blastime, bops
                pmtime = @elapsed PaddedMatrices.jmul!(C2, A, B)
                pmops = gopc / pmtime
                @show s, pmtime, pmops
                @test C1 ≈ C2
            end
        end
    end
end



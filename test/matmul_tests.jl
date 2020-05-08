
@testset "MatMul" begin
    @testset "jmul" begin
        for T in (Float32, Float64, Int32, Int64)
            @show T, @__LINE__
            @time for i in 2:(T <: Integer ? 10 : 15)
                # s = round(Int, i^1.6505149978319904)
                s = i^3
                # @time for s in 2:200
                gopc = 2e-9*s^3
                A = rand(T, s, s); At = copy(A');
                B = rand(T, s, s); Bt = copy(B');
                C1 = similar(A); C2 = similar(C1); C3 = similar(C1); C4 = similar(C1); C5 = similar(C1);
                blastime = @elapsed mul!(C1, A, B)
                bops = gopc / blastime
                @show s, blastime, bops
                pmtime_nn = @elapsed PaddedMatrices.jmul!(C2, A, B); pmops_nn = gopc / pmtime_nn
                pmtime_nt = @elapsed PaddedMatrices.jmul!(C3, A, Bt'); pmops_nt = gopc / pmtime_nt
                @show s, pmtime_nn, pmops_nn, pmtime_nt, pmops_nt
                pmtime_tn = @elapsed PaddedMatrices.jmul!(C4, At', B); pmops_tn = gopc / pmtime_tn
                pmtime_tt = @elapsed PaddedMatrices.jmul!(C5, At', Bt'); pmops_tt = gopc / pmtime_tt
                @show s, pmtime_tn, pmops_tn, pmtime_tt, pmops_tt
                @test C1 ≈ C2 ≈ C3 ≈ C4 ≈ C5
            end
        end
    end
    @testset "FixedSize" begin
        for s in 2:32
            A = @FixedSize rand(s,s);
            B = @FixedSize rand(s,s);
            Aa = Array(A); Ba = Array(B);
            @test Aa * Ba ≈ A * B
        end
    end
end



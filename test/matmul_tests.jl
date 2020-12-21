
const TRAVIS_SKIP = VERSION.minor != 4 && !isnothing(get(ENV, "TRAVIS_BRANCH", nothing))

function test_fixed_size(M, K, N)
    A = @StrideArray rand(M,K);
    B = @StrideArray rand(K,N);
    At = (@StrideArray rand(K,M))';
    Bt = (@StrideArray rand(N,K))';
    Aa = Array(A); Ba = Array(B);
    Aat = Array(At); Bat = Array(Bt);
    time = @elapsed(@test Aa * Ba ≈ A * B)
    @show M, K, N, time
    if !TRAVIS_SKIP || (isodd(M) & isodd(N))
        time = @elapsed(@test Aa * Bat ≈ A * Bt)
        @show M, K, N, time
        time = @elapsed(@test Aat * Ba ≈ At * B)
        @show M, K, N, time
        time = @elapsed(@test Aat * Bat ≈ At * Bt)
        @show M, K, N, time
    end
    nothing
end

function test_fixed_size(M)
    K = N = M
    A = @StrideArray rand(M,K);
    B = @StrideArray rand(K,N);
    Aa = Array(A); Ba = Array(B);
    time = @elapsed(@test Aa * Ba ≈ A * B)
    @show M, K, N, time
    if !TRAVIS_SKIP || isodd(M)
        time = @elapsed(@test Aa * Ba' ≈ A * B')
        @show M, K, N, time
        time = @elapsed(@test Aa' * Ba ≈ A' * B)
        @show M, K, N, time
        time = @elapsed(@test Aa' * Ba' ≈ A' * B')
        @show M, K, N, time
    end
    nothing
end

# gmul(A, B) = LinearAlgebra.generic_matmatmul('N','N', A, B)

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
                C1 = Matrix{T}(undef, s, s);
                blastime = @elapsed mul!(C1, A, B)
                # blastime = @elapsed (C1 = gmul(A, B))
                C2 = similar(C1); C3 = similar(C1); C4 = similar(C1); C5 = similar(C1);
                bops = gopc / blastime
                @show s, blastime, bops
                pmtime_nn = @elapsed PaddedMatrices.jmul!(C2, A, B); pmops_nn = gopc / pmtime_nn
                @test C1 ≈ C2
                pmtime_nt = @elapsed PaddedMatrices.jmul!(C3, A, Bt'); pmops_nt = gopc / pmtime_nt
                @test C1 ≈ C3
                @show s, pmtime_nn, pmops_nn, pmtime_nt, pmops_nt
                pmtime_tn = @elapsed PaddedMatrices.jmul!(C4, At', B); pmops_tn = gopc / pmtime_tn
                @test C1 ≈ C4
                pmtime_tt = @elapsed PaddedMatrices.jmul!(C5, At', Bt'); pmops_tt = gopc / pmtime_tt
                @test C1 ≈ C5
                @show s, pmtime_tn, pmops_tn, pmtime_tt, pmops_tt
            end
        end
    end
    @testset "FixedSize" begin
        r = 2:7
        for M ∈ r, K ∈ r, N ∈ r
            test_fixed_size(M, K, N)
        end
        r = 8:33
        for M ∈ r
            test_fixed_size(M)
        end
        M = K = N = 80
        A = @StrideArray rand(M,K);
        B = @StrideArray rand(K,N);
        C = StrideArray{Float64}(undef, (StaticInt(M),StaticInt(N)));
        M, K, N = 23, 37, 19
        @views begin
            Av = A[1:M, 1:K]; 
            Bv = B[1:K, 1:N]; 
            Cv = C[1:M, 1:N]; 
            Avsl = A[StaticInt(1):M, StaticInt(1):K]; 
            Bvsl = B[StaticInt(1):K, StaticInt(1):N]; 
            Cvsl = C[StaticInt(1):M, StaticInt(1):N]; 
            Avsr = A[StaticInt(1):StaticInt(M), StaticInt(1):StaticInt(K)]; 
            Bvsr = B[StaticInt(1):StaticInt(K), StaticInt(1):StaticInt(N)]; 
            Cvsr = C[StaticInt(1):StaticInt(M), StaticInt(1):StaticInt(N)]; 
        end
        Creference = Array(Av) * Array(Bvsl);
        time = @elapsed mul!(Cv, Av, Bv)
        @test Creference ≈ Cv
        @show M, K, N, time
        time = @elapsed mul!(Cvsl, Avsl, Bvsl)
        @test Creference ≈ Cv
        @show M, K, N, time
        time = @elapsed mul!(Cvsr, Avsr, Bvsr)
        @test Creference ≈ Cv
        @show M, K, N, time

        Av = A[1:M, 1:K]; 
        Bv = B[1:K, 1:N]; 
        Cv = C[1:M, 1:N]; 
        Avsl = A[StaticInt(1):M, StaticInt(1):K]; 
        Bvsl = B[StaticInt(1):K, StaticInt(1):N]; 
        Cvsl = C[StaticInt(1):M, StaticInt(1):N]; 
        Avsr = A[StaticInt(1):StaticInt(M), StaticInt(1):StaticInt(K)]; 
        Bvsr = B[StaticInt(1):StaticInt(K), StaticInt(1):StaticInt(N)]; 
        Cvsr = C[StaticInt(1):StaticInt(M), StaticInt(1):StaticInt(N)]; 
        Creference = Array(Av) * Array(Bvsl);
        time = @elapsed mul!(Cv, Av, Bv)
        @test Creference ≈ Cv
        @show M, K, N, time
        time = @elapsed mul!(Cvsl, Avsl, Bvsl)
        @test Creference ≈ Cv
        @show M, K, N, time
        time = @elapsed mul!(Cvsr, Avsr, Bvsr)
        @test Creference ≈ Cv
        @show M, K, N, time

    end
end



using PaddedMatrices
using Test


using LinearAlgebra, VectorizationBase
using PaddedMatrices: jmul!, loopmuladd!, matmul_sizes

M, K, N = 240, 512, 320;
A = rand(M, K); B = rand(K, N); C1 = rand(M, N);

ptrA = PtrMatrix{M, K, Float64, M}(pointer(A));
ptrB = PtrMatrix{K, N, Float64, K}(pointer(B));
ptrC = PtrMatrix{M, N, Float64, M}(pointer(C1));

fill!(C1, 0.0); loopmuladd!(ptrC, ptrA, ptrB)
C1 ≈ A * B

# M unknown test
ptrA_mrem = PtrMatrix{-1, K, Float64, M}(pointer(A), M);
ptrC_mrem = PtrMatrix{-1, N, Float64, M}(pointer(C1), M);

using BenchmarkTools
@benchmark loopmuladd!($ptrC_mrem, $ptrA_mrem, $ptrB)

fill!(C1, 0.0); loopmuladd!(ptrC_mrem, ptrA_mrem, ptrB)
C1 ≈ A * B
VectorizationBase.maybestaticsize(ptrA_mrem, Val(1))
VectorizationBase.maybestaticsize(ptrC_mrem, Val(1))

using LoopVectorization: @avx, @_avx, @avx_debug
@macroexpand @avx for I ∈ CartesianIndices(B)
        B[I] = A[I]
    end

using LoopVectorization: @avx_debug
M2, K2, N2 = matmul_sizes(ptrC_mrem, ptrA_mrem, ptrB)
ls = @avx_debug for m ∈ 1:M2
        for n ∈ 1:N2
            Cₘₙ = zero(T)
            for k ∈ 1:K2
                Cₘₙ += ptrA_mrem[m,k] * ptrB[k,n]
            end
            ptrC_mrem[m,n] += Cₘₙ
        end
    end;
ls.loops


using LoopVectorization: @_avx
function loopmuladd_!(
    C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix
)
    M, K, N = matmul_sizes(C, A, B)
    @_avx for m ∈ 1:M
        for n ∈ 1:N
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:K
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] += Cₘₙ
        end
    end
    nothing
end



using PaddedMatrices
using SIMDPirates: lifetime_start!, lifetime_end!
using SIMDPirates: alloca, gep

function regtest!(F::FixedSizeMatrix{M,N}, A::FixedSizeMatrix{M,K}, B::FixedSizeMatrix{K,K}, C::FixedSizeMatrix{K,N}) where {M,N,K}
    D = FixedSizeMatrix{M,K,Float64}(undef)
    E = FixedSizeMatrix{K,N,Float64}(undef)
    mul!(D, A, B)
    mul!(E, B, C)
    mul!(F, D, E)    
end
function allocatest!(F::FixedSizeMatrix{M,N}, A::FixedSizeMatrix{M,K}, B::FixedSizeMatrix{K,K}, C::FixedSizeMatrix{K,N}) where {M,N,K}
    # D = FixedSizeMatrix{M,K,Float64}(undef)
    # E = FixedSizeMatrix{N,L,Float64}(undef)
    Dlen = stride(A,2)*K
    Elen = stride(B,2)*N
    aptr = alloca(Dlen + Elen)
    D = PtrMatrix{M,K,Float64}(aptr)
    # lifetime_start!(D)
    E = PtrMatrix{K,N,Float64}(gep(aptr,Dlen))
    # lifetime_start!(E)
    mul!(D, A, B)
    mul!(E, B, C)
    mul!(F, D, E)
    # lifetime_end!(D)
    # lifetime_end!(E)
end
M, N, K =110,120,130;
F1 = FixedSizeMatrix{M,N,Float64}(undef);
F2 = FixedSizeMatrix{M,N,Float64}(undef);
A = @Mutable rand(M,K);
B = @Mutable rand(K,K);
C = @Mutable rand(K,N);

regtest!(F1, A, B, C);
allocatest!(F2, A, B, C);

@test F1 ≈ F2

@testset "PaddedMatrices.jl" begin

    @testset "Broadcast" begin
        M, K, N = 47, 85, 74
        T = Float64
        for T ∈ (Float32, Float64)
            A = @Mutable rand($T, M, K);
            B = @Mutable rand($T, K, N);
            x = @Mutable rand($T, M);
            y = @Mutable rand($T, N);

            Aa = Array(A); Ba = Array(B); xa = Array(x); ya = Array(y);
            
            C = @. (A *ˡ B) * x + y' - 43;
            Ca = @. $(Aa * Ba) * xa + ya' - 43;
            @test C ≈ Ca
            
        end
    end
end

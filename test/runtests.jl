exit()

using PaddedMatrices
using Test

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

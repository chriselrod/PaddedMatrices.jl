


@generated function LinearAlgebra.dot(
            A::AbstractFixedSizePaddedArray{S,NTuple{W,Core.VecElement{T}},N,P,L},
            B::AbstractFixedSizePaddedArray{S,NTuple{W,Core.VecElement{T}},N,P,L}
        ) where {S,W,T,N,P,L}


    if N > 1
        ST = ntuple(n -> S.parameters[n], Val(N))
        return quote
            out = SIMDPirates.vbroadcast(NTuple{W,Core.VecElement{T}}, zero(T))
            ind = 0
            @nloops $(N-1) i j -> 1:$ST[j+1] begin
                for i_0 ∈ 1:$(ST[1])
                    out = SIMDPirates.vmuladd(A[ind + i_0], B[ind + i_0], out)
                end
                ind += P
            end
            out
        end
    else #if N == 1
        return quote
            out = SIMDPirates.vbroadcast(NTuple{W,Core.VecElement{T}}, zero(T))
            for i ∈ 1:$(S.parameters[1])
                out = SIMDPirates.vmuladd(A[i], B[i], out)
            end
            out
        end
    end
end

@generated function dot_self(
            A::AbstractFixedSizePaddedArray{S,NTuple{W,Core.VecElement{T}},N,P,L}
        ) where {S,W,T,N,P,L}


    if N > 1
        ST = ntuple(n -> S.parameters[n], Val(N))
        return quote
            out = SIMDPirates.vbroadcast(NTuple{W,Core.VecElement{T}}, zero(T))
            ind = 0
            @nloops $(N-1) i j -> 1:$ST[j+1] begin
                for i_0 ∈ 1:$(ST[1])
                    out = SIMDPirates.vmuladd(A[ind + i_0], A[ind + i_0], out)
                end
                ind += P
            end
            out
        end
    else #if N == 1
        return quote
            out = SIMDPirates.vbroadcast(NTuple{W,Core.VecElement{T}}, zero(T))
            for i ∈ 1:$(S.parameters[1])
                out = SIMDPirates.vmuladd(A[i], A[i], out)
            end
            out
        end
    end
end


function determine_pattern(M,N,P)
    # 4 x 5 preferred for avx512
    # 32 registers means in C = A * B,
    # we have 4*5 = 20 filled from A,
    # filling one column of C at a time (4) using one column from B, for 29 total.
    #
    # 3 x 3 for avx2
    # We have 3*3 = 9 from A, filling 1 column (3) from C, using one column from B (3), for 15 total.

    if VectorizationBase.REGISTER_COUNT == 32
        km = 4
        kn = 6
    else
        km = 3
        kn = 3
    end
    md, mr = divrem(M, km)
    nd, nr = divrem(N, kn)
    mk, mr, nd, nr, km, kn
end



function matrix_of_vecs_mul_quote(M,N,P,W,T)
    mk, mr, nk, nr, kernel_size_m, kernel_size_n = determine_pattern(M,N)
    # if mr == 0
    #     @assert mk > 0
    #     mk -= 1
    #     mr += kernel_size_m
    # end
    if nr == 0
        @assert nk > 0
        nk -= 1
        nr += kernel_size_n
    end
    mindexpr = :(m + mkern * $kernel_size_m)
    nindexpr = :(n + $nr + nkern * $kernel_size_n)
    nkernexpr = quote
        for nkern in 0:$nk-1
            Base.Cartesian.@nexpr $kernel_size_m m -> begin
                Base.Cartesian.@nexprs $kernel_size_n n -> A_m_n = A[$mindexpr,$nindexpr]
            end
            for p in 1:$P
                Base.Cartesian.@nexors $kernel_size_m m -> C_m = C[$mindexpr,p]
                Base.Cartesian.@nexprs $kernel_size_n n -> begin
                    vB = B[$nindexpr,p]
                    Base.Cartesian.@nexprs $kernel_size_m m -> begin
                        C_m = SIMDPirates.vmuladd(A_m_n, vB, C_m)
                    end
                end
                Base.Cartesian.@nexprs $M m -> C[$mindexpr,p] = C_m
            end
        end
    end
    q = quote
        $(Expr(:meta,:inline))
        # Inline, to prevent C from getting heap allocated, so long as it doesn't escape the calling function.
        C = MutableFixedSizePaddedMatrix{$M,$P,Core.VecElement{$W,$T}}(undef)
        @inbounds for mkern in 0:$(mk-1)
            # handle remainder in N first, to initialize
            Base.Cartesian.@nexpr $kernel_size_m m -> begin
                Base.Cartesian.@nexprs $nr n -> A_m_n = A[$mindexpr,n]
            end
            for p in 1:$P
                vB = B[1,p]
                Base.Cartesian.@nexprs $kernel_size_m m -> begin
                    C_m = SIMDPirates.evmul( A_m_1, vB)
                end
                Base.Cartesian.@nexprs $(nr-1) n -> begin
                    vB = B[n+1,p] 
                    Base.Cartesian.@nexprs $kernel_size_m m -> begin
                        C_m = SIMDPirates.vmuladd(A_m_{n+1}, vB, C_m)
                    end
                end
                Base.Cartesian.@nexprs $M m -> C[$mindexpr,p] = C_m
            end
            $(nk > 0 ? nkernexpr : nothing)
        end        
    end
    if mr > 0
        mindexpr2 = :(m + $(mk*kernel_size_m))
        nkernexpr2 = quote
            for nkern in 0:$nk-1
                Base.Cartesian.@nexpr $mr m -> begin
                    Base.Cartesian.@nexprs $kernel_size_n n -> A_m_n = A[$mindexpr2,$nindexpr]
                end
                for p in 1:$P
                    Base.Cartesian.@nexors $mr m -> C_m = C[$mindexpr2,p]
                    Base.Cartesian.@nexprs $kernel_size_n n -> begin
                        vB = B[$nindexpr,p]
                        Base.Cartesian.@nexprs $mr m -> begin
                            C_m = SIMDPirates.vmuladd(A_m_n, vB, C_m)
                        end
                    end
                    Base.Cartesian.@nexprs $M m -> C[$mindexpr2,p] = C_m
                end
            end
        end
        push!(q.args, quote
        @inbounds begin
                # handle remainder in N first, to initialize
                Base.Cartesian.@nexpr $mr m -> begin
                    Base.Cartesian.@nexprs $nr n -> A_m_n = A[$mindexpr2,n]
                end
                for p in 1:$P
                    vB = B[1,p]
                    Base.Cartesian.@nexprs $mr m -> begin
                        C_m = SIMDPirates.evmul(A_m_1, vB)
                    end
                    Base.Cartesian.@nexprs $(nr-1) n -> begin
                        vB = B[n+1,p] 
                        Base.Cartesian.@nexprs $mr m -> begin
                            C_m = SIMDPirates.vmuladd(A_m_{n+1}, vB, C_m)
                        end
                    end
                    Base.Cartesian.@nexprs $M m -> C[$mindexpr2,p] = C_m
                end
                $(nk > 0 ? nkernexpr2 : nothing)
            end
        end)
    end
    push!(q.args, :C)
    q
end

@generated function Base.:*(
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            B::AbstractFixedSizePaddedMatrix{N,P,NTuple{W,Core.VecElement{T}}}
        ) where {M,N,P,W,T}

    matrix_of_vecs_mul_quote(M,N,P,W,T)
end


@generated function Base.:*(
            A::LinearAlgebra.Adjoint{T,<: AbstractFixedSizePaddedMatrix{N,M,NTuple{W,Core.VecElement{T}}}},
            B::AbstractFixedSizePaddedMatrix{N,P,NTuple{W,Core.VecElement{T}}}
        ) where {M,N,P,W,T}

    matrix_of_vecs_mul_quote(M,N,P,W,T)
end
    







function matmul_sizes(C, A, B)
    MC, NC = maybestaticsize(C, Val{1:2}())
    MA, KA = maybestaticsize(A, Val{1:2}())
    KB, NB = maybestaticsize(B, Val{1:2}())
    M = VectorizationBase.static_promote(MC, MA)
    K = VectorizationBase.static_promote(KA, KB)
    N = VectorizationBase.static_promote(NC, NB)
    M, K, N
end

let GEMMLOOPSET = LoopVectorization.LoopSet(
    :(for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
          Cₘₙ = zero(eltype(C))
          for k ∈ 1:size(A,2)
              Cₘₙ += A[m,k] * B[k,n]
          end
          C[m,n] = Cₘₙ
      end)
    );
    order = LoopVectorization.choose_order(GEMMLOOPSET)
    mr = order[5]
    nr = last(order)
    @eval const mᵣ = $mr
    @eval const nᵣ = $nr
    for T ∈ [Int16, Int32, Int64, Float32, Float64]
    
    end

    
end

function loopmul!(C, A, B)
    M, K, N = matmul_sizes(C, A, B)
    @avx for m ∈ 1:M
        for n ∈ 1:N
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:K
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    C
end
function loopmuladd!(
    C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix
)
    M, K, N = matmul_sizes(C, A, B)
    @avx for m ∈ 1:M
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

function Base.copyto!(B::AbstractStrideArray{S,T,N}, A::AbstractStrideArray{S,T,N}) where {S,T,N}
    @avx for I ∈ CartesianIndices(B)
        B[I] = A[I]
    end
end
@inline prefetch2(x, i) = SIMDPirates.prefetch(gep(x, (VectorizationBase.extract_data(i) - 1,)), Val{2}(), Val{0}())
@inline prefetch3(x, i) = SIMDPirates.prefetch(gep(x, (VectorizationBase.extract_data(i) - 1,)), Val{1}(), Val{0}())
@inline prefetch2(x, i, j) = SIMDPirates.prefetch(gep(x, (VectorizationBase.extract_data(i) - 1, VectorizationBase.extract_data(j) - 1)), Val{2}(), Val{0}())
@inline prefetch3(x, i, j) = SIMDPirates.prefetch(gep(x, (VectorizationBase.extract_data(i) - 1, VectorizationBase.extract_data(j) - 1)), Val{1}(), Val{0}())
function copyto_prefetch2!(B::AbstractStrideArray{S,T,N}, A::AbstractStrideArray{S,T,N}, C::AbstractStrideArray{S,T,N}) where {S,T,N}
    Cptr = stridedpointer(C)
    for j ∈ 1:size(B,2)
        Cptr = gesp(Cptr, (0, (j-1)))
        @_avx unroll=4 for i ∈ 1:size(B,1)
            dummy = prefetch2(Cptr, i)
            B[i,j] = A[i,j]
        end
    end
end
function copyto_prefetch3!(B::AbstractStrideArray{S,T,N}, A::AbstractStrideArray{S,T,N}, C::AbstractStrideArray{S,T,N}) where {S,T,N}
    Cptr = stridedpointer(C)
    for j ∈ 1:size(B,2)
        Cptr = gesp(Cptr, (0, (j-1)))
        @_avx unroll=4 for i ∈ 1:size(B,1)
            dummy = prefetch3(Cptr, i)
            B[i,j] = A[i,j]
        end
    end
end

@generated function jmul!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}) where {Tc, Ta, Tb}
    L₁, L₂, L₃ = cache_sizes()
    W = VectorizationBase.pick_vector_width(promote_type(Tc, Ta, Tb))
    # We aim for using roughly half of the L1, L2, and L3 caches
    # kc = 5L₁ ÷ (12nᵣ * sizeof(Tb))
    kc = VectorizationBase.prevpow2( L₁ ÷ (2nᵣ * sizeof(Tb)) )
    mcrep =  L₂ ÷ (2kc * sizeof(Ta) * mᵣ * W)
    # mcrep =  5L₂ ÷ (12kc * sizeof(Ta) * mᵣ * W)
    # mcrep = VectorizationBase.prevpow2( L₂ ÷ (2kc * sizeof(Ta) * mᵣ * W) )
    mc = mcrep * mᵣ * W
    ncrep = VectorizationBase.prevpow2( L₃ ÷ (2kc * sizeof(Tb) * nᵣ) )
    nc = ncrep * nᵣ
    quote
        M, K, N = matmul_sizes(C, A, B)
        Niter, Nrem = divrem(N, $nc)
        Kiter, Krem = divrem(K, $kc)
        Miter, Mrem = divrem(M, $mc)
        (iszero(Miter) && iszero(Niter)) && return loopmul!(C, A, B)
        fill!(C, zero($Tc))
        # Cptr = stridedpointer(C)
        Aptr = stridedpointer(A)
        Bptr = stridedpointer(B)
        Cptr = stridedpointer(C)
        ptrL2 = pointer(L2CACHE); ptrL3 = pointer(L3CACHE);
        GC.@preserve C A B L2CACHE L3CACHE begin
            for no in 0:Niter-1
                for ko in 0:Kiter-1
                    # pack kc x nc block of B
                    Bpacked = PtrMatrix{$kc,$nc,$Tb,$kc}(ptrL3)
                    Bpmat = PtrMatrix{$kc,$nc}(gesp(Bptr, (ko*$kc, no*$nc)))
                    copyto!(Bpacked, Bpmat)
                    # Bprefetch = PtrMatrix{$kc,$nc}(gesp(Bptr, ((ko+1)*$kc, no*$nc)))
                    # copyto!(Bpacked, Bpmat, Bprefetch)
                    for mo in 0:Miter-1
                        # pack mc x kc block of A
                        Apacked = PtrMatrix{$mc,$kc,$Ta,$mc}(ptrL2)
                        Apmat = PtrMatrix{$mc,$kc}(gesp(Aptr, (mo*$mc, ko*$kc)))
                        copyto!(Apacked, Apmat)
                        # Aprefetch = PtrMatrix{$mc,$kc}(gesp(Aptr, ((mo+1)*$mc, ko*$kc)))
                        # copyto_prefetch3!(Apacked, Apmat, Aprefetch)
                        Cpmat = PtrMatrix{$mc,$nc}(gesp(Cptr, (mo*$mc, no*$nc)))
                        loopmuladd!(Cpmat, Apacked, Bpacked)
                    end
                    # Mrem
                    if Mrem > 0
                        Apacked_mrem = PtrMatrix{-1,$kc,$Ta,$mc}(ptrL2, Mrem)
                        Apmat_mrem = PtrMatrix{-1,$kc}(gesp(Aptr, (Miter*$mc, ko*$kc)), Mrem)
                        copyto!(Apacked_mrem, Apmat_mrem)
                        Cpmat_mrem = PtrMatrix{-1,$nc}(gesp(Cptr, (Miter*$mc, no*$nc)), Mrem)
                        loopmuladd!(Cpmat_mrem, Apacked_mrem, Bpacked)
                    end
                end
                # Krem
                if Krem > 0
                    # pack kc x nc block of B
                    Bpacked_krem = PtrMatrix{-1,$nc,$Tb,$kc}(ptrL3, Krem)
                    Bpmat_krem = PtrMatrix{-1,$nc}(gesp(Bptr, (Kiter*$kc, no*$nc)), Krem)
                    copyto!(Bpacked_krem, Bpmat_krem)
                    # Bpacked_krem = PtrMatrix{-1,$nc}(gesp(Bptr, (Kiter*$kc, no*$nc)), Krem)
                    for mo in 0:Miter-1
                        # pack mc x kc block of A
                        Apacked_krem = PtrMatrix{$mc,-1,$Ta,$mc}(ptrL2, Krem)
                        Apmat_krem = PtrMatrix{$mc,-1}(gesp(Aptr, (mo*$mc, Kiter*$kc)), Krem)
                        copyto!(Apacked_krem, Apmat_krem)
                        Cpmat = PtrMatrix{$mc,$nc}(gesp(Cptr, (mo*$mc, no*$nc)))
                        loopmuladd!(Cpmat, Apacked_krem, Bpacked_krem)
                    end
                    # Mrem
                    if Mrem > 0
                        Apacked_mrem_krem = PtrMatrix{-1,-1,$Ta,$mc}(ptrL2, Mrem, Krem)
                        Apmat_mrem_krem = PtrMatrix{-1,-1}(gesp(Aptr, (Miter*$mc, Kiter*$kc)), Mrem, Krem)
                        copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
                        Cpmat_mrem = PtrMatrix{-1,$nc}(gesp(Cptr, (Miter*$mc, no*$nc)), Mrem)
                        loopmuladd!(Cpmat_mrem, Apacked_mrem_krem, Bpacked_krem)
                    end
                end
            end
            # Nrem
            if Nrem > 0
                for ko in 0:Kiter-1
                    # pack kc x nc block of B
                    Bpacked_nrem = PtrMatrix{$kc,-1,$Tb,$kc}(ptrL3, Nrem)
                    Bpmat_nrem = PtrMatrix{$kc,-1}(gesp(Bptr, (ko*$kc, Niter*$nc)), Nrem)
                    copyto!(Bpacked_nrem, Bpmat_nrem)
                    # Bpacked_nrem = PtrMatrix{$kc,-1}(gesp(Bptr, (ko*$kc, Niter*$nc)), Nrem)
                    for mo in 0:Miter-1
                        # pack mc x kc block of A
                        Apacked = PtrMatrix{$mc,$kc,$Ta,$mc}(ptrL2)
                        Apmat = PtrMatrix{$mc,$kc}(gesp(Aptr, (mo*$mc, ko*$kc)))
                        copyto!(Apacked, Apmat)
                        Cpmat_nrem = PtrMatrix{$mc,-1}(gesp(Cptr, (mo*$mc, Niter*$nc)), Nrem)
                        loopmuladd!(Cpmat_nrem, Apacked, Bpacked_nrem)
                    end
                    # Mrem
                    if Mrem > 0
                        Apacked_mrem = PtrMatrix{-1,$kc,$Ta,$mc}(ptrL2, Mrem)
                        Apmat_mrem = PtrMatrix{-1,$kc}(gesp(Aptr, (Miter*$mc, ko*$kc)), Mrem)
                        copyto!(Apacked_mrem, Apmat_mrem)
                        Cpmat_mrem_nrem = PtrMatrix{-1,-1}(gesp(Cptr, (Miter*$mc, Niter*$nc)), Mrem, Nrem)
                        loopmuladd!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem)
                    end
                end
                # Krem
                if Krem > 0
                    # pack kc x nc block of B
                    Bpacked_krem_nrem = PtrMatrix{-1,-1,$Tb,$kc}(ptrL3, Krem, Nrem)
                    Bpmat_krem_nrem = PtrMatrix{-1,-1}(gesp(Bptr, (Kiter*$kc, Niter*$nc)), Krem, Nrem)
                    copyto!(Bpacked_krem_nrem, Bpmat_krem_nrem)
                    # Bpacked_krem_nrem = PtrMatrix{-1,-1}(gesp(Bptr, (Kiter*$kc, Niter*$nc)), Krem, Nrem)
                    for mo in 0:Miter-1
                        # pack mc x kc block of A
                        Apacked_krem = PtrMatrix{$mc,-1,$Ta,$mc}(ptrL2, Krem)
                        Apmat_krem = PtrMatrix{$mc,-1}(gesp(Aptr, (mo*$mc, Kiter*$kc)), Krem)
                        copyto!(Apacked_krem, Apmat_krem)
                        Cpmat_nrem = PtrMatrix{$mc,-1}(gesp(Cptr, (mo*$mc, Niter*$nc)), Nrem)
                        loopmuladd!(Cpmat_nrem, Apacked_krem, Bpacked_krem_nrem)
                    end
                    # Mrem
                    if Mrem > 0
                        Apacked_mrem_krem = PtrMatrix{-1,-1,$Ta,$mc}(ptrL2, Mrem, Krem)
                        Apmat_mrem_krem = PtrMatrix{-1,-1}(gesp(Aptr, (Miter*$mc, Kiter*$kc)), Mrem, Krem)
                        copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
                        Cpmat_mrem_nrem = PtrMatrix{-1,-1}(gesp(Cptr, (Miter*$mc, Niter*$nc)), Mrem, Nrem)
                        loopmuladd!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem)
                    end
                end
            end
        end # GC.@preserve
        C
    end # quote
end # function 




function LinearAlgebra.mul!(
    c::AbstractVector{T}, A::AbstractStrideMatrix{M,N,T}, b::AbstractVector{T}
) where {M,N,T}
    @assert size(c,1) == size(A,1)
    @assert size(b,1) == size(A,2)
    @avx for m ∈ 1:size(A,1)
        cₘ = zero(T)
        for n ∈ 1:size(A,2)
            cₘ += A[m,n] * b[n]
        end
        c[m] = cₘ
    end
    C
end
function nmul!(
    c::AbstractVector{T}, A::AbstractStrideMatrix{M,N,T}, b::AbstractVector{T}
) where {M,N,T}
    @assert size(c,1) == size(A,1)
    @assert size(b,1) == size(A,2)
    @avx for m ∈ 1:size(A,1)
        cₘ = zero(T)
        for n ∈ 1:size(A,2)
            cₘ -= A[m,n] * b[n]
        end
        c[m] = cₘ
    end
    C
end


LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractMatrix, B::AbstractMatrix) = loopmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractStrideMatrix, B::AbstractMatrix) = loopmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractStrideMatrix) = loopmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractStrideMatrix, B::AbstractMatrix) = loopmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractMatrix, B::AbstractStrideMatrix) = loopmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractStrideMatrix, B::AbstractStrideMatrix) = loopmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractStrideMatrix, B::AbstractStrideMatrix) = loopmul!(C, A, B)
function nmul!(
    C::AbstractStrideMatrix{<:Any,<:Any,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {T}
    M, K, N = matmul_sizes(C, A, B)
    @avx for m ∈ 1:M
        for n ∈ 1:N
            Cₘₙ = zero(T)
            for k ∈ 1:K
                Cₘₙ -= A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    C
end

@inline function Base.:*(
    sp::StackPointer,
    A::AbstractStrideMatrix{<:Any,<:Any,T},
    B::AbstractStrideMatrix{<:Any,<:Any,T}
) where {T}
    sp, D = PtrArray{T}(sp, (maybestaticsize(A, Val{1}()),maybestaticsize(B, Val{2}())))
    sp, mul!(D, A, B)
end
function Base.:*(
    A::AbstractStrideMatrix{M,<:Any,T},
    B::AbstractStrideMatrix{<:Any,N,T}
) where {M,N,T}
    mul!(FixedSizeArray{Tuple{M,N},T}(undef), A, B)
end
function Base.:*(
    A::AbstractStrideMatrix{M,<:Any,T},
    B::AbstractStrideMatrix{<:Any,-1,T}
) where {M,T}
    mul!(FixedSizeArray{T}(undef, (Static{M}(), size(B,2))), A, B)
end
function Base.:*(
    A::AbstractStrideMatrix{-1,<:Any,T},
    B::AbstractStrideMatrix{<:Any,N,T}
) where {N,T}
    mul!(FixedSizeArray{T}(undef, (size(A,1), Static{N}())), A, B)
end
function Base.:*(
    A::AbstractStrideMatrix{-1,<:Any,T},
    B::AbstractStrideMatrix{<:Any,-1,T}
) where {T}
    mul!(StrideArray{T}(undef, (size(A,1),size(B,2))), A, B)
end

function muladd!(
    C::AbstractStrideMatrix{<:Any,<:Any,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {T}
    M, K, N = matmul_sizes(C, A, B)
    @avx for m ∈ 1:M
        for n ∈ 1:N
            Cₘₙ = zero(T)
            for k ∈ 1:K
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] += Cₘₙ
        end
    end
    C
end
function mulsub!(
    C::AbstractStrideMatrix{<:Any,<:Any,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {T}
    M, K, N = matmul_sizes(C, A, B)
    @avx for m ∈ 1:M
        for n ∈ 1:N
            Cₘₙ = zero(T)
            for k ∈ 1:K
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ - C[m,n]
        end
    end
    C
end
function nmuladd!(
    C::AbstractStrideMatrix{<:Any,<:Any,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {T}
    M, K, N = matmul_sizes(C, A, B)
    @avx for m ∈ 1:M
        for n ∈ 1:N
            Cₘₙ = zero(T)
            for k ∈ 1:K
                Cₘₙ -= A[m,k] * B[k,n]
            end
            C[m,n] += Cₘₙ
        end
    end
    C
end
function nmulsub!(
    C::AbstractStrideMatrix{<:Any,<:Any,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {T}
    M, K, N = matmul_sizes(C, A, B)
    @avx for m ∈ 1:M
        for n ∈ 1:N
            Cₘₙ = zero(T)
            for k ∈ 1:K
                Cₘₙ -= A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ - C[m,n]
        end
    end
    C
end

@inline extract_λ(a) = a
@inline extract_λ(a::UniformScaling) = a.λ
@inline function Base.:*(A::AbstractFixedSizeArray{S,T,N,X,L}, bλ::Union{T,UniformScaling{T}}) where {S,T<:Real,N,X,L}
    mv = FixedSizeArray{S,T,N,X,L}(undef)
    b = extract_λ(bλ)
    @avx for i ∈ eachindex(A)
        mv[i] = A[i] * b
    end
    mv
end
@inline function Base.:*(Aadj::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,X,L}}, bλ::Union{T,UniformScaling{T}}) where {S,T<:Real,N,X,L}
    mv = FixedSizeArray{S,T,N,X,L}(undef)
    A = Aadj.parent
    b = extract_λ(bλ)
    @avx for i ∈ eachindex(A)
        mv[i] = A[i] * b
    end
    mv'
end
function Base.:*(
    sp::StackPointer,
    A::AbstractFixedSizeArray{S,T,N,X,L},
    bλ::Union{T,UniformScaling{T}}
) where {S,T<:Real,N,X,L}
    mv = PtrArray{S,T,N,X,L}(pointer(sp,T))
    b = extract_λ(bλ)
    @avx for i ∈ eachindex(A)
        mv[i] = A[i] * b
    end
    sp + VectorizationBase.align(sizeof(T)*L), mv
end
function Base.:*(
    sp::StackPointer,
    Aadj::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,X,L}},
    bλ::Union{T,UniformScaling{T}}
) where {S,T<:Real,N,X,L}
    mv = PtrArray{S,T,N,X,L}(pointer(sp,T))
    A = Aadj.parent
    b = extract_λ(bλ)
    @avx for i ∈ eachindex(A)
        mv[i] = A[i] * b
    end
    sp + VectorizationBase.align(sizeof(T)*L), mv'
end
@inline function Base.:*(a::T, B::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
    mv = FixedSizeArray{S,T,N,X,L}(undef)
    @avx for i ∈ eachindex(B)
        mv[i] = a * B[i]
    end
        # mv
    ConstantFixedSizeArray(mv)
end

@inline function nmul!(
    D::AbstractMatrix{T},
    A′::LinearAlgebra.Adjoint{T,<:AbstractMatrix{T}},
    X::AbstractMatrix{T}
) where {T <: BLAS.BlasFloat}
    BLAS.gemm!('T','N',-one(T),A′.parent,X,zero(T),D)
end


function LinearAlgebra.mul!(
    C::AbstractStrideMatrix{<:Any,<:Any,T},
    A::LinearAlgebra.Diagonal{T,<:AbstractStrideVector{<:Any,T}},
    B::AbstractMatrix{T}
) where {T}
    M, K, N = matmul_sizes(C, A, B)
    MandK = VectorizationBase.static_promote(M, K)
    vA = parent(A)
    @avx for n ∈ 1:N, m ∈ 1:MandK
        C[m,n] = vA[m] * B[m,n]
    end
    C
end
function Base.:*(
    A::LinearAlgebra.Diagonal{T,<:AbstractStrideVector{M,T}},
    B::AbstractStrideMatrix{M,N,T}
) where {M,N,T}
    mul!(similar(B), A, B)
end
function Base.:*(
    sp::StackPointer,
    A::LinearAlgebra.Diagonal{T,<:AbstractStrideVector{M,T}},
    B::AbstractStrideMatrix{M,N,T}
) where {M,N,T}
    sp, C = similar(sp, B)
    sp, mul!(C, A, B)
end


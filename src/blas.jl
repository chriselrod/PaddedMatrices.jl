


@inline function divrem_fast(x::I, y::I) where {I <: Integer}
    x32 = x % UInt32; y32 = y % UInt32
    d = Base.sdiv_int(x32, y32)
    r = x32 - d * y32
    d % I, r % I
end
@inline div_fast(x::I, y::I) where {I <: Integer} = Base.sdiv_int(x % UInt32, y % UInt32) % I

function Base.copyto!(B::AbstractStrideArray{S,T,N}, A::AbstractStrideArray{S,T,N}) where {S,T,N}
    @avx for I ∈ eachindex(A, B)
        B[I] = A[I]
    end
    B
end
# function copyto_prefetch2!(B::AbstractStrideArray{S,T,N}, A::AbstractStrideArray{S,T,N}, C::AbstractStrideArray{S,T,N}) where {S,T,N}
#     Cptr = stridedpointer(C)
#     for j ∈ 1:size(B,2)
#         Cptr = gesp(Cptr, (0, (j-1)))
#         @_avx unroll=4 for i ∈ 1:size(B,1)
#             dummy = prefetch2(Cptr, i)
#             B[i,j] = A[i,j]
#         end
#     end
# end
# function copyto_prefetch3!(B::AbstractStrideArray{S,T,N}, A::AbstractStrideArray{S,T,N}, C::AbstractStrideArray{S,T,N}) where {S,T,N}
#     Cptr = stridedpointer(C)
#     for j ∈ 1:size(B,2)
#         Cptr = gesp(Cptr, (0, (j-1)))
#         @_avx unroll=4 for i ∈ 1:size(B,1)
#             dummy = prefetch3(Cptr, i)
#             B[i,j] = A[i,j]
#         end
#     end
# end

function matmul_params(::Type{T}) where {T}
    W = VectorizationBase.pick_vector_width(T)
    # kc = 10(L₁ ÷ (20nᵣ * sizeof(T)))
    # kc = 15(L₁ ÷ (20nᵣ * sizeof(T)))
    L₁ratio = L₁ ÷ (nᵣ * sizeof(T))
    kc = round(Int, 0.65L₁ratio)
    # mcrep =  L₂ ÷ (2kc * sizeof(T) * mᵣ * W)
    mcrep =  5L₂ ÷ (8kc * sizeof(T) * mᵣ * W)
    ncrep = L₃ ÷ (kc * sizeof(T) * nᵣ)
    # ncrep = 5L₃ ÷ (16kc * sizeof(T) * nᵣ)
    mc = mcrep * mᵣ * W
    nc = round(Int, 0.4ncrep * nᵣ) #* VectorizationBase.NUM_CORES
    mc, kc, nc
end
function matmul_params_val(::Type{T}) where {T}
    mc, kc, nc = matmul_params(T)
    Val(mc), Val(kc), Val(nc)
end



function pack_B(Bptr, kc, nc, ::Type{Tb}, koffset::Int, noffset::Int) where {Tb}
    # Bpacked = PtrMatrix(threadlocal_L3CACHE_pointer(Tb), kc, nc, kc*sizeof(Tb))
    Bpacked = PtrMatrix(threadlocal_L3CACHE_pointer(Tb), kc, nc, VectorizationBase.align(kc,Tb)*sizeof(Tb))
    Bpmat = PtrMatrix(gesp(Bptr, (koffset, noffset)), kc, nc)
    copyto!(Bpacked, Bpmat)
    Bpacked
end
function pack_A(Aptr, mc, kc, ::Type{Ta}, moffset::Int, koffset::Int) where {Ta}
    # Apacked = PtrArray{Tuple{mᵣW,-1,-1},Ta,3,Tuple{1,mᵣW,-1},2,1,false}(threadlocal_L2CACHE_pointer(Ta), (kc,mcreps), ((mᵣW*kc),))
    # Apacked = PtrMatrix(threadlocal_L2CACHE_pointer(Ta), mc, kc, mc*sizeof(Ta))
    Apacked = PtrMatrix(threadlocal_L2CACHE_pointer(Ta), mc, kc, VectorizationBase.align(mc, Ta)*sizeof(Ta))
    Apmat = PtrMatrix(gesp(Aptr, (moffset, koffset)), mc, kc)
    # @show (reinterpret(Int, pointer(Apmat)) - reinterpret(Int,pointer(Aptr))) >>> 3
    # copyto!(Apacked, Apmat)
    Apacked, Apmat
end

function jmulpackAonly!(
    C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, ::Val{mc}, ::Val{kc}, ::Val{nc}, (M, K, N) = matmul_sizes(C, A, B)
) where {Tc, Ta, Tb, mc, kc, nc}
    W = VectorizationBase.pick_vector_width(Tc)
    mᵣW = mᵣ * W

    num_m_iter = cld(M, mc)
    mreps_per_iter = div_fast(M, num_m_iter)
    mreps_per_iter += mᵣW - 1
    mcrepetitions = mreps_per_iter ÷ mᵣW
    mreps_per_iter = mᵣW * mcrepetitions
    Miter = num_m_iter - 1
    Mrem = M - Miter * mreps_per_iter
    #(iszero(Miter) && iszero(Niter)) && return loopmul!(C, A, B)
    num_k_iter = cld(K, kc)
    kreps_per_iter = ((div_fast(K, num_k_iter)) + 3) & -4
    Krem = K - (num_k_iter - 1) * kreps_per_iter
    Kiter = num_k_iter - 1

    Aptr = stridedpointer(A)
    Bptr = stridedpointer(B)
    Cptr = stridedpointer(C)
    GC.@preserve C A B LCACHEARRAY begin
        # Krem
        # pack kreps_per_iter x nc block of B
        Bpacked = PtrMatrix(Bptr, Krem, N)
        moffset = 0
        for mo in 0:Miter
            msize = ifelse(mo == Miter, Mrem, mreps_per_iter)
            # pack mreps_per_iter x kreps_per_iter block of A
            Apacked, Asubset = pack_A(Aptr, msize, Krem, Tc, moffset, 0)
            Cpmat = PtrMatrix(gesp(Cptr, (moffset, 0)), msize, N)
            packaloopmul!(Cpmat, Apacked, Asubset, Bpacked, α, β, (msize, Krem, N))
            moffset += mreps_per_iter
        end
        koffset = Krem
        for ko in 1:Kiter
            # pack kreps_per_iter x nc block of B
            Bpacked = PtrMatrix(gesp(Bptr, (koffset, 0)), kreps_per_iter, N)
            moffset = 0
            for mo in 0:Miter
                msize = ifelse(mo == Miter, Mrem, mreps_per_iter)
                # pack mreps_per_iter x kreps_per_iter block of A
                Apacked, Asubset = pack_A(Aptr, msize, kreps_per_iter, Tc, moffset, koffset)
                Cpmat = PtrMatrix(gesp(Cptr, (moffset, 0)), msize, N)
                packaloopmul!(Cpmat, Apacked, Asubset, Bpacked, α, Val{1}(), (msize, kreps_per_iter, N))
                moffset += mreps_per_iter
            end
            koffset += kreps_per_iter
        end
    end # GC.@preserve
    C
end
function jmulpackAB!(
    C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, ::Val{mc}, ::Val{kc}, ::Val{nc}, (M, K, N) = matmul_sizes(C, A, B)
) where {Tc, Ta, Tb, mc, kc, nc}
    W = VectorizationBase.pick_vector_width(Tc)
    mᵣW = mᵣ * W

    num_n_iter = cld(N, nc)
    nreps_per_iter = div_fast(N, num_n_iter)
    nreps_per_iter += nᵣ - 1
    ncrepetitions = div_fast(nreps_per_iter, nᵣ)
    nreps_per_iter = nᵣ * ncrepetitions
    Niter = num_n_iter - 1
    Nrem = N - Niter * nreps_per_iter

    num_m_iter = cld(M, mc)
    mreps_per_iter = div_fast(M, num_m_iter)
    mreps_per_iter += mᵣW - 1
    mcrepetitions = div_fast(mreps_per_iter, mᵣW)
    mreps_per_iter = mᵣW * mcrepetitions
    Miter = num_m_iter - 1
    Mrem = M - Miter * mreps_per_iter
    
    num_k_iter = cld(K, kc)
    kreps_per_iter = ((div_fast(K, num_k_iter)) + 3) & -4
    Krem = K - (num_k_iter - 1) * kreps_per_iter
    Kiter = num_k_iter - 1

    # @show mreps_per_iter, kreps_per_iter, nreps_per_iter
    # @show Miter, Kiter, Niter
    # @show Mrem, Krem, Nrem
    
    Aptr = stridedpointer(A)
    Bptr = stridedpointer(B)
    Cptr = stridedpointer(C)
    # ptrL3 = threadlocal_L3CACHE_pointer(Tc)
    # ptrL2 = threadlocal_L2CACHE_pointer(Tc)
    noffset = 0
    GC.@preserve C A B LCACHEARRAY begin
        for no in 0:Niter
            # Krem
            # pack kc x nc block of B
            nsize = ifelse(no == Niter, Nrem, nreps_per_iter)
            Bpacked = pack_B(Bptr, Krem, nsize, Tc, 0, noffset)
            moffset = 0
            for mo in 0:Miter
                msize = ifelse(mo == Miter, Mrem, mreps_per_iter)
                # pack mreps_per_iter x kreps_per_iter block of A
                Apacked, Asubset = pack_A(Aptr, msize, Krem, Tc, moffset, 0)
                Cpmat = PtrMatrix(gesp(Cptr, (moffset, noffset)), msize, nsize)
                packaloopmul!(Cpmat, Apacked, Asubset, Bpacked, α, β, (msize, Krem, nsize))
                moffset += mreps_per_iter
            end
            koffset = Krem
            for ko in 1:Kiter
                # pack kreps_per_iter x nc block of B
                Bpacked = pack_B(Bptr, kreps_per_iter, nsize, Tc, koffset, noffset)
                moffset = 0
                for mo in 0:Miter
                    msize = ifelse(mo == Miter, Mrem, mreps_per_iter)
                    # pack mreps_per_iter x kreps_per_iter block of A
                    Apacked, Asubset = pack_A(Aptr, msize, kreps_per_iter, Tc, moffset, koffset)
                    Cpmat = PtrMatrix(gesp(Cptr, (moffset, noffset)), msize, nsize)
                    packaloopmul!(Cpmat, Apacked, Asubset, Bpacked, α, Val{1}(), (msize, kreps_per_iter, nsize))
                    moffset += mreps_per_iter
                end
                koffset += kreps_per_iter
            end
            noffset += nreps_per_iter
        end
    end # GC.@preserve
    C
end


# contiguousstride1(::Any) = false
# contiguousstride1(::DenseArray) = true
# contiguousstride1(A::LinearAlgebra.StridedArray) = isone(stride1(A))
# contiguousstride1(::SubArray{T,N,P,S}) where {T,N,P,S<:Tuple{Int,Vararg}} = false



function jmul!(
    C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, ::Val{mc}, ::Val{kc}, ::Val{nc}, (M, K, N) = matmul_sizes(C, A, B)
) where {Tc, Ta, Tb, mc, kc, nc}
    # if K * N ≤ kc * nc#L₃ * VectorizationBase.NUM_CORES
        # W = VectorizationBase.pick_vector_width(Tc)
        # if contiguousstride1(A) && ( (M ≤ 72)  || ((2M ≤ 5mc) && iszero(stride(A,2) % W)))
    if mc * kc ≥ M * K
        loopmul!(C, A, B, α, β, (M,K,N))
        return C
    elseif kc * nc > K * N
    # else
        return jmulpackAonly!(C, A, B, α, β, Val{mc}(), Val{kc}(), Val{nc}(), (M,K,N))
    else
        return jmulpackAB!(C, A, B, α, β, Val{mc}(), Val{kc}(), Val{nc}(), (M,K,N))
    end
    # else
    #     return jmulh!(C, A, B, α, β, Val{mc}(), Val{kc}(), Val{nc}(), (M,K,N))
    # end
end # function

@inline jmul!(C::AbstractMatrix, A::LinearAlgebra.Adjoint, B::LinearAlgebra.Adjoint, α, β) = (jmul!(C', B', A'); C)
@inline function jmul!(
    C::AbstractMatrix,
    A::AbstractStrideArray{Sa,Ta,2,Tuple{Xa,1}},
    B::AbstractStrideArray{Sb,Tb,2,Tuple{Xb,1}}
) where {Sa, Ta, Xa, Sb, Tb, Xb}
    jmul!(C', B', A')
    C
end

# function jmult!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, ::Val{mc}, ::Val{kc}, ::Val{nc}) where {Tc, Ta, Tb, mc, kc, nc, log2kc}
#     M, K, N = matmul_sizes(C, A, B)
#     Niter, Nrem = divrem(N, Static{nc}())
#     Miter, Mrem = divrem(M, Static{mc}())
#     if iszero(Niter) & iszero(Miter)
#         if Mrem*sizeof(Ta) > 255
#             loopmulprefetch!(C, A, B, α, β); return C
#         else
#             loopmul!(C, A, B, α, β); return C
#         end
#     end
#     Km1 = VectorizationBase.staticm1(K)
#     Kiter, _Krem = divrem(Km1, Static{kc}())
#     Krem = VectorizationBase.staticp1(_Krem)
#     Aptr = stridedpointer(A)
#     Bptr = stridedpointer(B)
#     Cptr = stridedpointer(C)
#     Base.Threads.@sync GC.@preserve C A B LCACHEARRAY begin
#         for no in 0:Niter-1
#             # Krem
#             # pack kc x nc block of B
#             # Base.Threads.@spawn begin
#             begin
#                 Bpacked_krem = pack_B_krem(Bptr, Val{kc}(), Val{nc}(), Tb, Krem, no)
#                 Base.Threads.@sync begin
#                     for mo in 0:Miter-1
#                         # pack mc x kc block of A
#                         Base.Threads.@spawn begin
#                             # begin
#                             Apacked_krem = pack_A_krem(Aptr, Val{mc}(), Ta, mo, Krem)
#                             Cpmat = PtrMatrix(gesp(Cptr, (mo*mc, no*nc)), mc, nc)
#                             loopmulprefetch!(Cpmat, Apacked_krem, Bpacked_krem, α, β)
#                         end
#                     end
#                     # Mrem
#                     if Mrem > 0
#                         Base.Threads.@spawn begin
#                             # begin
#                             Apacked_mrem_krem = pack_A_mrem_krem(Aptr, Val{mc}(), Ta, Mrem, Krem, Miter)
#                             Cpmat_mrem = PtrMatrix(gesp(Cptr, (Miter*mc, no*nc)), Mrem, nc)
#                             loopmulprefetch!(Cpmat_mrem, Apacked_mrem_krem, Bpacked_krem, α, β)
#                         end
#                     end
#                 end # sync
#                 k = VectorizationBase.unwrap(Krem)
#                 for ko in 1:Kiter
#                     # pack kc x nc block of B
#                     Bpacked = pack_B(Bptr, Val{kc}(), Val{nc}(), Tb, k, no)
#                     # Bprefetch = PtrMatrix{kc,nc}(gesp(Bptr, ((ko+1)*kc, no*nc)))
#                     # copyto!(Bpacked, Bpmat, Bprefetch)
#                     Base.Threads.@sync begin
#                         for mo in 0:Miter-1
#                             Base.Threads.@spawn begin
#                                 # begin
#                                 Apacked = pack_A(Aptr, Val{mc}(), Val{kc}(), Ta, mo, k)
#                                 Cpmat = PtrMatrix(gesp(Cptr, (mo*mc, no*nc)), mc, nc)
#                                 loopmulprefetch!(Cpmat, Apacked, Bpacked, α, Val{1}())
#                             end
#                         end
#                         # Mrem
#                         if Mrem > 0
#                             Base.Threads.@spawn begin
#                                 # begin
#                                 Apacked_mrem = pack_A_mrem(Aptr, Val{mc}(), Val{kc}(), Ta, Mrem, k, Miter)
#                                 Cpmat_mrem = PtrMatrix(gesp(Cptr, (Miter*mc, no*nc)), Mrem, nc)
#                                 loopmulprefetch!(Cpmat_mrem, Apacked_mrem, Bpacked, α, Val{1}())
#                             end
#                         end
#                     end # sync
#                     k += kc
#                 end
#             end # spawnp
#         end # nloop
#         # Nrem
#         if Nrem > 0
#             # Base.Threads.@spawn begin
#             begin
#                 # Krem
#                 # pack kc x nc block of B
#                 Bpacked_krem_nrem = pack_B_krem_nrem(Bptr, Val{kc}(), Val{nc}(), Tb, Krem, Nrem, Niter)
#                 # Bpacked_krem_nrem = PtrMatrix{-1,-1}(gesp(Bptr, (Kiter*kc, Niter*nc)), Krem, Nrem)
#                 Base.Threads.@sync begin
#                     for mo in 0:Miter-1
#                         Base.Threads.@spawn begin
#                             # begin
#                             # pack mc x kc block of A
#                             Apacked_krem = pack_A_krem(Aptr, Val{mc}(), Ta, mo, Krem)
#                             Cpmat_nrem = PtrMatrix(gesp(Cptr, (mo*mc, Niter*nc)), mc, Nrem)
#                             loopmulprefetch!(Cpmat_nrem, Apacked_krem, Bpacked_krem_nrem, α, β)
#                         end
#                     end
#                     # Mrem
#                     if Mrem > 0
#                         Base.Threads.@spawn begin
#                             # begin
#                             Apacked_mrem_krem = pack_A_mrem_krem(Aptr, Val{mc}(), Ta, Mrem, Krem, Miter)
#                             Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
#                             loopmulprefetch!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem, α, β)
#                         end
#                     end
#                 end # sync
#                 k = VectorizationBase.unwrap(Krem)
#                 for ko in 1:Kiter
#                     # pack kc x nc block of B
#                     Bpacked_nrem = pack_B_nrem(Bptr, Val{kc}(), Val{nc}(), Tb, k, Nrem, Niter)
#                     Base.Threads.@sync begin
#                         for mo in 0:Miter-1
#                             Base.Threads.@spawn begin
#                                 # begin
#                                 # pack mc x kc block of A
#                                 Apacked = pack_A(Aptr, Val{mc}(), Val{kc}(), Ta, mo, k)
#                                 Cpmat_nrem = PtrMatrix(gesp(Cptr, (mo*mc, Niter*nc)), mc, Nrem)
#                                 loopmulprefetch!(Cpmat_nrem, Apacked, Bpacked_nrem, α, Val{1}())
#                             end
#                         end
#                         # Mrem
#                         if Mrem > 0
#                             Base.Threads.@spawn begin
#                                 # begin
#                                 Apacked_mrem = pack_A_mrem(Aptr, Val{mc}(), Val{kc}(), Ta, Mrem, k, Miter)
#                                 Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
#                                 loopmulprefetch!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem, α, Val{1}())
#                             end
#                         end
#                     end # sync
#                     k += kc
#                 end
#             end
#         end
#     end # GC.@preserve, sync
#     C
# end

maybeinline(::Any) = false
@generated function maybeinline(::AbstractFixedSizeMatrix{M,N,T}) where {M,N,T}
    sizeof(T) * M * N < 24mᵣ * nᵣ
end
                                                                       

@inline function jmul!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}) where {Tc, Ta, Tb}
    maybeinline(C) && return inlineloopmul!(C, A, B, Val{1}(), Val{0}())
    mc, kc, nc = matmul_params_val(Tc)
    jmul!(C, A, B, Val{1}(), Val{0}(), mc, kc, nc)
end
@inline function jmul!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α) where {Tc, Ta, Tb}
    maybeinline(C) && return inlineloopmul!(C, A, B, α, Val{0}())
    mc, kc, nc = matmul_params_val(Tc)
    jmul!(C, A, B, α, Val{0}(), mc, kc, nc)
end
@inline function jmul!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β) where {Tc, Ta, Tb}
    maybeinline(C) && return inlineloopmul!(C, A, B, α, β)
    mc, kc, nc = matmul_params_val(Tc)
    jmul!(C, A, B, α, β, mc, kc, nc)
end
function jmult!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}) where {Tc, Ta, Tb}
    mc, kc, nc = matmul_params_val(Tc)
    jmult!(C, A, B, mc, kc, nc)
end





# function packarray_A!(dest::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}}, src::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,K}}) where {Mᵣ,K,Mᵢ}
#     # @inbounds for mᵢ ∈ axes(dest,3), k ∈ axes(dest,2), mᵣ ∈ axes(dest,1)
#     @avx for mᵢ ∈ axes(dest,3), k ∈ axes(dest,2), mᵣ ∈ axes(dest,1)
#         dest[mᵣ,k,mᵢ] = src[mᵣ,mᵢ,k]
#     end
# end
function packarray_B!(dest::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ},T}, src::AbstractStrideArray{Tuple{K,Nᵣ,Nᵢ},T}) where {Nᵣ,K,Nᵢ,T}
    @avx for k ∈ axes(dest,2), nᵢ ∈ axes(dest,3), nᵣ ∈ axes(dest,1)
        dest[nᵣ,k,nᵢ] = src[k,nᵣ,nᵢ]
    end
end
function packarray_B!(dest::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ},T}, src::AbstractStrideArray{Tuple{K,Nᵣ,Nᵢ},T}, nrem) where {Nᵣ,K,Nᵢ,T}
    nᵢaxis = 1:(size(src,3) - !iszero(nrem))
    @avx inline=true for nᵢ ∈ nᵢaxis, k ∈ axes(dest,2), nᵣᵢ ∈ axes(dest,1)
        dest[nᵣᵢ,k,nᵢ] = src[k,nᵣᵢ,nᵢ]
    end
    if !iszero(nrem)
        nᵢ = size(dest,3)
        nᵣₛ = Static{nᵣ}()
        @avx inline=true for k ∈ axes(dest,2), nᵣᵢ ∈ 1:nᵣₛ
            dest[nᵣᵢ,k,nᵢ] = nᵣᵢ ≤ nrem ? src[k,nᵣᵢ,nᵢ] : zero(T)
        end
        # @inbounds for k ∈ axes(dest,2)
             # @simd ivdep for nᵣᵢ ∈ 1:nᵣₛ
                 # dest[nᵣᵢ,k,nᵢ] = nᵣᵢ ≤ nrem ? src[k,nᵣᵢ,nᵢ] : zero(T)
             # end
        # end
    end
end


#=
function calc_blocksize(iter, step, max, ::Val{round_to}) where {round_to}
    if iter + step > max
        remaining = max - iter
        remaining, divrem(remaining, round_to)
    elseif nciter + 2nc > N
        half_remaining = (max - iter) >>> 1
        half_remaining, (half_remaining ÷ nᵣ, 0)
    else
        nc, (nc ÷ nᵣ, 0)
    end
end
function calc_blocksize(iter, step, max)
    if iter + nc > N
        N - nciter
    elseif nciter + 2nc > N
        (max - iter) >>> 1
    else
        nc
    end
end



function jmulh!(
    C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α = Val(1), β = Val(0)
) where {Tc,Ta,Tb}


    W, Wshift = VectorizationBase.pick_vector_width_shift(Tc)
    mᵣW = mᵣ << Wshift
    mc, kc, nc = matmul_params(Tc)
    M, K, N = matmul_sizes(C, A, B)

    mcrep = mc ÷ mᵣW
    Ablocklen = mc * kc
    ncrep = nc ÷ nᵣ

    Aptr = stridedpointer(A)
    Bptr = stridedpointer(B)
    Cptr = stridedpointer(C)
    ptrL3 = threadlocal_L3CACHE_pointer(Tc)
    ptrL2 = threadlocal_L2CACHE_pointer(Tc)

    
    nciter = 0
    GC.@preserve C A B LCACHEARRAY begin
        while nciter < N
            ncblock, (ncreps, ncrem) = calc_blocksize(nciter, nc, N, Val{nᵣ}())

            
            kciter = 0
            while kciter < K
                kcblock = calc_blocksize(kciter, kc, K)


                ncreprem, ncrepremrem = divrem(Nrem, nᵣ)
                ncrepremc = ncreprem + !(iszero(ncrepremrem))
                Bpacked_krem = PtrArray{Tuple{nᵣ,-1,-1},Tc,3,Tuple{1,nᵣ,-1},2,1,false}(ptrL3, (kcblock,ncblock), (nᵣ * kcblock,))
                Bpmat_krem = PtrArray{Tuple{-1,nᵣ,-1},Tb,3}(gesp(Bptr, (0, Noff)), (,ncrepremc))
                packarray_B!(Bpacked_krem, Bpmat_krem, ncrepremrem)
                Bpacked_krem_nrem = PtrMatrix{-1,-1,Tc,1,nᵣ,2,0,false}(gep(stridedpointer(Bpacked_krem), (0,0,ncreprem)), (ncrepremrem,Krem), tuple())


                
                mciter = 0
                while mciter < M
                    mcblock, (mcreps, mcrem) = calc_blocksize(mciter, mc, M, Val{mᵣW}())


                    Apacked_krem = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,1,false}(ptrL2, (Krem,mcreps), (Krem*$mᵣW,))
                    Apmat_krem = PtrArray{Tuple{$mᵣW,-1,-1},$Ta,3,Tuple{1,$mᵣW,-1},2,1,false}(gep(Aptr, (mo*$mc, 0)), ($mcrep,Krem), Aptr.strides)
                    packarray_A!(Apacked_krem, Apmat_krem)
                    Cptr_off = gep(Cptr, (mo*$mc, no*$nc))
                    Cx = first(Cptr.strides)
                    Cpmat = PtrArray{Tuple{$mᵣW,-1,nᵣ,-1},$Tc,4,Tuple{1,$mᵣW,-1,-1},2,2,false}(Cptr_off, ($mcrep,$ncrep), (Cx,Cx*nᵣ))
                    loopmul!(Cpmat, Apacked_krem, Bpacked_krem, α, β)


                    
                    mciter += mcblock
                end
                kciter = kcblock
            end
            nciter += ncblock
        end        
    end
    
end
=#

# function Apack_krem(Aptr::AbstractStridedPointer{Ta}, ptrL2, ::Val{mᵣW}, ::Type{Tc}, mcrepetitions, Krem, mo, mreps_per_iter) where {mᵣW, Ta, Tc}
#     Apacked_krem = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,1,false}(ptrL2, (Krem,mcrepetitions), (Krem*mᵣW,))
#     Apmat_krem = PtrArray{Tuple{mᵣW,-1,-1},Ta,3}(gesp(Aptr, (mo*mreps_per_iter, 0)), (mcrepetitions,Krem))
#     Apacked_krem, Apmat_krem
# end
function Apack(
    Aptr::AbstractStridedPointer{Ta}, ptrL2, ::Val{mᵣW}, ::Type{Tc}, Msub, Ksub, m, k
) where {mᵣW, Ta, Tc}
    mreps_per_iter = mᵣW * Msub
    # Apacked = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,1,false}(ptrL2, (kreps_per_iter,mcrepetitions), ((mᵣW*kc),))
    # Apmat = PtrArray{Tuple{mᵣW,-1,-1},Ta,3}(gesp(Aptr, (mo*mreps_per_iter, k)), (mcrepetitions,kreps_per_iter))
    Apacked = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,1,false}(ptrL2, (Ksub,Msub), ((mᵣW*Ksub),))
    Apmat = PtrArray{Tuple{mᵣW,-1,-1},Ta,3}(gesp(Aptr, (m*mreps_per_iter, k)), (Msub,Ksub))
    Apacked, Apmat
end


roundupnᵣ(x) = (xnr = x + nᵣ; xnr - (xnr % nᵣ))

function jmulh!(
    C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α = Val(1), β = Val(0)
# ) where {Tc, Ta, Tb}
) where {Ta, Tb, Tc}
    mc, kc, nc = matmul_params_val(Tc)
    jmulh!(C, A, B, α, β, mc, kc, nc)
end
function jmulh!(
    C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, ::Val{mc}, ::Val{kc}, ::Val{nc}, (M, K, N) = matmul_sizes(C, A, B)
# ) where {Tc, Ta, Tb}
) where {Ta, Tb, Tc, mc, kc, nc}
    W = VectorizationBase.pick_vector_width(Tc)
    mᵣW = mᵣ * W

    num_n_iter = cld(N, nc)
    nreps_per_iter = div_fast(N, num_n_iter)
    nreps_per_iter += nᵣ - 1
    ncrepetitions = div_fast(nreps_per_iter, nᵣ)
    nreps_per_iter = nᵣ * ncrepetitions
    Niter = num_n_iter - 1
    Nrem = N - Niter * nreps_per_iter

    num_m_iter = cld(M, mc)
    mreps_per_iter = div_fast(M, num_m_iter)
    mreps_per_iter += mᵣW - 1
    mcrepetitions = div_fast(mreps_per_iter, mᵣW)
    mreps_per_iter = mᵣW * mcrepetitions
    Miter = num_m_iter - 1
    Mrem = M - Miter * mreps_per_iter
    
    num_k_iter = cld(K, kc)
    kreps_per_iter = ((div_fast(K, num_k_iter)) + 3) & -4
    Krem = K - (num_k_iter - 1) * kreps_per_iter
    Kiter = num_k_iter - 1

    # @show mreps_per_iter, kreps_per_iter, nreps_per_iter
    # @show Miter, Kiter, Niter
    # @show Mrem, Krem, Nrem
    
    Aptr = stridedpointer(A)
    Bptr = stridedpointer(B)
    Cptr = stridedpointer(C)
    ptrL3 = threadlocal_L3CACHE_pointer(Tc)
    ptrL2 = threadlocal_L2CACHE_pointer(Tc)
    Noff = 0
    GC.@preserve C A B LCACHEARRAY begin
        for no in 0:Niter-1
            # Krem
            # pack kc x nc block of B
            
            Bpacked_krem = PtrArray{Tuple{nᵣ,-1,-1},Tc,3,Tuple{1,nᵣ,-1},2,1,false}(ptrL3, (Krem,ncrepetitions), (nᵣ * Krem,))
            Bpmat_krem = PtrArray{Tuple{-1,nᵣ,-1},Tb,3}(gesp(Bptr, (0, Noff)), (Krem,ncrepetitions))
            packarray_B!(Bpacked_krem, Bpmat_krem)
            for mo in 0:Miter-1
                # pack mc x kc block of A
                # Apacked_krem = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,1,false}(ptrL2, (Krem,mcrepetitions), (Krem*mᵣW,))
                # Apmat_krem = PtrArray{Tuple{mᵣW,-1,-1},Ta,3}(gesp(Aptr, (mo*mreps_per_iter, 0)), (mcrepetitions,Krem))
                Apacked_krem, Apmat_krem = Apack(Aptr, ptrL2, Val(mᵣW), Tc, mcrepetitions, Krem, mo, 0)
                # packarray_A!(Apacked_krem, Apmat_krem)
                Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noff))
                Cx = first(Cptr.strides)
                Cpmat = PtrArray{Tuple{mᵣW,-1,nᵣ,-1},Tc,4,Tuple{1,mᵣW,-1,-1},2,2,false}(Cptr_off, (mcrepetitions,ncrepetitions), (Cx,Cx*nᵣ))
                packaloopmul!(Cpmat, Apacked_krem, Apmat_krem, Bpacked_krem, α, β)
            end
            # Mrem
            if Mrem > 0
                Apacked_mrem_krem = PtrMatrix(ptrL2, Mrem, Krem, VectorizationBase.align(Mrem, Tc))
                Apmat_mrem_krem = PtrMatrix(gesp(Aptr, (Miter*mreps_per_iter, 0)), Mrem, Krem)
                # copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
                Cpmat_mrem = PtrArray{Tuple{-1,nᵣ,-1},Tc,3}(gesp(Cptr, (Miter*mreps_per_iter, Noff)), (Mrem, ncrepetitions))
                packaloopmul!(Cpmat_mrem, Apacked_mrem_krem, Apmat_mrem_krem, Bpacked_krem, α, β)
            end
            k = Krem
            for ko in 1:Kiter
                # pack kc x nc block of B
                Bpacked = PtrArray{Tuple{nᵣ,-1,-1},Tc,3,Tuple{1,nᵣ,-1},2,1,false}(ptrL3, (kreps_per_iter,ncrepetitions), (nᵣ * kreps_per_iter,))
                Bpmat = PtrArray{Tuple{-1,nᵣ,-1},Tb,3}(gesp(Bptr, (k, Noff)), (kreps_per_iter,ncrepetitions))
                packarray_B!(Bpacked, Bpmat)
                for mo in 0:Miter-1
                    # pack mc x kc block of A
                    Apacked, Apmat = Apack(Aptr, ptrL2, Val(mᵣW), Tc, mcrepetitions, kreps_per_iter, mo, k)
                    Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noff))
                    Cx = first(Cptr.strides)
                    Cpmat = PtrArray{Tuple{mᵣW,-1,nᵣ,-1},Tc,4,Tuple{1,mᵣW,-1,-1},2,2,false}(Cptr_off, (mcrepetitions,ncrepetitions), (Cx,Cx*nᵣ))
                    packaloopmul!(Cpmat, Apacked, Apmat, Bpacked, α, Val(1))
                end
                # Mrem
                if Mrem > 0
                    Apacked_mrem = PtrMatrix(ptrL2, Mrem, kreps_per_iter, VectorizationBase.align(Mrem,Tc))
                    Apmat_mrem = PtrMatrix(gesp(Aptr, (Miter*mreps_per_iter, k)), Mrem, kreps_per_iter)
                    # copyto!(Apacked_mrem, Apmat_mrem)
                    Cpmat_mrem = PtrArray{Tuple{-1,nᵣ,-1},Tc,3}(gesp(Cptr, (Miter*mreps_per_iter, Noff)), (Mrem, ncrepetitions))
                    packaloopmul!(Cpmat_mrem, Apacked_mrem, Apmat_mrem, Bpacked, α, Val(1))
                end
                k += kreps_per_iter
            end
            Noff += nreps_per_iter
        end
        # Nrem
        if Nrem > 0
            # calcnrrem = false
            # Krem
            # pack kc x nc block of B
            ncreprem, ncrepremrem = divrem_fast(Nrem, nᵣ)
            ncrepremc = ncreprem + !(iszero(ncrepremrem))
            lastBstride = nᵣ * Krem
            Bpacked_krem = PtrArray{Tuple{nᵣ,-1,-1},Tc,3,Tuple{1,nᵣ,-1},2,1,false}(ptrL3, (Krem,ncrepremc), (lastBstride,))
            Bpmat_krem = PtrArray{Tuple{-1,nᵣ,-1},Tb,3}(gesp(Bptr, (0, Noff)), (Krem,ncrepremc))  # Note the last axis extends past array's end!
            packarray_B!(Bpacked_krem, Bpmat_krem, ncrepremrem) 
            Noffrem = Noff + ncreprem*nᵣ
            Bpacked_krem_nrem = PtrMatrix{-1,-1,Tc,1,nᵣ,2,0,false}(gep(ptrL3, lastBstride * ncreprem), (ncrepremrem,Krem), tuple())
            for mo in 0:Miter-1
                # pack mc x kc block of A
                Apacked_krem, Apmat_krem = Apack(Aptr, ptrL2, Val(mᵣW), Tc, mcrepetitions, Krem, mo, 0)
                Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noff))
                Cx = first(Cptr.strides)
                Cpmat = PtrArray{Tuple{mᵣW,-1,nᵣ,-1},Tc,4,Tuple{1,mᵣW,-1,-1},2,2,false}(Cptr_off, (mcrepetitions,ncreprem), (Cx,Cx*nᵣ))
                packaloopmul!(Cpmat, Apacked_krem, Apmat_krem, Bpacked_krem, α, β)
                # @show Apacked_krem
                if ncrepremrem > 0
                    # if calcnrrem && ncrepremrem > 0
                    Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noffrem))
                    Cpmat_nrem = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,2,false}(Cptr_off, (mcrepetitions,ncrepremrem), (Cx,Cx*nᵣ))
                    loopmul!(Cpmat_nrem, Apacked_krem, Bpacked_krem_nrem, α, β)
                end
            end
            # Mrem
            if Mrem > 0
                Apacked_mrem_krem = PtrMatrix(ptrL2, Mrem, Krem, VectorizationBase.align(Mrem, Tc))
                Apmat_mrem_krem = PtrMatrix(gesp(Aptr, (Miter*mreps_per_iter, 0)), Mrem, Krem)
                Cpmat_mrem = PtrArray{Tuple{-1,nᵣ,-1},Tc,3}(gesp(Cptr, (Miter*mreps_per_iter, Noff)), (Mrem, ncreprem))
                packaloopmul!(Cpmat_mrem, Apacked_mrem_krem, Apmat_mrem_krem, Bpacked_krem, α, β)

                if ncrepremrem > 0
                    # if calcnrrem && ncrepremrem > 0
                    Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mreps_per_iter, Noff + ncreprem * nᵣ)), Mrem, ncrepremrem)
                    loopmul!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem', α, β, (Mrem, Krem, ncrepremrem))
                end

            end
            k = Krem
            for ko in 1:Kiter
                # pack kc x nc block of B
                lastBstride = nᵣ * kreps_per_iter
                Bpacked = PtrArray{Tuple{nᵣ,-1,-1},Tc,3,Tuple{1,nᵣ,-1},2,1,false}(ptrL3, (kreps_per_iter,ncrepremc), (lastBstride,))
                Bpmat = PtrArray{Tuple{-1,nᵣ,-1},Tb,3}(gesp(Bptr, (k, Noff)), (kreps_per_iter,ncrepremc)) # Note the last axis extends past array's end!
                packarray_B!(Bpacked, Bpmat, ncrepremrem)
                Bpacked_nrem = PtrMatrix{-1,-1,Tc,1,nᵣ,2,0,false}(gep(ptrL3, lastBstride * ncreprem), (ncrepremrem,kreps_per_iter), tuple())
                for mo in 0:Miter-1
                    # pack mc x kc block of A
                    Apacked, Apmat = Apack(Aptr, ptrL2, Val(mᵣW), Tc, mcrepetitions, kreps_per_iter, mo, k)
                    Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noff))
                    Cx = first(Cptr.strides)
                    Cpmat = PtrArray{Tuple{mᵣW,-1,nᵣ,-1},Tc,4,Tuple{1,mᵣW,-1,-1},2,2,false}(Cptr_off, (mcrepetitions,ncreprem), (Cx,Cx*nᵣ))
                    packaloopmul!(Cpmat, Apacked, Apmat, Bpacked, α, Val(1))
                    # if calcnrrem && ncrepremrem > 0
                    if ncrepremrem > 0
                        Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noffrem))
                        Cpmat_nrem = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,2,false}(Cptr_off, (mcrepetitions,ncrepremrem), (Cx,Cx*nᵣ))
                        loopmul!(Cpmat_nrem, Apacked, Bpacked_nrem, α, Val(1))
                    end

                end
                # Mrem
                if Mrem > 0
                    Apacked_mrem = PtrMatrix(ptrL2, Mrem, kreps_per_iter, VectorizationBase.align(Mrem,Tc))
                    Apmat_mrem = PtrMatrix(gesp(Aptr, (Miter*mreps_per_iter, k)), Mrem, kreps_per_iter)
                    Cpmat_mrem = PtrArray{Tuple{-1,nᵣ,-1},Tc,3}(gesp(Cptr, (Miter*mreps_per_iter, Noff)), (Mrem, ncreprem))
                    packaloopmul!(Cpmat_mrem, Apacked_mrem, Apmat_mrem, Bpacked, α, Val(1))

                    # if calcnrrem && ncrepremrem > 0
                    if ncrepremrem > 0
                        Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mreps_per_iter, Noff + ncreprem * nᵣ)), Mrem, ncrepremrem)
                        loopmul!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem', α, Val(1), (Mrem, kreps_per_iter, ncrepremrem))
                    end

                end
                k += kreps_per_iter
            end
        end
    end # GC.@preserve
    C
 end # function 

@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractMatrix, B::AbstractMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractStrideMatrix, B::AbstractMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractStrideMatrix, B::AbstractMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractStrideMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractStrideMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)

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
    sp + align(sizeof(T)*L), mv
end
@inline function Base.:*(a::T, B::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
    mv = FixedSizeArray{S,T,N,X,L}(undef)
    @avx for i ∈ eachindex(B)
        mv[i] = a * B[i]
    end
        # mv
    ConstantFixedSizeArray(mv)
end

# @inline function nmul!(
#     D::AbstractMatrix{T},
#     A′::LinearAlgebra.Adjoint{T,<:AbstractMatrix{T}},
#     X::AbstractMatrix{T}
# ) where {T <: BLAS.BlasFloat}
#     BLAS.gemm!('T','N',-one(T),A′.parent,X,zero(T),D)
# end


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





# @static if Base.libllvm_version ≥ v"10.0.0"
#     function llvmmul!(
#         C::AbstractMutableFixedSizeMatrix{M,N,T,1,XC},
#         A::AbstractMutableFixedSizeMatrix{M,K,T,1,XA},
#         B::AbstractMutableFixedSizeMatrix{K,N,T,1,XB}
#     ) where {M,K,N,T,XC,XA,XB}
#         vA = SIMDPirates.vcolumnwiseload(pointer(A), XA, Val{M}(), Val{K}())
#         vB = SIMDPirates.vcolumnwiseload(pointer(B), XB, Val{K}(), Val{N}())
#         vC = SIMDPirates.vmatmul(vA, vB, Val{M}(), Val{K}(), Val{N}())
#         vcolumnwisestore!(pointer(C), vC, XC, Val{M}(), Val{N}())
#         C
#     end
# end



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
          C[m,n] += Cₘₙ
      end)
    );
    order = LoopVectorization.choose_order(GEMMLOOPSET)
    mr = order[5]
    nr = last(order)
    @eval const mᵣ = $mr
    @eval const nᵣ = $nr
    # for T ∈ [Int16, Int32, Int64, Float32, Float64]
    # end

    
end

function loopmul!(C, A, B)
    M, K, N = matmul_sizes(C, A, B)
    @avx for n ∈ 1:N, m ∈ 1:M
        Cₘₙ = zero(eltype(C))
        for k ∈ 1:K
            Cₘₙ += A[m,k] * B[k,n]
            dummy1 = prefetch0(A, m, k + 3)
        end
        C[m,n] = Cₘₙ
    end
    C
end
function loopmuladd!(
    C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix
)
    M, K, N = matmul_sizes(C, A, B)
    @avx for n ∈ 1:N, m ∈ 1:M
        Cₘₙ = zero(eltype(C))
        for k ∈ 1:K
            Cₘₙ += A[m,k] * B[k,n]
            dummy1 = prefetch0(A, m, k + 3)
            # dummy1 = prefetch0(A, m + mᵣ, k)
            # dummy2 = prefetch0(B, k, n, 0, nᵣ)
        end
        C[m,n] += Cₘₙ
    end
    nothing
end

function Base.copyto!(B::AbstractStrideArray{S,T,N}, A::AbstractStrideArray{S,T,N}) where {S,T,N}
    @avx for I ∈ CartesianIndices(B)
        B[I] = A[I]
    end
end
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

function matmul_params(::Type{Tc}, ::Type{Ta}, ::Type{Tb}) where {Tc, Ta, Tb}
    # L₁, L₂, L₃ = cache_sizes()
    W = VectorizationBase.pick_vector_width(promote_type(Tc, Ta, Tb))
    # We aim for using roughly half of the L1, L2, and L3 caches
    # kc = 4(L₁ ÷ (12nᵣ * sizeof(Tb)))
    kc = 12(L₁ ÷ (32nᵣ * sizeof(Tb)))
    # kc = L₁ ÷ (2nᵣ * sizeof(Tb))
    # kc = VectorizationBase.prevpow2( L₁ ÷ (2nᵣ * sizeof(Tb)) )
    log2kc = VectorizationBase.intlog2(kc)
    # mcrep =  3L₂ ÷ (4kc * sizeof(Ta) * mᵣ * W)
    mcrep =  L₂ ÷ (2kc * sizeof(Ta) * mᵣ * W)
    # mcrep =  5L₂ ÷ (12kc * sizeof(Ta) * mᵣ * W)
    # mcrep = VectorizationBase.prevpow2( L₂ ÷ (2kc * sizeof(Ta) * mᵣ * W) )
    # ncrep = 7L₃ ÷ (8kc * sizeof(Tb) * nᵣ)
    ncrep = L₃ ÷ (kc * sizeof(Tb) * nᵣ)
    # ncrep = 3L₃ ÷ (4kc * sizeof(Tb) * nᵣ)
    # ncrep = L₃ ÷ (2kc * sizeof(Tb) * nᵣ)
    mc = mcrep * mᵣ * W
    nc = ncrep * nᵣ
    mc, kc, nc, log2kc
end
@generated function matmul_params_val(::Type{Tc}, ::Type{Ta}, ::Type{Tb}) where {Tc, Ta, Tb}
    mc, kc, nc, log2kc = matmul_params(Tc, Ta, Tb)
    Val(mc), Val(kc), Val(nc), Val(log2kc)
end

# function jmul!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, ::Val{mc}, ::Val{kc}, ::Val{nc}, ::Val{log2kc}) where {Tc, Ta, Tb, mc, kc, nc, log2kc}
#     M, K, N = matmul_sizes(C, A, B)
#     Niter, Nrem = divrem(N, nc)
#     # Kiter, Krem = divrem(K, $kc)
#     Miter, Mrem = divrem(M, mc)
#     (iszero(Miter) && iszero(Niter)) && return loopmul!(C, A, B)
#     Km1 = K - 1
#     Kiter = Km1 >>> log2kc
#     Krem = (Km1 & (kc - 1)) + 1 # Krem guaranteed to be > 0
#     resize_Acache!(Ta, Val{mc}(), Val{kc}(), Miter, Kiter)
#     # fill!(C, zero(Tc))
#     # Cptr = stridedpointer(C)
#     Aptr = stridedpointer(A)
#     Bptr = stridedpointer(B)
#     Cptr = stridedpointer(C)
#     # basea = pointer(ACACHE)
#     # divfactor = sizeof(Ta) * mc * kc
#     GC.@preserve C A B ACACHE BCACHE begin
#         # GC.@preserve C A B LCACHE begin
#         for no in 0:Niter-1
#             # Krem
#             # pack kc x nc block of B
#             # Base.Threads.@spawn begin
#             begin
#                 Bpacked_krem = PtrMatrix{-1,nc,Tb,kc}(threadlocal_L3CACHE_pointer(Tb), Krem)
#                 Bpmat_krem = PtrMatrix{-1,nc}(gesp(Bptr, (0, no*nc)), Krem)
#                 copyto!(Bpacked_krem, Bpmat_krem)
#                 # Bpacked_krem = PtrMatrix{-1,nc}(gesp(Bptr, (Kiter*kc, no*nc)), Krem)
#                 # Base.Threads.@sync begin
#                 for mo in 0:Miter-1
#                     # pack mc x kc block of A
#                     # Base.Threads.@spawn begin
#                     begin
#                         Apacked_krem = PtrMatrix{mc,-1,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, mo, 0), Krem)
#                         # Apacked_krem = PtrMatrix{mc,-1,Ta,mc}(gep(A_pointer(Ta), mo*mc), Krem)
#                         # @show reinterpret(Int,pointer(Apacked_krem) - basea) / divfactor
#                         if iszero(no)
#                             Apmat_krem = PtrMatrix{mc,-1}(gesp(Aptr, (mo*mc, 0)), Krem)
#                             copyto!(Apacked_krem, Apmat_krem)
#                         end
#                         Cpmat = PtrMatrix{mc,nc}(gesp(Cptr, (mo*mc, no*nc)))
#                         loopmul!(Cpmat, Apacked_krem, Bpacked_krem)
#                     end
#                 end
#                 # Mrem
#                 if Mrem > 0
#                     # Base.Threads.@spawn begin
#                     begin
#                         Apacked_mrem_krem = PtrMatrix{-1,-1,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, Miter, 0), Mrem, Krem)
#                         # Apacked_mrem_krem = PtrMatrix{-1,-1,Ta,mc}(gep(A_pointer(Ta), Miter*mc), Mrem, Krem)
#                         # @show reinterpret(Int,pointer(Apacked_mrem_krem) - basea) / divfactor
#                         if iszero(no)
#                             Apmat_mrem_krem = PtrMatrix{-1,-1}(gesp(Aptr, (Miter*mc, 0)), Mrem, Krem)
#                             copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
#                         end
#                         Cpmat_mrem = PtrMatrix{-1,nc}(gesp(Cptr, (Miter*mc, no*nc)), Mrem)
#                         loopmul!(Cpmat_mrem, Apacked_mrem_krem, Bpacked_krem)
#                     end
#                 end
#                 # end # sync
#                 k = Krem
#                 for ko in 1:Kiter
#                     # pack kc x nc block of B
#                     Bpacked = PtrMatrix{kc,nc,Tb,kc}(threadlocal_L3CACHE_pointer(Tb))
#                     Bpmat = PtrMatrix{kc,nc}(gesp(Bptr, (k, no*nc)))
#                     copyto!(Bpacked, Bpmat)
#                     # Bprefetch = PtrMatrix{kc,nc}(gesp(Bptr, ((ko+1)*kc, no*nc)))
#                     # copyto!(Bpacked, Bpmat, Bprefetch)
#                     # Base.Threads.@sync begin
#                     for mo in 0:Miter-1
#                         # Base.Threads.@spawn begin
#                         begin
#                             # pack mc x kc block of A
#                             # Apmat = PtrMatrix{mc,kc}(gesp(Aptr, (mo*mc, ko*kc)))
#                             Apacked = PtrMatrix{mc,kc,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, mo, ko))
#                             # Apacked = PtrMatrix{mc,kc,Ta,mc}(gep(A_pointer(Ta), mo*mc + k * (Miter+1)*mc))
#                         # @show reinterpret(Int,pointer(Apacked) - basea) / divfactor
#                             if iszero(no)
#                                 Apmat = PtrMatrix{mc,kc}(gesp(Aptr, (mo*mc, k)))
#                                 copyto!(Apacked, Apmat)
#                             end
#                             # Aprefetch = PtrMatrix{mc,kc}(gesp(Aptr, ((mo+1)*mc, ko*kc)))
#                             # copyto_prefetch3!(Apacked, Apmat, Aprefetch)
#                             Cpmat = PtrMatrix{mc,nc}(gesp(Cptr, (mo*mc, no*nc)))
#                             loopmuladd!(Cpmat, Apacked, Bpacked)
#                         end
#                     end
#                     # Mrem
#                     if Mrem > 0
#                         # Base.Threads.@spawn begin
#                         begin
#                             Apacked_mrem = PtrMatrix{-1,kc,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, Miter, ko), Mrem)
#                             # Apacked_mrem = PtrMatrix{-1,kc,Ta,mc}(gep(A_pointer(Ta), Miter*mc + k * (Miter+1)*mc), Mrem)
#                             # @show reinterpret(Int,pointer(Apacked_mrem) - basea) / divfactor
#                             if iszero(no)
#                                 Apmat_mrem = PtrMatrix{-1,kc}(gesp(Aptr, (Miter*mc, k)), Mrem)
#                                 copyto!(Apacked_mrem, Apmat_mrem)
#                             end
#                             Cpmat_mrem = PtrMatrix{-1,nc}(gesp(Cptr, (Miter*mc, no*nc)), Mrem)
#                             loopmuladd!(Cpmat_mrem, Apacked_mrem, Bpacked)
#                         end
#                     end
#                     # end # sync
#                     k += kc
#                 end
#             end # spawn
#         end # nloop
#         # Nrem
#         if Nrem > 0
#             # Base.Threads.@spawn begin
#             # begin
#             # Krem
#             # pack kc x nc block of B
#             Bpacked_krem_nrem = PtrMatrix{-1,-1,Tb,kc}(threadlocal_L3CACHE_pointer(Tb), Krem, Nrem)
#             Bpmat_krem_nrem = PtrMatrix{-1,-1}(gesp(Bptr, (0, Niter*nc)), Krem, Nrem)
#             copyto!(Bpacked_krem_nrem, Bpmat_krem_nrem)
#             # Bpacked_krem_nrem = PtrMatrix{-1,-1}(gesp(Bptr, (Kiter*kc, Niter*nc)), Krem, Nrem)
#             # Base.Threads.@sync begin
#             for mo in 0:Miter-1
#                 # Base.Threads.@spawn begin
#                 begin
#                     # pack mc x kc block of A
#                     Apacked_krem = PtrMatrix{mc,-1,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, mo, 0), Krem)
#                     # Apacked_krem = PtrMatrix{mc,-1,Ta,mc}(gep(A_pointer(Ta), mo*mc), Krem)
#                     # @show reinterpret(Int,pointer(Apacked_krem) - basea) / divfactor
#                     if iszero(Niter)
#                         Apmat_krem = PtrMatrix{mc,-1}(gesp(Aptr, (mo*mc, 0)), Krem)
#                         copyto!(Apacked_krem, Apmat_krem)
#                     end
#                     Cpmat_nrem = PtrMatrix{mc,-1}(gesp(Cptr, (mo*mc, Niter*nc)), Nrem)
#                     loopmul!(Cpmat_nrem, Apacked_krem, Bpacked_krem_nrem)
#                 end
#             end
#             # Mrem
#             if Mrem > 0
#                 # Base.Threads.@spawn begin
#                 begin
#                     Apacked_mrem_krem = PtrMatrix{-1,-1,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, Miter, 0), Mrem, Krem)
#                     # Apacked_mrem_krem = PtrMatrix{-1,-1,Ta,mc}(gep(A_pointer(Ta), Miter*mc), Mrem, Krem)
#                     # @show reinterpret(Int,pointer(Apacked_mrem_krem) - basea) / divfactor
#                     if iszero(Niter)
#                         Apmat_mrem_krem = PtrMatrix{-1,-1}(gesp(Aptr, (Miter*mc, 0)), Mrem, Krem)
#                         copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
#                     end
#                     Cpmat_mrem_nrem = PtrMatrix{-1,-1}(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
#                     loopmul!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem)
#                 end
#             end
#             # end # sync
#             k = Krem
#             for ko in 1:Kiter
#                 # pack kc x nc block of B
#                 Bpacked_nrem = PtrMatrix{kc,-1,Tb,kc}(threadlocal_L3CACHE_pointer(Tb), Nrem)
#                 Bpmat_nrem = PtrMatrix{kc,-1}(gesp(Bptr, (k, Niter*nc)), Nrem)
#                 copyto!(Bpacked_nrem, Bpmat_nrem)
#                 # Bpacked_nrem = PtrMatrix{kc,-1}(gesp(Bptr, (ko*kc, Niter*nc)), Nrem)
#                 # Base.Threads.@sync begin
#                 for mo in 0:Miter-1
#                     # Base.Threads.@spawn begin
#                     begin
#                         # pack mc x kc block of A
#                         Apacked = PtrMatrix{mc,kc,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, mo, ko))
#                         # Apacked = PtrMatrix{mc,kc,Ta,mc}(gep(A_pointer(Ta), mo*mc + k * (Miter+1)*mc))
#                         # @show reinterpret(Int,pointer(Apacked) - basea) / divfactor
#                         if iszero(Niter)
#                             Apmat = PtrMatrix{mc,kc}(gesp(Aptr, (mo*mc, k)))
#                             copyto!(Apacked, Apmat)
#                         end
#                         Cpmat_nrem = PtrMatrix{mc,-1}(gesp(Cptr, (mo*mc, Niter*nc)), Nrem)
#                         loopmuladd!(Cpmat_nrem, Apacked, Bpacked_nrem)
#                     end
#                 end
#                 # Mrem
#                 if Mrem > 0
#                     # Base.Threads.@spawn begin
#                     begin
#                         Apacked_mrem = PtrMatrix{-1,kc,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, Miter, ko), Mrem)
#                         # Apacked_mrem = PtrMatrix{-1,kc,Ta,mc}(gep(A_pointer(Ta), Miter*mc + k * (Miter+1)*mc), Mrem)
#                         # @show reinterpret(Int,pointer(Apacked_mrem_krem) - basea) / divfactor
#                         if iszero(Niter)
#                             Apmat_mrem = PtrMatrix{-1,kc}(gesp(Aptr, (Miter*mc, k)), Mrem)
#                             copyto!(Apacked_mrem, Apmat_mrem)
#                         end
#                         Cpmat_mrem_nrem = PtrMatrix{-1,-1}(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
#                         loopmuladd!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem)
#                     end
#                 end
#                 # end # sync
#                 k += kc
#             end
#         end
#     end # GC.@preserve
#     C
# end
# function jmult!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, ::Val{mc}, ::Val{kc}, ::Val{nc}, ::Val{log2kc}) where {Tc, Ta, Tb, mc, kc, nc, log2kc}
#     M, K, N = matmul_sizes(C, A, B)
#     Niter, Nrem = divrem(N, nc)
#     # Kiter, Krem = divrem(K, $kc)
#     Miter, Mrem = divrem(M, mc)
#     (iszero(Miter) && iszero(Niter)) && return loopmul!(C, A, B)
#     Km1 = K - 1
#     Kiter = Km1 >>> log2kc
#     Krem = (Km1 & (kc - 1)) + 1 # Krem guaranteed to be > 0
#     resize_Acache!(Ta, Val{mc}(), Val{kc}(), Miter, Kiter)
#     # fill!(C, zero(Tc))
#     # Cptr = stridedpointer(C)
#     Aptr = stridedpointer(A)
#     Bptr = stridedpointer(B)
#     Cptr = stridedpointer(C)
#     Base.Threads.@sync GC.@preserve C A B ACACHE BCACHE begin
#     # GC.@preserve C A B LCACHE begin
#         for no in 0:Niter-1
#             # Krem
#             # pack kc x nc block of B
#             Base.Threads.@spawn begin
#             # begin
#                 Bpacked_krem = PtrMatrix{-1,nc,Tb,kc}(threadlocal_L3CACHE_pointer(Tb), Krem)
#                 Bpmat_krem = PtrMatrix{-1,nc}(gesp(Bptr, (0, no*nc)), Krem)
#                 copyto!(Bpacked_krem, Bpmat_krem)
#                 # Bpacked_krem = PtrMatrix{-1,nc}(gesp(Bptr, (Kiter*kc, no*nc)), Krem)
#                 Base.Threads.@sync begin for mo in 0:Miter-1
#                     # pack mc x kc block of A
#                     Base.Threads.@spawn begin
#                     # begin
#                         Apacked_krem = PtrMatrix{mc,-1,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, mo, 0), Krem)
#                         if iszero(no)
#                             Apmat_krem = PtrMatrix{mc,-1}(gesp(Aptr, (mo*mc, 0)), Krem)
#                             copyto!(Apacked_krem, Apmat_krem)
#                         end
#                         Cpmat = PtrMatrix{mc,nc}(gesp(Cptr, (mo*mc, no*nc)))
#                         loopmul!(Cpmat, Apacked_krem, Bpacked_krem)
#                     end
#                 end
#                 # Mrem
#                 if Mrem > 0
#                     Base.Threads.@spawn begin
#                     # begin
#                         Apacked_mrem_krem = PtrMatrix{-1,-1,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, Miter, 0), Mrem, Krem)
#                         if iszero(no)
#                             Apmat_mrem_krem = PtrMatrix{-1,-1}(gesp(Aptr, (Miter*mc, 0)), Mrem, Krem)
#                             copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
#                         end
#                         Cpmat_mrem = PtrMatrix{-1,nc}(gesp(Cptr, (Miter*mc, no*nc)), Mrem)
#                         loopmul!(Cpmat_mrem, Apacked_mrem_krem, Bpacked_krem)
#                     end
#                 end
#                 end # sync
#                 k = Krem
#                 for ko in 1:Kiter
#                     # pack kc x nc block of B
#                     Bpacked = PtrMatrix{kc,nc,Tb,kc}(threadlocal_L3CACHE_pointer(Tb))
#                     Bpmat = PtrMatrix{kc,nc}(gesp(Bptr, (k, no*nc)))
#                     copyto!(Bpacked, Bpmat)
#                     # Bprefetch = PtrMatrix{kc,nc}(gesp(Bptr, ((ko+1)*kc, no*nc)))
#                     # copyto!(Bpacked, Bpmat, Bprefetch)
#                     Base.Threads.@sync begin
#                     for mo in 0:Miter-1
#                         Base.Threads.@spawn begin
#                         # begin
#                             # pack mc x kc block of A
#                             # Apmat = PtrMatrix{mc,kc}(gesp(Aptr, (mo*mc, ko*kc)))
#                             Apacked = PtrMatrix{mc,kc,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, mo, ko))
#                             if iszero(no)
#                                 Apmat = PtrMatrix{mc,kc}(gesp(Aptr, (mo*mc, k)))
#                                 copyto!(Apacked, Apmat)
#                             end
#                             # Aprefetch = PtrMatrix{mc,kc}(gesp(Aptr, ((mo+1)*mc, ko*kc)))
#                             # copyto_prefetch3!(Apacked, Apmat, Aprefetch)
#                             Cpmat = PtrMatrix{mc,nc}(gesp(Cptr, (mo*mc, no*nc)))
#                             loopmuladd!(Cpmat, Apacked, Bpacked)
#                         end
#                     end
#                     # Mrem
#                     if Mrem > 0
#                         Base.Threads.@spawn begin
#                         # begin
#                             Apacked_mrem = PtrMatrix{-1,kc,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, Miter, ko), Mrem)
#                             if iszero(no)
#                                 Apmat_mrem = PtrMatrix{-1,kc}(gesp(Aptr, (Miter*mc, k)), Mrem)
#                                 copyto!(Apacked_mrem, Apmat_mrem)
#                             end
#                             Cpmat_mrem = PtrMatrix{-1,nc}(gesp(Cptr, (Miter*mc, no*nc)), Mrem)
#                             loopmuladd!(Cpmat_mrem, Apacked_mrem, Bpacked)
#                         end
#                     end
#                     end
#                     k += kc
#                 end
#             end # spawn
#         end # nloop
#         # Nrem
#         if Nrem > 0
#             Base.Threads.@spawn begin
#             # begin
#                 # Krem
#                 # pack kc x nc block of B
#                 Bpacked_krem_nrem = PtrMatrix{-1,-1,Tb,kc}(threadlocal_L3CACHE_pointer(Tb), Krem, Nrem)
#                 Bpmat_krem_nrem = PtrMatrix{-1,-1}(gesp(Bptr, (0, Niter*nc)), Krem, Nrem)
#                 copyto!(Bpacked_krem_nrem, Bpmat_krem_nrem)
#                 # Bpacked_krem_nrem = PtrMatrix{-1,-1}(gesp(Bptr, (Kiter*kc, Niter*nc)), Krem, Nrem)
#                 Base.Threads.@sync begin for mo in 0:Miter-1
#                     Base.Threads.@spawn begin
#                     # begin
#                         # pack mc x kc block of A
#                         Apacked_krem = PtrMatrix{mc,-1,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, mo, 0), Krem)
#                         if iszero(Niter)
#                             Apmat_krem = PtrMatrix{mc,-1}(gesp(Aptr, (mo*mc, 0)), Krem)
#                             copyto!(Apacked_krem, Apmat_krem)
#                         end
#                         Cpmat_nrem = PtrMatrix{mc,-1}(gesp(Cptr, (mo*mc, Niter*nc)), Nrem)
#                         loopmul!(Cpmat_nrem, Apacked_krem, Bpacked_krem_nrem)
#                     end
#                 end
#                 # Mrem
#                 if Mrem > 0
#                     Base.Threads.@spawn begin
#                     # begin
#                         Apacked_mrem_krem = PtrMatrix{-1,-1,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, Miter, 0), Mrem, Krem)
#                         if iszero(Niter)
#                             Apmat_mrem_krem = PtrMatrix{-1,-1}(gesp(Aptr, (Miter*mc, 0)), Mrem, Krem)
#                             copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
#                         end
#                         Cpmat_mrem_nrem = PtrMatrix{-1,-1}(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
#                         loopmul!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem)
#                     end
#                 end
#                 end # sync
#                 k = Krem
#                 for ko in 1:Kiter
#                     # pack kc x nc block of B
#                     Bpacked_nrem = PtrMatrix{kc,-1,Tb,kc}(threadlocal_L3CACHE_pointer(Tb), Nrem)
#                     Bpmat_nrem = PtrMatrix{kc,-1}(gesp(Bptr, (k, Niter*nc)), Nrem)
#                     copyto!(Bpacked_nrem, Bpmat_nrem)
#                     # Bpacked_nrem = PtrMatrix{kc,-1}(gesp(Bptr, (ko*kc, Niter*nc)), Nrem)
#                     Base.Threads.@sync begin
#                         for mo in 0:Miter-1
#                             Base.Threads.@spawn begin
#                                 # begin
#                                 # pack mc x kc block of A
#                                 Apacked = PtrMatrix{mc,kc,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, mo, ko))
#                                 if iszero(Niter)
#                                     Apmat = PtrMatrix{mc,kc}(gesp(Aptr, (mo*mc, k)))
#                                     copyto!(Apacked, Apmat)
#                                 end
#                                 Cpmat_nrem = PtrMatrix{mc,-1}(gesp(Cptr, (mo*mc, Niter*nc)), Nrem)
#                                 loopmuladd!(Cpmat_nrem, Apacked, Bpacked_nrem)
#                             end
#                         end
#                         # Mrem
#                         if Mrem > 0
#                             Base.Threads.@spawn begin
#                                 # begin
#                                 Apacked_mrem = PtrMatrix{-1,kc,Ta,mc}(A_pointer(Ta, Val{mc}(), Val{kc}(), Miter, Miter, ko), Mrem)
#                                 if iszero(Niter)
#                                     Apmat_mrem = PtrMatrix{-1,kc}(gesp(Aptr, (Miter*mc, k)), Mrem)
#                                     copyto!(Apacked_mrem, Apmat_mrem)
#                                 end
#                                 Cpmat_mrem_nrem = PtrMatrix{-1,-1}(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
#                                 loopmuladd!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem)
#                             end
#                         end
#                     end # sync
#                     k += kc
#                 end
#             end
#         end
#     end # GC.@preserve
#     C
# end


function pack_B_krem(Bptr, ::Val{kc}, ::Val{nc}, ::Type{Tb}, Krem::Int, no::Int) where {nc, Tb, kc}
    Bpacked_krem = PtrMatrix{-1,nc,Tb,kc}(threadlocal_L3CACHE_pointer(Tb), Krem)
    Bpmat_krem = PtrMatrix{-1,nc}(gesp(Bptr, (0, no*nc)), Krem)
    copyto!(Bpacked_krem, Bpmat_krem)
    Bpacked_krem
end
function pack_B(Bptr, ::Val{kc}, ::Val{nc}, ::Type{Tb}, k::Int, no::Int) where {nc, Tb, kc}
    Bpacked = PtrMatrix{kc,nc,Tb,kc}(threadlocal_L3CACHE_pointer(Tb))
    Bpmat = PtrMatrix{kc,nc}(gesp(Bptr, (k, no*nc)))
    copyto!(Bpacked, Bpmat)
    Bpacked
end
function pack_B_krem_nrem(Bptr, ::Val{kc}, ::Val{nc}, ::Type{Tb}, Krem::Int, Nrem::Int, Niter::Int) where {nc, Tb, kc}
    Bpacked_krem_nrem = PtrMatrix{-1,-1,Tb,kc}(threadlocal_L3CACHE_pointer(Tb), Krem, Nrem)
    Bpmat_krem_nrem = PtrMatrix{-1,-1}(gesp(Bptr, (0, Niter*nc)), Krem, Nrem)
    copyto!(Bpacked_krem_nrem, Bpmat_krem_nrem)
    Bpacked_krem_nrem
end
function pack_B_nrem(Bptr, ::Val{kc}, ::Val{nc}, ::Type{Tb}, k::Int, Nrem::Int, Niter::Int) where {nc, Tb, kc}
    Bpacked_nrem = PtrMatrix{kc,-1,Tb,kc}(threadlocal_L3CACHE_pointer(Tb), Nrem)
    Bpmat_nrem = PtrMatrix{kc,-1}(gesp(Bptr, (k, Niter*nc)), Nrem)
    copyto!(Bpacked_nrem, Bpmat_nrem)
    Bpacked_nrem
end
function pack_A_krem(Aptr, ::Val{mc}, ::Type{Ta}, mo::Int, Krem::Int) where {mc, Ta}
    Apacked_krem = PtrMatrix{mc,-1,Ta,mc}(threadlocal_L2CACHE_pointer(Ta), Krem)
    Apmat_krem = PtrMatrix{mc,-1}(gesp(Aptr, (mo*mc, 0)), Krem)
    copyto!(Apacked_krem, Apmat_krem)
    Apacked_krem
end
function pack_A_mrem(Aptr, ::Val{mc}, ::Val{kc}, ::Type{Ta}, Mrem::Int, k::Int, Miter::Int) where {mc, kc, Ta}
    Apacked_mrem = PtrMatrix{-1,kc,Ta,mc}(threadlocal_L2CACHE_pointer(Ta), Mrem)
    Apmat_mrem = PtrMatrix{-1,kc}(gesp(Aptr, (Miter*mc, k)), Mrem)
    copyto!(Apacked_mrem, Apmat_mrem)
    Apacked_mrem
end
function pack_A_mrem_krem(Aptr, ::Val{mc}, ::Type{Ta}, Mrem::Int, Krem::Int, Miter::Int) where {mc, Ta}
    Apacked_mrem_krem = PtrMatrix{-1,-1,Ta,mc}(threadlocal_L2CACHE_pointer(Ta), Mrem, Krem)
    Apmat_mrem_krem = PtrMatrix{-1,-1}(gesp(Aptr, (Miter*mc, 0)), Mrem, Krem)
    copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
    Apacked_mrem_krem
end
function pack_A(Aptr, ::Val{mc}, ::Val{kc}, ::Type{Ta}, mo::Int, k::Int) where {mc, kc, Ta}
    Apacked = PtrMatrix{mc,kc,Ta,mc}(threadlocal_L2CACHE_pointer(Ta))
    Apmat = PtrMatrix{mc,kc}(gesp(Aptr, (mo*mc, k)))
    # @show (reinterpret(Int, pointer(Apmat)) - reinterpret(Int,pointer(Aptr))) >>> 3
    copyto!(Apacked, Apmat)
    Apacked
end


function jmul!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, ::Val{mc}, ::Val{kc}, ::Val{nc}, ::Val{log2kc}) where {Tc, Ta, Tb, mc, kc, nc, log2kc}
    M, K, N = matmul_sizes(C, A, B)
    Niter, Nrem = divrem(N, nc)
    # Kiter, Krem = divrem(K, kc)
    Miter, Mrem = divrem(M, mc)
    (iszero(Miter) && iszero(Niter)) && return loopmul!(C, A, B)
    Km1 = K - 1
    if (1 << log2kc) == kc
        Kiter = Km1 >>> log2kc
        Krem = (Km1 & (kc - 1)) # Krem guaranteed to be > 0
    else
        Kiter, Krem = divrem(Km1, kc)
    end
    Krem += 1
    # fill!(C, zero(Tc))
    # Cptr = stridedpointer(C)
    Aptr = stridedpointer(A)
    Bptr = stridedpointer(B)
    Cptr = stridedpointer(C)
    GC.@preserve C A B LCACHE begin
        for no in 0:Niter-1
            # Krem
            # pack kc x nc block of B
            Bpacked_krem = pack_B_krem(Bptr, Val{kc}(), Val{nc}(), Tb, Krem, no)
            # Bpacked_krem = PtrMatrix{-1,nc}(gesp(Bptr, (Kiter*kc, no*nc)), Krem)
            for mo in 0:Miter-1
                # pack mc x kc block of A
                Apacked_krem = pack_A_krem(Aptr, Val{mc}(), Ta, mo, Krem)
                Cpmat = PtrMatrix{mc,nc}(gesp(Cptr, (mo*mc, no*nc)))
                loopmul!(Cpmat, Apacked_krem, Bpacked_krem)
            end
            # Mrem
            if Mrem > 0
                Apacked_mrem_krem = pack_A_mrem_krem(Aptr, Val{mc}(), Ta, Mrem, Krem, Miter)
                Cpmat_mrem = PtrMatrix{-1,nc}(gesp(Cptr, (Miter*mc, no*nc)), Mrem)
                loopmul!(Cpmat_mrem, Apacked_mrem_krem, Bpacked_krem)
            end
            k = Krem
            for ko in 1:Kiter
                # pack kc x nc block of B
                Bpacked = pack_B(Bptr, Val{kc}(), Val{nc}(), Tb, k, no)
                for mo in 0:Miter-1
                    # pack mc x kc block of A
                    Apacked = pack_A(Aptr, Val{mc}(), Val{kc}(), Ta, mo, k)
                    Cpmat = PtrMatrix{mc,nc}(gesp(Cptr, (mo*mc, no*nc)))
                    loopmuladd!(Cpmat, Apacked, Bpacked)
                end
                # Mrem
                if Mrem > 0
                    Apacked_mrem = pack_A_mrem(Aptr, Val{mc}(), Val{kc}(), Ta, Mrem, k, Miter)
                    Cpmat_mrem = PtrMatrix{-1,nc}(gesp(Cptr, (Miter*mc, no*nc)), Mrem)
                    loopmuladd!(Cpmat_mrem, Apacked_mrem, Bpacked)
                end
                k += kc
            end
        end
        # Nrem
        if Nrem > 0
            # Krem
            # pack kc x nc block of B
            Bpacked_krem_nrem = pack_B_krem_nrem(Bptr, Val{kc}(), Val{nc}(), Tb, Krem, Nrem, Niter)
            for mo in 0:Miter-1
                # pack mc x kc block of A
                Apacked_krem = pack_A_krem(Aptr, Val{mc}(), Ta, mo, Krem)
                Cpmat_nrem = PtrMatrix{mc,-1}(gesp(Cptr, (mo*mc, Niter*nc)), Nrem)
                loopmul!(Cpmat_nrem, Apacked_krem, Bpacked_krem_nrem)
            end
            # Mrem
            if Mrem > 0
                Apacked_mrem_krem = pack_A_mrem_krem(Aptr, Val{mc}(), Ta, Mrem, Krem, Miter)
                Cpmat_mrem_nrem = PtrMatrix{-1,-1}(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
                loopmul!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem)
            end
            k = Krem
            for ko in 1:Kiter
                # pack kc x nc block of B
                Bpacked_nrem = pack_B_nrem(Bptr, Val{kc}(), Val{nc}(), Tb, k, Nrem, Niter)
                for mo in 0:Miter-1
                    # pack mc x kc block of A
                    Apacked = pack_A(Aptr, Val{mc}(), Val{kc}(), Ta, mo, k)
                    Cpmat_nrem = PtrMatrix{mc,-1}(gesp(Cptr, (mo*mc, Niter*nc)), Nrem)
                    loopmuladd!(Cpmat_nrem, Apacked, Bpacked_nrem)
                end
                # Mrem
                if Mrem > 0
                    Apacked_mrem = pack_A_mrem(Aptr, Val{mc}(), Val{kc}(), Ta, Mrem, k, Miter)
                    Cpmat_mrem_nrem = PtrMatrix{-1,-1}(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
                    loopmuladd!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem)
                end
                k += kc
            end
        end
    end # GC.@preserve
    C
end # function

function jmult!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, ::Val{mc}, ::Val{kc}, ::Val{nc}, ::Val{log2kc}) where {Tc, Ta, Tb, mc, kc, nc, log2kc}
    M, K, N = matmul_sizes(C, A, B)
    Niter, Nrem = divrem(N, nc)
    # Kiter, Krem = divrem(K, $kc)
    Miter, Mrem = divrem(M, mc)
    (iszero(Miter) && iszero(Niter)) && return loopmul!(C, A, B)
    Km1 = K - 1
    if (1 << log2kc) == kc
        Kiter = Km1 >>> log2kc
        Krem = (Km1 & (kc - 1)) # Krem guaranteed to be > 0
    else
        Kiter, Krem = divrem(Km1, kc)
    end
    Krem += 1
    # fill!(C, zero(Tc))
    # Cptr = stridedpointer(C)
    Aptr = stridedpointer(A)
    Bptr = stridedpointer(B)
    Cptr = stridedpointer(C)
    Base.Threads.@sync GC.@preserve C A B LCACHE begin
        for no in 0:Niter-1
            # Krem
            # pack kc x nc block of B
            # Base.Threads.@spawn begin
            begin
                Bpacked_krem = pack_B_krem(Bptr, Val{kc}(), Val{nc}(), Tb, Krem, no)
                Base.Threads.@sync begin
                    for mo in 0:Miter-1
                        # pack mc x kc block of A
                        Base.Threads.@spawn begin
                            # begin
                            Apacked_krem = pack_A_krem(Aptr, Val{mc}(), Ta, mo, Krem)
                            Cpmat = PtrMatrix{mc,nc}(gesp(Cptr, (mo*mc, no*nc)))
                            loopmul!(Cpmat, Apacked_krem, Bpacked_krem)
                        end
                    end
                    # Mrem
                    if Mrem > 0
                        Base.Threads.@spawn begin
                            # begin
                            Apacked_mrem_krem = pack_A_mrem_krem(Aptr, Val{mc}(), Ta, Mrem, Krem, Miter)
                            Cpmat_mrem = PtrMatrix{-1,nc}(gesp(Cptr, (Miter*mc, no*nc)), Mrem)
                            loopmul!(Cpmat_mrem, Apacked_mrem_krem, Bpacked_krem)
                        end
                    end
                end # sync
                k = Krem
                for ko in 1:Kiter
                    # pack kc x nc block of B
                    Bpacked = pack_B(Bptr, Val{kc}(), Val{nc}(), Tb, k, no)
                    # Bprefetch = PtrMatrix{kc,nc}(gesp(Bptr, ((ko+1)*kc, no*nc)))
                    # copyto!(Bpacked, Bpmat, Bprefetch)
                    Base.Threads.@sync begin
                        for mo in 0:Miter-1
                            Base.Threads.@spawn begin
                                # begin
                                Apacked = pack_A(Aptr, Val{mc}(), Val{kc}(), Ta, mo, k)
                                Cpmat = PtrMatrix{mc,nc}(gesp(Cptr, (mo*mc, no*nc)))
                                loopmuladd!(Cpmat, Apacked, Bpacked)
                            end
                        end
                        # Mrem
                        if Mrem > 0
                            Base.Threads.@spawn begin
                                # begin
                                Apacked_mrem = pack_A_mrem(Aptr, Val{mc}(), Val{kc}(), Ta, Mrem, k, Miter)
                                Cpmat_mrem = PtrMatrix{-1,nc}(gesp(Cptr, (Miter*mc, no*nc)), Mrem)
                                loopmuladd!(Cpmat_mrem, Apacked_mrem, Bpacked)
                            end
                        end
                    end # sync
                    k += kc
                end
            end # spawnp
        end # nloop
        # Nrem
        if Nrem > 0
            # Base.Threads.@spawn begin
            begin
                # Krem
                # pack kc x nc block of B
                Bpacked_krem_nrem = pack_B_krem_nrem(Bptr, Val{kc}(), Val{nc}(), Tb, Krem, Nrem, Niter)
                # Bpacked_krem_nrem = PtrMatrix{-1,-1}(gesp(Bptr, (Kiter*kc, Niter*nc)), Krem, Nrem)
                Base.Threads.@sync begin
                    for mo in 0:Miter-1
                        Base.Threads.@spawn begin
                            # begin
                            # pack mc x kc block of A
                            Apacked_krem = pack_A_krem(Aptr, Val{mc}(), Ta, mo, Krem)
                            Cpmat_nrem = PtrMatrix{mc,-1}(gesp(Cptr, (mo*mc, Niter*nc)), Nrem)
                            loopmul!(Cpmat_nrem, Apacked_krem, Bpacked_krem_nrem)
                        end
                    end
                    # Mrem
                    if Mrem > 0
                        Base.Threads.@spawn begin
                            # begin
                            Apacked_mrem_krem = pack_A_mrem_krem(Aptr, Val{mc}(), Ta, Mrem, Krem, Miter)
                            Cpmat_mrem_nrem = PtrMatrix{-1,-1}(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
                            loopmul!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem)
                        end
                    end
                end # sync
                k = Krem
                for ko in 1:Kiter
                    # pack kc x nc block of B
                    Bpacked_nrem = pack_B_nrem(Bptr, Val{kc}(), Val{nc}(), Tb, k, Nrem, Niter)
                    Base.Threads.@sync begin
                        for mo in 0:Miter-1
                            Base.Threads.@spawn begin
                                # begin
                                # pack mc x kc block of A
                                Apacked = pack_A(Aptr, Val{mc}(), Val{kc}(), Ta, mo, k)
                                Cpmat_nrem = PtrMatrix{mc,-1}(gesp(Cptr, (mo*mc, Niter*nc)), Nrem)
                                loopmuladd!(Cpmat_nrem, Apacked, Bpacked_nrem)
                            end
                        end
                        # Mrem
                        if Mrem > 0
                            Base.Threads.@spawn begin
                                # begin
                                Apacked_mrem = pack_A_mrem(Aptr, Val{mc}(), Val{kc}(), Ta, Mrem, k, Miter)
                                Cpmat_mrem_nrem = PtrMatrix{-1,-1}(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
                                loopmuladd!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem)
                            end
                        end
                    end # sync
                    k += kc
                end
            end
        end
    end # GC.@preserve, sync
    C
end


function jmul!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}) where {Tc, Ta, Tb}
    mc, kc, nc, log2kc = matmul_params_val(Tc, Ta, Tb)
    jmul!(C, A, B, mc, kc, nc, log2kc)
end
function jmult!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}) where {Tc, Ta, Tb}
    mc, kc, nc, log2kc = matmul_params_val(Tc, Ta, Tb)
    jmult!(C, A, B, mc, kc, nc, log2kc)
end





    # @generated function jmulto!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}) where {Tc, Ta, Tb}
    #     mc, kc, nc, log2kc = matmul_params(Tc, Ta, Tb)
    #     :(jmulto!(C, A, B, Val{$mc}(), Val{$kc}(), Val{$nc}(), Val{$log2kc}()))
    # end # function 

    # function jmulto!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, ::Val{mc}, ::Val{kc}, ::Val{nc}, ::Val{log2kc}) where {Tc, Ta, Tb, mc, kc, nc, log2kc}
    #     M, K, N = matmul_sizes(C, A, B)
    #     Niter, Nrem = divrem(N, nc)
    #     # Kiter, Krem = divrem(K, $kc)
    #     Miter, Mrem = divrem(M, mc)
    #     (iszero(Miter) && iszero(Niter)) && return loopmul!(C, A, B)
    #     Km1 = K - 1
    #     Kiter = Km1 >>> log2kc
    #     Krem = (Km1 & (kc - 1)) + 1 # Krem guaranteed to be > 0
    #     # fill!(C, zero(Tc))
    #     # Cptr = stridedpointer(C)
    #     Aptr = stridedpointer(A)
    #     Bptr = stridedpointer(B)
    #     Cptr = stridedpointer(C)
    #     Base.Threads.@sync GC.@preserve C A B LCACHE begin
    #     # GC.@preserve C A B LCACHE begin
    #         for no in 0:Niter-1
    #             # Krem
    #             # pack kc x nc block of B
#             Base.Threads.@spawn begin
#             # begin
#                 Bpacked_krem = PtrMatrix{-1,nc,Tb,kc}(threadlocal_L3CACHE_pointer(Ta), Krem)
#                 Bpmat_krem = PtrMatrix{-1,nc}(gesp(Bptr, (0, no*nc)), Krem)
#                 copyto!(Bpacked_krem, Bpmat_krem)
#                 # Bpacked_krem = PtrMatrix{-1,nc}(gesp(Bptr, (Kiter*kc, no*nc)), Krem)
#                 # Base.Threads.@sync begin
#                     for mo in 0:Miter-1
#                     # pack mc x kc block of A
#                     # Base.Threads.@spawn begin
#                     begin
#                         Apacked_krem = PtrMatrix{mc,-1,Ta,mc}(threadlocal_L2CACHE_pointer(Ta), Krem)
#                         Apmat_krem = PtrMatrix{mc,-1}(gesp(Aptr, (mo*mc, 0)), Krem)
#                         copyto!(Apacked_krem, Apmat_krem)
#                         Cpmat = PtrMatrix{mc,nc}(gesp(Cptr, (mo*mc, no*nc)))
#                         loopmul!(Cpmat, Apacked_krem, Bpacked_krem)
#                     end
#                 end
#                 # Mrem
#                 if Mrem > 0
#                     # Base.Threads.@spawn begin
#                     # begin
#                         Apacked_mrem_krem = PtrMatrix{-1,-1,Ta,mc}(threadlocal_L2CACHE_pointer(Ta), Mrem, Krem)
#                         Apmat_mrem_krem = PtrMatrix{-1,-1}(gesp(Aptr, (Miter*mc, 0)), Mrem, Krem)
#                         copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
#                         Cpmat_mrem = PtrMatrix{-1,nc}(gesp(Cptr, (Miter*mc, no*nc)), Mrem)
#                         loopmul!(Cpmat_mrem, Apacked_mrem_krem, Bpacked_krem)
#                     # end
#                 end
#                 # end # sync
#                 k = Krem
#                 for ko in 1:Kiter
#                     # pack kc x nc block of B
#                     Bpacked = PtrMatrix{kc,nc,Tb,kc}(threadlocal_L3CACHE_pointer(Ta))
#                     Bpmat = PtrMatrix{kc,nc}(gesp(Bptr, (k, no*nc)))
#                     copyto!(Bpacked, Bpmat)
#                     # Bprefetch = PtrMatrix{kc,nc}(gesp(Bptr, ((ko+1)*kc, no*nc)))
#                     # copyto!(Bpacked, Bpmat, Bprefetch)
#                     # Base.Threads.@sync begin
#                     for mo in 0:Miter-1
#                         # Base.Threads.@spawn begin
#                         begin
#                             # pack mc x kc block of A
#                             # Apmat = PtrMatrix{mc,kc}(gesp(Aptr, (mo*mc, ko*kc)))
#                             Apacked = PtrMatrix{mc,kc,Ta,mc}(threadlocal_L2CACHE_pointer(Ta))
#                             Apmat = PtrMatrix{mc,kc}(gesp(Aptr, (mo*mc, k)))
#                             copyto!(Apacked, Apmat)
#                             # Aprefetch = PtrMatrix{mc,kc}(gesp(Aptr, ((mo+1)*mc, ko*kc)))
#                             # copyto_prefetch3!(Apacked, Apmat, Aprefetch)
#                             Cpmat = PtrMatrix{mc,nc}(gesp(Cptr, (mo*mc, no*nc)))
#                             loopmuladd!(Cpmat, Apacked, Bpacked)
#                         end
#                     end
#                     # Mrem
#                     if Mrem > 0
#                         # Base.Threads.@spawn begin
#                         begin
#                             Apacked_mrem = PtrMatrix{-1,kc,Ta,mc}(threadlocal_L2CACHE_pointer(Ta), Mrem)
#                             Apmat_mrem = PtrMatrix{-1,kc}(gesp(Aptr, (Miter*mc, k)), Mrem)
#                             copyto!(Apacked_mrem, Apmat_mrem)
#                             Cpmat_mrem = PtrMatrix{-1,nc}(gesp(Cptr, (Miter*mc, no*nc)), Mrem)
#                             loopmuladd!(Cpmat_mrem, Apacked_mrem, Bpacked)
#                         end
#                     end
#                     # end
#                     k += kc
#                 end
#             end # spawn
#         end # nloop
#         # Nrem
#         if Nrem > 0
#             Base.Threads.@spawn begin
#             # begin
#                 # Krem
#                 # pack kc x nc block of B
#                 Bpacked_krem_nrem = PtrMatrix{-1,-1,Tb,kc}(threadlocal_L3CACHE_pointer(Ta), Krem, Nrem)
#                 Bpmat_krem_nrem = PtrMatrix{-1,-1}(gesp(Bptr, (0, Niter*nc)), Krem, Nrem)
#                 copyto!(Bpacked_krem_nrem, Bpmat_krem_nrem)
#                 # Bpacked_krem_nrem = PtrMatrix{-1,-1}(gesp(Bptr, (Kiter*kc, Niter*nc)), Krem, Nrem)
#                 # Base.Threads.@sync begin
#                 for mo in 0:Miter-1
#                     # Base.Threads.@spawn begin
#                     begin
#                         # pack mc x kc block of A
#                         Apacked_krem = PtrMatrix{mc,-1,Ta,mc}(threadlocal_L2CACHE_pointer(Ta), Krem)
#                         Apmat_krem = PtrMatrix{mc,-1}(gesp(Aptr, (mo*mc, 0)), Krem)
#                         copyto!(Apacked_krem, Apmat_krem)
#                         Cpmat_nrem = PtrMatrix{mc,-1}(gesp(Cptr, (mo*mc, Niter*nc)), Nrem)
#                         loopmul!(Cpmat_nrem, Apacked_krem, Bpacked_krem_nrem)
#                     end
#                 end
#                 # Mrem
#                 if Mrem > 0
#                     # Base.Threads.@spawn begin
#                     begin
#                         Apacked_mrem_krem = PtrMatrix{-1,-1,Ta,mc}(threadlocal_L2CACHE_pointer(Ta), Mrem, Krem)
#                         Apmat_mrem_krem = PtrMatrix{-1,-1}(gesp(Aptr, (Miter*mc, 0)), Mrem, Krem)
#                         copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
#                         Cpmat_mrem_nrem = PtrMatrix{-1,-1}(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
#                         loopmul!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem)
#                     end
#                 end
#                 # end # sync
#                 k = Krem
#                 for ko in 1:Kiter
#                     # pack kc x nc block of B
#                     Bpacked_nrem = PtrMatrix{kc,-1,Tb,kc}(threadlocal_L3CACHE_pointer(Ta), Nrem)
#                     Bpmat_nrem = PtrMatrix{kc,-1}(gesp(Bptr, (k, Niter*nc)), Nrem)
#                     copyto!(Bpacked_nrem, Bpmat_nrem)
#                     # Bpacked_nrem = PtrMatrix{kc,-1}(gesp(Bptr, (ko*kc, Niter*nc)), Nrem)
#                     # Base.Threads.@sync begin
#                     for mo in 0:Miter-1
#                         # Base.Threads.@spawn begin
#                         begin
#                             # pack mc x kc block of A
#                             Apacked = PtrMatrix{mc,kc,Ta,mc}(threadlocal_L2CACHE_pointer(Ta))
#                             Apmat = PtrMatrix{mc,kc}(gesp(Aptr, (mo*mc, k)))
#                             copyto!(Apacked, Apmat)
#                             Cpmat_nrem = PtrMatrix{mc,-1}(gesp(Cptr, (mo*mc, Niter*nc)), Nrem)
#                             loopmuladd!(Cpmat_nrem, Apacked, Bpacked_nrem)
#                         end
#                     end
#                     # Mrem
#                     if Mrem > 0
#                         # Base.Threads.@spawn begin
#                         begin
#                             Apacked_mrem = PtrMatrix{-1,kc,Ta,mc}(threadlocal_L2CACHE_pointer(Ta), Mrem)
#                             Apmat_mrem = PtrMatrix{-1,kc}(gesp(Aptr, (Miter*mc, k)), Mrem)
#                             copyto!(Apacked_mrem, Apmat_mrem)
#                             Cpmat_mrem_nrem = PtrMatrix{-1,-1}(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
#                             loopmuladd!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem)
#                         end
#                     end
#                     # end # sync
#                     k += kc
#                 end
#             end
#         end
#     end # GC.@preserve
#     C
# end

@generated function jmuladd!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}) where {Tc, Ta, Tb}
    mc, kc, nc, log2kc = matmul_params(Tc, Ta, Tb)
    quote
        M, K, N = matmul_sizes(C, A, B)
        Niter, Nrem = divrem(N, $nc)
        # Kiter, Krem = divrem(K, $kc)
        Miter, Mrem = divrem(M, $mc)
        (iszero(Miter) && iszero(Niter)) && return loopmul!(C, A, B)
        Kiter = K >>> $log2kc
        Krem = (K & $(kc - 1))
        fill!(C, zero($Tc))
        # Cptr = stridedpointer(C)
        Aptr = stridedpointer(A)
        Bptr = stridedpointer(B)
        Cptr = stridedpointer(C)
        ptrL2 = threadlocal_L2CACHE_pointer($Ta, 1)
        ptrL3 = Base.unsafe_convert(Ptr{$Tb}, ptrL2 + $(align(sizeof(Ta) * mc * kc)))#threadlocal_L3CACHE_pointer($Tb, 1)
        GC.@preserve C A B LCACHE begin
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
                        # Apmat = PtrMatrix{$mc,$kc}(gesp(Aptr, (mo*$mc, ko*$kc)))
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

function packarray!(dest::AbstractStrideArray{Tuple{M,K,N}}, src::AbstractStrideArray{Tuple{M,N,K}}) where {M,K,N}
    @inbounds for n ∈ 1:size(dest,3), k ∈ 1:size(dest,2), m ∈ 1:size(dest,1)
        dest[m,k,n] = src[m,n,k]
    end
end
function loopmul!(
    C::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,N}}, A::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}}, B::AbstractStrideMatrix{K,N}
) where {Mᵣ,Mᵢ,K,N}
    Mᵣs = Static{Mᵣ}(); Mᵢs = Static{Mᵢ}()
    @avx for n ∈ 1:size(C,3), mᵢ ∈ 1:Mᵢs, mᵣ ∈ 1:Mᵣs
        Cₘₙ = zero(eltype(C))
        for k ∈ 1:size(B,1)
            Cₘₙ += A[mᵣ,k,mᵢ] * B[k,n]
        end
        C[mᵣ,mᵢ,n] = Cₘₙ
    end
    nothing
end
function loopmuladd!(
    C::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,N}}, A::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}}, B::AbstractStrideMatrix{K,N}
) where {Mᵣ,Mᵢ,K,N}
    Mᵣs = Static{Mᵣ}(); Mᵢs = Static{Mᵢ}()
    @avx for n ∈ 1:size(C,3), mᵢ ∈ 1:Mᵢs, mᵣ ∈ 1:Mᵣs
        Cₘₙ = zero(eltype(C))
        for k ∈ 1:size(B,1)
            Cₘₙ += A[mᵣ,k,mᵢ] * B[k,n]
        end
        C[mᵣ,mᵢ,n] += Cₘₙ
    end
    nothing
end

@generated function jmulh!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}) where {Tc, Ta, Tb}
    W, Wshift = VectorizationBase.pick_vector_width_shift(promote_type(Tc, Ta, Tb))
    mc, kc, nc, log2kc = matmul_params(Tc, Ta, Tb)
    # W = VectorizationBase.pick_vector_width(promote_type(Tc, Ta, Tb))
    # We aim for using roughly half of the L1, L2, and L3 caches
    # kc = 5L₁ ÷ (12nᵣ * sizeof(Tb))
    mᵣW = mᵣ << Wshift
    mcrep = mc ÷ mᵣW
    Ablocklen = mc * kc
    quote
        M, K, N = matmul_sizes(C, A, B)
        Niter, Nrem = divrem(N, $nc)
        Miter, Mrem = divrem(M, $mc)
        (iszero(Miter) && iszero(Niter)) && return loopmul!(C, A, B)
        Kiter, Krem = divrem(K - 1, $kc)
        Krem += 1
        resize_Acache!(Ta, Val{$mc}(), Val{$kc}(), Miter, Kiter)
        # Kiter = K >>> $log2kc
        # Krem = K & $(kc - 1)
        # fill!(C, zero($Tc))
        # Cptr = stridedpointer(C)
        Aptr = stridedpointer(A)
        Bptr = stridedpointer(B)
        Cptr = stridedpointer(C)
        # ptrL2 = threadlocal_L2CACHE_pointer($Ta, 1)
        # ptrL3 = Base.unsafe_convert(Ptr{$Tb}, ptrL2 + $(align(sizeof(Ta) * mc * kc)))
        ptrL3 = B_pointer($Tb)
        ptrL2 = A_pointer($Ta)
        GC.@preserve C A B ACACHE BCACHE begin
            for no in 0:Niter-1
                # Krem
                # pack kc x nc block of B
                Bpacked_krem = PtrMatrix{-1,$nc,$Tb,$kc}(ptrL3, Krem)
                Bpmat_krem = PtrMatrix{-1,$nc}(gesp(Bptr, (0, no*$nc)), Krem)
                copyto!(Bpacked_krem, Bpmat_krem)
                for mo in 0:Miter-1
                    # pack mc x kc block of A
                    ptr_Apacked = gep(ptrL2, mo * $Ablocklen)
                    Apacked_krem = PtrArray{Tuple{$mᵣW,-1,$mcrep},$Ta,3,Tuple{1,$mᵣW,-1},1,1,false,-1}(ptr_Apacked, (Krem,), (Krem*$mᵣW,))
                    # Apacked_krem = PtrMatrix{$mc,-1,$Ta,$mc}(ptrL2, Krem)
                    if iszero(no)
                        Apmat_krem = PtrArray{Tuple{$mᵣW,$mcrep,-1},$Ta,3,Tuple{1,$mᵣW,-1},1,1,false,-1}(gep(Aptr, (mo*$mc, 0)), (Krem,), Aptr.strides)
                        packarray!(Apacked_krem, Apmat_krem)
                    end
                    Cptr_off = gep(Cptr, (mo*$mc, no*$nc))
                    Cpmat = PtrArray{Tuple{$mᵣW,$mcrep,$nc},$Tc,3,Tuple{1,$mᵣW,-1},0,1,false,$(mc*nc)}(Cptr_off, tuple(), Cptr.strides)
                    loopmul!(Cpmat, Apacked_krem, Bpacked_krem)
                end
                # Mrem
                if Mrem > 0
                    ptr_Apacked = gep(ptrL2, Miter * $Ablocklen)
                    Apacked_mrem_krem = PtrMatrix{-1,-1,$Ta,$mc}(ptr_Apacked, Mrem, Krem)
                    if iszero(no)
                        Apmat_mrem_krem = PtrMatrix{-1,-1}(gesp(Aptr, (Miter*$mc, 0)), Mrem, Krem)
                        copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
                    end
                    Cpmat_mrem = PtrMatrix{-1,$nc}(gesp(Cptr, (Miter*$mc, no*$nc)), Mrem)
                    loopmul!(Cpmat_mrem, Apacked_mrem_krem, Bpacked_krem)
                end
                k = Krem
                for ko in 1:Kiter
                    # pack kc x nc block of B
                    Bpacked = PtrMatrix{$kc,$nc,$Tb,$kc}(ptrL3)
                    Bpmat = PtrMatrix{$kc,$nc}(gesp(Bptr, (k, no*$nc)))
                    copyto!(Bpacked, Bpmat)
                    for mo in 0:Miter-1
                        # pack mc x kc block of A
                        ptr_Apacked = gep(ptrL2, (mo + (Miter + 1) * ko) * $Ablocklen)
                        Apacked = PtrArray{Tuple{$mᵣW,$kc,$mcrep},$Ta,3,Tuple{1,$(mᵣ*W),$(mᵣW*kc)},0,0,false,$(mc*kc)}(ptr_Apacked, tuple(), tuple())
                        if iszero(no)
                            Aptr_off = gep(Aptr, (mo*$mc, k))
                            Apmat = PtrArray{Tuple{$mᵣW,$mcrep,$kc},$Ta,3,Tuple{1,$mᵣW,-1},0,1,false,$(mc*kc)}(Aptr_off, tuple(), Aptr.strides)
                            packarray!(Apacked, Apmat)
                        end
                        Cptr_off = gep(Cptr, (mo*$mc, no*$nc))
                        Cpmat = PtrArray{Tuple{$mᵣW,$mcrep,$nc},$Tc,3,Tuple{1,$mᵣW,-1},0,1,false,$(mc*nc)}(Cptr_off, tuple(), Cptr.strides)
                        loopmuladd!(Cpmat, Apacked, Bpacked)
                    end
                    # Mrem
                    if Mrem > 0
                        ptr_Apacked = gep(ptrL2, (Miter + (Miter + 1) * ko) * $Ablocklen)
                        Apacked_mrem = PtrMatrix{-1,$kc,$Ta,$mc}(ptr_Apacked, Mrem)
                        if iszero(no)
                            Apmat_mrem = PtrMatrix{-1,$kc}(gesp(Aptr, (Miter*$mc, k)), Mrem)
                            copyto!(Apacked_mrem, Apmat_mrem)
                        end
                        Cpmat_mrem = PtrMatrix{-1,$nc}(gesp(Cptr, (Miter*$mc, no*$nc)), Mrem)
                        loopmuladd!(Cpmat_mrem, Apacked_mrem, Bpacked)
                    end
                    k += $kc
                end
            end
            # Nrem
            if Nrem > 0
                # Krem
                # pack kc x nc block of B
                Bpacked_krem_nrem = PtrMatrix{-1,-1,$Tb,$kc}(ptrL3, Krem, Nrem)
                Bpmat_krem_nrem = PtrMatrix{-1,-1}(gesp(Bptr, (0, Niter*$nc)), Krem, Nrem)
                copyto!(Bpacked_krem_nrem, Bpmat_krem_nrem)
                for mo in 0:Miter-1
                    # pack mc x kc block of A
                    ptr_Apacked = gep(ptrL2, mo * $Ablocklen)
                    Apacked_krem = PtrArray{Tuple{$mᵣW,-1,$mcrep},$Ta,3,Tuple{1,$mᵣW,-1},1,1,false,-1}(ptr_Apacked, (Krem,), (Krem*$mᵣW,))
                    # Apacked_krem = PtrMatrix{$mc,-1,$Ta,$mc}(ptrL2, Krem)
                    if iszero(Niter)
                        Apmat_krem = PtrArray{Tuple{$mᵣW,$mcrep,-1},$Ta,3,Tuple{1,$mᵣW,-1},1,1,false,-1}(gep(Aptr, (mo*$mc, 0)), (Krem,), Aptr.strides)
                        packarray!(Apacked_krem, Apmat_krem)
                    end
                    Cptr_off = gep(Cptr, (mo*$mc, Niter*$nc))
                    Cpmat_nrem = PtrArray{Tuple{$mᵣW,$mcrep,-1},$Tc,3,Tuple{1,$mᵣW,-1},1,1,false,-1}(Cptr_off, (Nrem,), Cptr.strides)
                    loopmul!(Cpmat_nrem, Apacked_krem, Bpacked_krem_nrem)
                end
                # Mrem
                if Mrem > 0
                    Apacked_mrem_krem = PtrMatrix{-1,-1,$Ta,$mc}(ptrL2, Mrem, Krem)
                    if iszero(Niter)
                        Apmat_mrem_krem = PtrMatrix{-1,-1}(gesp(Aptr, (Miter*$mc, 0)), Mrem, Krem)
                        copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
                    end
                    Cpmat_mrem_nrem = PtrMatrix{-1,-1}(gesp(Cptr, (Miter*$mc, Niter*$nc)), Mrem, Nrem)
                    loopmul!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem)
                end
                k = Krem
                for ko in 1:Kiter
                    # pack kc x nc block of B
                    Bpacked_nrem = PtrMatrix{$kc,-1,$Tb,$kc}(ptrL3, Nrem)
                    Bpmat_nrem = PtrMatrix{$kc,-1}(gesp(Bptr, (k, Niter*$nc)), Nrem)
                    copyto!(Bpacked_nrem, Bpmat_nrem)
                    for mo in 0:Miter-1
                        # pack mc x kc block of A
                        ptr_Apacked = gep(ptrL2, (mo + (Miter + 1) * ko) * $Ablocklen)
                        Apacked = PtrArray{Tuple{$mᵣW,$kc,$mcrep},$Ta,3,Tuple{1,$(mᵣ*W),$(mᵣW*kc)},0,0,false,$(mc*kc)}(ptr_Apacked, tuple(), tuple())
                        if iszero(Niter)
                            Aptr_off = gep(Aptr, (mo*$mc, k))
                            Apmat = PtrArray{Tuple{$mᵣW,$mcrep,$kc},$Ta,3,Tuple{1,$mᵣW,-1},0,1,false,$(mc*kc)}(Aptr_off, tuple(), Aptr.strides)
                            packarray!(Apacked, Apmat)
                        end
                        Cptr_off = gep(Cptr, (mo*$mc, Niter*$nc))
                        Cpmat_nrem = PtrArray{Tuple{$mᵣW,$mcrep,-1},$Tc,3,Tuple{1,$mᵣW,-1},1,1,false,-1}(Cptr_off, (Nrem,), Cptr.strides)
                        loopmuladd!(Cpmat_nrem, Apacked, Bpacked_nrem)
                    end
                    # Mrem
                    if Mrem > 0
                        ptr_Apacked = gep(ptrL2, (Miter + (Miter + 1) * ko) * $Ablocklen)
                        Apacked_mrem = PtrMatrix{-1,$kc,$Ta,$mc}(ptr_Apacked, Mrem)
                        if iszero(Niter)
                            Apmat_mrem = PtrMatrix{-1,$kc}(gesp(Aptr, (Miter*$mc, k)), Mrem)
                            copyto!(Apacked_mrem, Apmat_mrem)
                        end
                        Cpmat_mrem_nrem = PtrMatrix{-1,-1}(gesp(Cptr, (Miter*$mc, Niter*$nc)), Mrem, Nrem)
                        loopmuladd!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem)
                    end
                    k += $kc
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


LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractMatrix, B::AbstractMatrix) = jmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractStrideMatrix, B::AbstractMatrix) = jmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractStrideMatrix, B::AbstractMatrix) = jmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractStrideMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractStrideMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)
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
    sp + align(sizeof(T)*L), mv
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
    sp + align(sizeof(T)*L), mv'
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


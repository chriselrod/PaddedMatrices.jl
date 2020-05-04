    

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
    kc = 15(L₁ ÷ (20nᵣ * sizeof(T)))
    # mcrep =  L₂ ÷ (2kc * sizeof(T) * mᵣ * W)
    mcrep =  5L₂ ÷ (8kc * sizeof(T) * mᵣ * W)
    ncrep = 8L₃ ÷ (16kc * sizeof(T) * nᵣ)
    # ncrep = 5L₃ ÷ (16kc * sizeof(T) * nᵣ)
    mc = mcrep * mᵣ * W
    nc = ncrep * nᵣ * VectorizationBase.NUM_CORES
    mc, kc, nc
end
function matmul_params_val(::Type{T}) where {T}
    mc, kc, nc = matmul_params(T)
    Val(mc), Val(kc), Val(nc)
end



function pack_B_krem(Bptr, ::Val{kc}, ::Val{nc}, ::Type{Tb}, Krem, no::Int) where {nc, Tb, kc}
    Bpacked_krem = PtrMatrix(threadlocal_L3CACHE_pointer(Tb), Krem, nc, VectorizationBase.align(Krem, Tb))
    Bpmat_krem = PtrMatrix(gesp(Bptr, (0, no*nc)), Krem, nc)
    copyto!(Bpacked_krem, Bpmat_krem)
    Bpacked_krem
end
function pack_B(Bptr, ::Val{kc}, ::Val{nc}, ::Type{Tb}, k::Int, no::Int) where {nc, Tb, kc}
    Bpacked = PtrMatrix(threadlocal_L3CACHE_pointer(Tb), kc, nc, kc)
    Bpmat = PtrMatrix(gesp(Bptr, (k, no*nc)), kc, nc)
    copyto!(Bpacked, Bpmat)
    Bpacked
end
function pack_B_krem_nrem(
    Bptr, ::Val{kc}, ::Val{nc}, ::Type{Tb}, Krem, Nrem, Niter
) where {nc, Tb, kc}
    Bpacked_krem_nrem = PtrMatrix(threadlocal_L3CACHE_pointer(Tb), Krem, Nrem, VectorizationBase.align(Krem, Tb))
    Bpmat_krem_nrem = PtrMatrix(gesp(Bptr, (0, Niter*nc)), Krem, Nrem)
    copyto!(Bpacked_krem_nrem, Bpmat_krem_nrem)
    Bpacked_krem_nrem
end
function pack_B_nrem(Bptr, ::Val{kc}, ::Val{nc}, ::Type{Tb}, k::Int, Nrem, Niter) where {nc, Tb, kc}
    Bpacked_nrem = PtrMatrix(threadlocal_L3CACHE_pointer(Tb), kc, Nrem, kc)
    Bpmat_nrem = PtrMatrix(gesp(Bptr, (k, Niter*nc)), kc, Nrem)
    copyto!(Bpacked_nrem, Bpmat_nrem)
    Bpacked_nrem
end
function pack_A_krem(Aptr, ::Val{mc}, ::Type{Ta}, mo::Int, Krem) where {mc, Ta}
    Apacked_krem = PtrMatrix(threadlocal_L2CACHE_pointer(Ta), mc, Krem, mc)
    Apmat_krem = PtrMatrix(gesp(Aptr, (mo*mc, 0)), mc, Krem)
    copyto!(Apacked_krem, Apmat_krem)
    Apacked_krem
end
function pack_A_krem(Aptr, mc, ::Type{Ta}, mo::Int, Krem) where {Ta}
    Apacked_krem = PtrMatrix(threadlocal_L2CACHE_pointer(Ta), mc, Krem, mc)# assumed aligned
    Apmat_krem = PtrMatrix(gesp(Aptr, (mo*mc, 0)), mc, Krem)
    copyto!(Apacked_krem, Apmat_krem)
    Apacked_krem
end
function pack_A_mrem(Aptr, ::Val{mc}, ::Val{kc}, ::Type{Ta}, Mrem, k::Int, Miter) where {mc, kc, Ta}
    Apacked_mrem = PtrMatrix(threadlocal_L2CACHE_pointer(Ta), Mrem, kc, VectorizationBase.align(Mrem, Ta))
    Apmat_mrem = PtrMatrix(gesp(Aptr, (Miter*mc, k)), Mrem, kc)
    copyto!(Apacked_mrem, Apmat_mrem)
    Apacked_mrem
end
function pack_A_mrem(Aptr, mc, kc, ::Type{Ta}, Mrem, k::Int, Miter) where {Ta}
    Apacked_mrem = PtrMatrix(threadlocal_L2CACHE_pointer(Ta), Mrem, kc, VectorizationBase.align(Mrem, Ta))
    Apmat_mrem = PtrMatrix(gesp(Aptr, (Miter*mc, k)), Mrem, kc)
    copyto!(Apacked_mrem, Apmat_mrem)
    Apacked_mrem
end
function pack_A_mrem_krem(Aptr, ::Val{mc}, ::Type{Ta}, Mrem, Krem, Miter) where {mc, Ta}
    Apacked_mrem_krem = PtrMatrix(threadlocal_L2CACHE_pointer(Ta), Mrem, Krem, VectorizationBase.align(Mrem, Ta))
    Apmat_mrem_krem = PtrMatrix(gesp(Aptr, (Miter*mc, 0)), Mrem, Krem)
    copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
    Apacked_mrem_krem
end
function pack_A_mrem_krem(Aptr, mc, ::Type{Ta}, Mrem, Krem, Miter) where {Ta}
    Apacked_mrem_krem = PtrMatrix(threadlocal_L2CACHE_pointer(Ta), Mrem, Krem, VectorizationBase.align(Mrem, Ta))
    Apmat_mrem_krem = PtrMatrix(gesp(Aptr, (Miter*mc, 0)), Mrem, Krem)
    copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
    Apacked_mrem_krem
end
function pack_A(Aptr, ::Val{mc}, ::Val{kc}, ::Type{Ta}, mo::Int, k::Int) where {mc, kc, Ta}
    Apacked = PtrMatrix(threadlocal_L2CACHE_pointer(Ta), mc, kc, mc)
    Apmat = PtrMatrix(gesp(Aptr, (mo*mc, k)), mc, kc)
    # @show (reinterpret(Int, pointer(Apmat)) - reinterpret(Int,pointer(Aptr))) >>> 3
    copyto!(Apacked, Apmat)
    Apacked
end
function pack_A(Aptr, mc, kc, ::Type{Ta}, mo::Int, k::Int) where {Ta}
    Apacked = PtrMatrix(threadlocal_L2CACHE_pointer(Ta), mc, kc, mc)
    Apmat = PtrMatrix(gesp(Aptr, (mo*mc, k)), mc, kc)
    # @show (reinterpret(Int, pointer(Apmat)) - reinterpret(Int,pointer(Aptr))) >>> 3
    copyto!(Apacked, Apmat)
    Apacked
end

function jmulpackAonly!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, ::Val{mc}, ::Val{kc}, ::Val{nc}) where {Tc, Ta, Tb, mc, kc, nc}
    M, K, N = matmul_sizes(C, A, B)
    W = VectorizationBase.pick_vector_width(Tc)
    mᵣW = mᵣ * W

    num_m_iter = cld(M, mc)
    mreps_per_iter = M ÷ num_m_iter
    mreps_per_iter += mᵣW - 1
    mcrepetitions = mreps_per_iter ÷ mᵣW
    mreps_per_iter = mᵣW * mcrepetitions
    Miter = num_m_iter - 1
    Mrem = M - Miter * mreps_per_iter
    
    #(iszero(Miter) && iszero(Niter)) && return loopmul!(C, A, B)
    num_k_iter = cld(K, kc)
    Krepetitions, Krem = divrem(K, num_k_iter)
    Krem += Krepetitions
    Kiter = num_k_iter - 1

    Aptr = stridedpointer(A)
    Bptr = stridedpointer(B)
    Cptr = stridedpointer(C)
    GC.@preserve C A B LCACHEARRAY begin
        # Krem
        # pack Krepetitions x nc block of B
        Bpacked_krem_nrem = PtrMatrix(Bptr, Krem, N)
        for mo in 0:Miter-1
            # pack mreps_per_iter x Krepetitions block of A
            Apacked_krem = pack_A_krem(Aptr, mreps_per_iter, Tc, mo, Krem)
            Cpmat_nrem = PtrMatrix(gesp(Cptr, (mo*mreps_per_iter, 0)), mreps_per_iter, N)
            loopmulprefetch!(Cpmat_nrem, Apacked_krem, Bpacked_krem_nrem, α, β)
        end
        # Mrem
        if Mrem > 0
            Apacked_mrem_krem = pack_A_mrem_krem(Aptr, mreps_per_iter, Tc, Mrem, Krem, Miter)
            Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mreps_per_iter, 0)), Mrem, N)
            loopmulprefetch!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem, α, β)
        end
        k = Krem
        for ko in 1:Kiter
            # pack Krepetitions x nc block of B
            Bpacked_nrem = PtrMatrix(gesp(Bptr, (k, 0)), Krepetitions, N)
            for mo in 0:Miter-1
                # pack mreps_per_iter x Krepetitions block of A
                Apacked = pack_A(Aptr, mreps_per_iter, Krepetitions, Tc, mo, k)
                Cpmat_nrem = PtrMatrix(gesp(Cptr, (mo*mreps_per_iter, 0)), mreps_per_iter, N)
                loopmulprefetch!(Cpmat_nrem, Apacked, Bpacked_nrem, α, Val{1}())
            end
            # Mrem
            if Mrem > 0
                Apacked_mrem = pack_A_mrem(Aptr, mreps_per_iter, Krepetitions, Tc, Mrem, k, Miter)
                Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mreps_per_iter, 0)), Mrem, N)
                loopmulprefetch!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem, α, Val{1}())
            end
            k += Krepetitions
        end
    end # GC.@preserve
    C
end

function jmul!(
    C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, ::Val{mc}, ::Val{kc}, ::Val{nc}
) where {Tc, Ta, Tb, mc, kc, nc}
    M, K, N = matmul_sizes(C, A, B)
    if N ≤ nc
        W = VectorizationBase.pick_vector_width(Tc)
        if isone(LinearAlgebra.stride1(A)) && ( (M ≤ 72)  || ((2M ≤ 5mc) && iszero(stride(A,2) % W)))
            loopmul!(C, A, B, α, β); return C
        elseif K ≤ 4kc
            return jmulpackAonly!(C, A, B, α, β, Val{mc}(), Val{kc}(), Val{nc}())
        end
    else#if N > 4nc
        return jmulh!(C, A, B, α, β)
    end
    Niter, Nrem = divrem(N, Static{nc}())
    Miter, Mrem = divrem(M, Static{mc}())
    Km1 = VectorizationBase.staticm1(K)
    Kiter, _Krem = divrem(Km1, Static{kc}())
    Krem = VectorizationBase.staticp1(_Krem)
    # fill!(C, zero(Tc))
    # Cptr = stridedpointer(C)
    Aptr = stridedpointer(A)
    Bptr = stridedpointer(B)
    Cptr = stridedpointer(C)
    GC.@preserve C A B LCACHEARRAY begin
        for no in 0:Niter - 1
            # Krem
            # pack kc x nc block of B
            Bpacked_krem = pack_B_krem(Bptr, Val{kc}(), Val{nc}(), Tb, Krem, no)
            # Bpacked_krem = PtrMatrix{-1,nc}(gesp(Bptr, (Kiter*kc, no*nc)), Krem)
            for mo in 0:Miter - 1
                # pack mc x kc block of A
                Apacked_krem = pack_A_krem(Aptr, Val{mc}(), Ta, mo, Krem)
                Cpmat = PtrMatrix(gesp(Cptr, (mo*mc, no*nc)), mc, nc)
                loopmulprefetch!(Cpmat, Apacked_krem, Bpacked_krem, α, β)
            end
            # Mrem
            if Mrem > 0
                Apacked_mrem_krem = pack_A_mrem_krem(Aptr, Val{mc}(), Ta, Mrem, Krem, Miter)
                Cpmat_mrem = PtrMatrix(gesp(Cptr, (Miter*mc, no*nc)), Mrem, nc)
                loopmulprefetch!(Cpmat_mrem, Apacked_mrem_krem, Bpacked_krem, α, β)
            end
            k = VectorizationBase.unwrap(Krem)
            for ko in 1:Kiter
                # pack kc x nc block of B
                Bpacked = pack_B(Bptr, Val{kc}(), Val{nc}(), Tb, k, no)
                for mo in 0:Miter-1
                    # pack mc x kc block of A
                    Apacked = pack_A(Aptr, Val{mc}(), Val{kc}(), Ta, mo, k)
                    Cpmat = PtrMatrix(gesp(Cptr, (mo*mc, no*nc)), mc, nc)
                    loopmulprefetch!(Cpmat, Apacked, Bpacked, α, Val{1}())
                end
                # Mrem
                if Mrem > 0
                    Apacked_mrem = pack_A_mrem(Aptr, Val{mc}(), Val{kc}(), Ta, Mrem, k, Miter)
                    Cpmat_mrem = PtrMatrix(gesp(Cptr, (Miter*mc, no*nc)), Mrem, nc)
                    loopmulprefetch!(Cpmat_mrem, Apacked_mrem, Bpacked, α, Val{1}())
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
                Cpmat_nrem = PtrMatrix(gesp(Cptr, (mo*mc, Niter*nc)), mc, Nrem)
                loopmulprefetch!(Cpmat_nrem, Apacked_krem, Bpacked_krem_nrem, α, β)
            end
            # Mrem
            if Mrem > 0
                Apacked_mrem_krem = pack_A_mrem_krem(Aptr, Val{mc}(), Ta, Mrem, Krem, Miter)
                Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
                loopmulprefetch!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem, α, β)
            end
            k = VectorizationBase.unwrap(Krem)
            for ko in 1:Kiter
                # pack kc x nc block of B
                Bpacked_nrem = pack_B_nrem(Bptr, Val{kc}(), Val{nc}(), Tb, k, Nrem, Niter)
                for mo in 0:Miter-1
                    # pack mc x kc block of A
                    Apacked = pack_A(Aptr, Val{mc}(), Val{kc}(), Ta, mo, k)
                    Cpmat_nrem = PtrMatrix(gesp(Cptr, (mo*mc, Niter*nc)), mc, Nrem)
                    loopmulprefetch!(Cpmat_nrem, Apacked, Bpacked_nrem, α, Val{1}())
                end
                # Mrem
                if Mrem > 0
                    Apacked_mrem = pack_A_mrem(Aptr, Val{mc}(), Val{kc}(), Ta, Mrem, k, Miter)
                    Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
                    loopmulprefetch!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem, α, Val{1}())
                end
                k += kc
            end
        end
    end # GC.@preserve
    C
end # function


function jmult!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, ::Val{mc}, ::Val{kc}, ::Val{nc}) where {Tc, Ta, Tb, mc, kc, nc, log2kc}
    M, K, N = matmul_sizes(C, A, B)
    Niter, Nrem = divrem(N, Static{nc}())
    Miter, Mrem = divrem(M, Static{mc}())
    if iszero(Niter) & iszero(Miter)
        if Mrem*sizeof(Ta) > 255
            loopmulprefetch!(C, A, B, α, β); return C
        else
            loopmul!(C, A, B, α, β); return C
        end
    end
    Km1 = VectorizationBase.staticm1(K)
    Kiter, _Krem = divrem(Km1, Static{kc}())
    Krem = VectorizationBase.staticp1(_Krem)
    Aptr = stridedpointer(A)
    Bptr = stridedpointer(B)
    Cptr = stridedpointer(C)
    Base.Threads.@sync GC.@preserve C A B LCACHEARRAY begin
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
                            Cpmat = PtrMatrix(gesp(Cptr, (mo*mc, no*nc)), mc, nc)
                            loopmulprefetch!(Cpmat, Apacked_krem, Bpacked_krem, α, β)
                        end
                    end
                    # Mrem
                    if Mrem > 0
                        Base.Threads.@spawn begin
                            # begin
                            Apacked_mrem_krem = pack_A_mrem_krem(Aptr, Val{mc}(), Ta, Mrem, Krem, Miter)
                            Cpmat_mrem = PtrMatrix(gesp(Cptr, (Miter*mc, no*nc)), Mrem, nc)
                            loopmulprefetch!(Cpmat_mrem, Apacked_mrem_krem, Bpacked_krem, α, β)
                        end
                    end
                end # sync
                k = VectorizationBase.unwrap(Krem)
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
                                Cpmat = PtrMatrix(gesp(Cptr, (mo*mc, no*nc)), mc, nc)
                                loopmulprefetch!(Cpmat, Apacked, Bpacked, α, Val{1}())
                            end
                        end
                        # Mrem
                        if Mrem > 0
                            Base.Threads.@spawn begin
                                # begin
                                Apacked_mrem = pack_A_mrem(Aptr, Val{mc}(), Val{kc}(), Ta, Mrem, k, Miter)
                                Cpmat_mrem = PtrMatrix(gesp(Cptr, (Miter*mc, no*nc)), Mrem, nc)
                                loopmulprefetch!(Cpmat_mrem, Apacked_mrem, Bpacked, α, Val{1}())
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
                            Cpmat_nrem = PtrMatrix(gesp(Cptr, (mo*mc, Niter*nc)), mc, Nrem)
                            loopmulprefetch!(Cpmat_nrem, Apacked_krem, Bpacked_krem_nrem, α, β)
                        end
                    end
                    # Mrem
                    if Mrem > 0
                        Base.Threads.@spawn begin
                            # begin
                            Apacked_mrem_krem = pack_A_mrem_krem(Aptr, Val{mc}(), Ta, Mrem, Krem, Miter)
                            Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
                            loopmulprefetch!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem, α, β)
                        end
                    end
                end # sync
                k = VectorizationBase.unwrap(Krem)
                for ko in 1:Kiter
                    # pack kc x nc block of B
                    Bpacked_nrem = pack_B_nrem(Bptr, Val{kc}(), Val{nc}(), Tb, k, Nrem, Niter)
                    Base.Threads.@sync begin
                        for mo in 0:Miter-1
                            Base.Threads.@spawn begin
                                # begin
                                # pack mc x kc block of A
                                Apacked = pack_A(Aptr, Val{mc}(), Val{kc}(), Ta, mo, k)
                                Cpmat_nrem = PtrMatrix(gesp(Cptr, (mo*mc, Niter*nc)), mc, Nrem)
                                loopmulprefetch!(Cpmat_nrem, Apacked, Bpacked_nrem, α, Val{1}())
                            end
                        end
                        # Mrem
                        if Mrem > 0
                            Base.Threads.@spawn begin
                                # begin
                                Apacked_mrem = pack_A_mrem(Aptr, Val{mc}(), Val{kc}(), Ta, Mrem, k, Miter)
                                Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
                                loopmulprefetch!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem, α, Val{1}())
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




function loopmul!(
    C::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,Nᵣ,Nᵢ}},
    A::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}},
    B::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ}},
    ::Val{1}, ::Val{0}
) where {Mᵣ,Mᵢ,K,Nᵣ,Nᵢ}
    Mᵣs = Static{Mᵣ}();# Mᵢs = Static{Mᵢ}()
    Nᵣs = Static{Nᵣ}();# Nᵢs = Static{Nᵢ}()
    @avx for nᵢ ∈ axes(C,4), mᵢ ∈ axes(A,3), mᵣ ∈ 1:Mᵣs, nᵣ ∈ 1:Nᵣs
         Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[mᵣ,k,mᵢ] * B[nᵣ,k,nᵢ]
        end
        C[mᵣ,mᵢ,nᵣ,nᵢ] = Cₘₙ
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,Nᵣ,Nᵢ}},
    A::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}},
    B::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ}},
    ::Val{1}, ::Val{1}
) where {Mᵣ,Mᵢ,K,Nᵣ,Nᵢ}
    Mᵣs = Static{Mᵣ}();# Mᵢs = Static{Mᵢ}()
    Nᵣs = Static{Nᵣ}();# Nᵢs = Static{Nᵢ}()
    @avx for nᵢ ∈ axes(C,4), mᵢ ∈ axes(A,3), mᵣ ∈ 1:Mᵣs, nᵣ ∈ 1:Nᵣs
         Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[mᵣ,k,mᵢ] * B[nᵣ,k,nᵢ]
        end
        C[mᵣ,mᵢ,nᵣ,nᵢ] += Cₘₙ
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,Nᵣ,Nᵢ}},
    A::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}},
    B::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ}},
    ::Val{1}, β
) where {Mᵣ,Mᵢ,K,Nᵣ,Nᵢ}
    Mᵣs = Static{Mᵣ}();# Mᵢs = Static{Mᵢ}()
    Nᵣs = Static{Nᵣ}();# Nᵢs = Static{Nᵢ}()
    @avx for nᵢ ∈ axes(C,4), mᵢ ∈ 1:Mᵢs, mᵣ ∈ 1:Mᵣs, nᵣ ∈ 1:Nᵣs
        Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[mᵣ,k,mᵢ] * B[nᵣ,k,nᵢ]
        end
        C[mᵣ,mᵢ,nᵣ,nᵢ] = Cₘₙ + β * C[mᵣ,mᵢ,nᵣ,nᵢ]
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,Nᵣ,Nᵢ}},
    A::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}},
    B::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ}},
    α, ::Val{0}
) where {Mᵣ,Mᵢ,K,Nᵣ,Nᵢ}
    Mᵣs = Static{Mᵣ}();# Mᵢs = Static{Mᵢ}()
    Nᵣs = Static{Nᵣ}();# Nᵢs = Static{Nᵢ}()
    @avx for nᵢ ∈ axes(C,4), mᵢ ∈ axes(A,3), mᵣ ∈ 1:Mᵣs, nᵣ ∈ 1:Nᵣs
         Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[mᵣ,k,mᵢ] * B[nᵣ,k,nᵢ]
        end
        C[mᵣ,mᵢ,nᵣ,nᵢ] = α * Cₘₙ
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,Nᵣ,Nᵢ}},
    A::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}},
    B::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ}},
    α, ::Val{1}
) where {Mᵣ,Mᵢ,K,Nᵣ,Nᵢ}
    Mᵣs = Static{Mᵣ}();# Mᵢs = Static{Mᵢ}()
    Nᵣs = Static{Nᵣ}();# Nᵢs = Static{Nᵢ}()
    @avx for nᵢ ∈ axes(C,4), mᵢ ∈ axes(A,3), mᵣ ∈ 1:Mᵣs, nᵣ ∈ 1:Nᵣs
        Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[mᵣ,k,mᵢ] * B[nᵣ,k,nᵢ]
        end
        C[mᵣ,mᵢ,nᵣ,nᵢ] += α * Cₘₙ
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,Nᵣ,Nᵢ}},
    A::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}},
    B::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ}},
    α, β
) where {Mᵣ,Mᵢ,K,Nᵣ,Nᵢ}
    Mᵣs = Static{Mᵣ}(); #Mᵢs = Static{Mᵢ}()
    Nᵣs = Static{Nᵣ}(); #Nᵢs = Static{Nᵢ}()
    @avx for nᵢ ∈ axes(C,4), mᵢ ∈ axes(A,3), mᵣ ∈ 1:Mᵣs, nᵣ ∈ 1:Nᵣs
        Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[mᵣ,k,mᵢ] * B[nᵣ,k,nᵢ]
        end
        C[mᵣ,mᵢ,nᵣ,nᵢ] = α * Cₘₙ + β * C[mᵣ,mᵢ,nᵣ,nᵢ]
    end
    nothing
end

function loopmul!(
    C::AbstractStrideArray{Tuple{M,Nᵣ,Nᵢ}},
    A::AbstractStrideArray{Tuple{M,K}},
    B::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ}},
    ::Val{1}, ::Val{0}
) where {M,K,Nᵣ,Nᵢ}
    Nᵣs = Static{Nᵣ}();# Nᵢs = Static{Nᵢ}()
    @avx for nᵢ ∈ axes(C,3), m ∈ axes(A,1), nᵣ ∈ 1:Nᵣs
        Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[m,k] * B[nᵣ,k,nᵢ]
        end
        C[m,nᵣ,nᵢ] = Cₘₙ
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{M,Nᵣ,Nᵢ}},
    A::AbstractStrideArray{Tuple{M,K}},
    B::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ}},
    ::Val{1}, ::Val{1}
) where {M,K,Nᵣ,Nᵢ}
    Nᵣs = Static{Nᵣ}();# Nᵢs = Static{Nᵢ}()
    @avx for nᵢ ∈ axes(C,3), m ∈ axes(A,1), nᵣ ∈ 1:Nᵣs
        Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[m,k] * B[nᵣ,k,nᵢ]
        end
        C[m,nᵣ,nᵢ] += Cₘₙ
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{M,Nᵣ,Nᵢ}},
    A::AbstractStrideArray{Tuple{M,K}},
    B::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ}},
    ::Val{1}, β
) where {M,K,Nᵣ,Nᵢ}
    Nᵣs = Static{Nᵣ}();# Nᵢs = Static{Nᵢ}()
    @avx for nᵢ ∈ axes(C,3), m ∈ axes(A,1), nᵣ ∈ 1:Nᵣs
        Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[m,k] * B[nᵣ,k,nᵢ]
        end
        C[m,nᵣ,nᵢ] = Cₘₙ + β * C[m,nᵣ,nᵢ]
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{M,Nᵣ,Nᵢ}},
    A::AbstractStrideArray{Tuple{M,K}},
    B::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ}},
    α, ::Val{0}
) where {M,K,Nᵣ,Nᵢ}
    Nᵣs = Static{Nᵣ}();# Nᵢs = Static{Nᵢ}()
    @avx for nᵢ ∈ axes(C,3), m ∈ axes(A,1), nᵣ ∈ 1:Nᵣs
        Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[m,k] * B[nᵣ,k,nᵢ]
        end
        C[m,nᵣ,nᵢ] = α * Cₘₙ
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{M,Nᵣ,Nᵢ}},
    A::AbstractStrideArray{Tuple{M,K}},
    B::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ}},
    α, ::Val{1}
) where {M,K,Nᵣ,Nᵢ}
    Nᵣs = Static{Nᵣ}();# Nᵢs = Static{Nᵢ}()
    @avx for nᵢ ∈ axes(C,3), m ∈ axes(A,1), nᵣ ∈ 1:Nᵣs
        Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[m,k] * B[nᵣ,k,nᵢ]
        end
        C[m,nᵣ,nᵢ] += α * Cₘₙ
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{M,Nᵣ,Nᵢ}},
    A::AbstractStrideArray{Tuple{M,K}},
    B::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ}},
    α, β
) where {M,K,Nᵣ,Nᵢ}
    Nᵣs = Static{Nᵣ}();# Nᵢs = Static{Nᵢ}()
    @avx for nᵢ ∈ axes(C,3), m ∈ axes(A,1), nᵣ ∈ 1:Nᵣs
        Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[m,k] * B[nᵣ,k,nᵢ]
        end
        C[m,nᵣ,nᵢ] = α * Cₘₙ + β * C[m,nᵣ,nᵢ]
    end
    nothing
end


function loopmul!(
    C::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,Nᵣ}},
    A::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}},
    B::AbstractStrideArray{Tuple{Nᵣ,K}},
    ::Val{1}, ::Val{0}
) where {Mᵣ,Mᵢ,K,Nᵣ}
    Mᵣs = Static{Mᵣ}();# Mᵢs = Static{Mᵢ}()
    @avx for mᵢ ∈ axes(A,3), mᵣ ∈ 1:Mᵣs, nᵣ ∈ axes(C,3)
         Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[mᵣ,k,mᵢ] * B[nᵣ,k]
        end
        C[mᵣ,mᵢ,nᵣ] = Cₘₙ
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,Nᵣ}},
    A::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}},
    B::AbstractStrideArray{Tuple{Nᵣ,K}},
    ::Val{1}, ::Val{1}
) where {Mᵣ,Mᵢ,K,Nᵣ}
    Mᵣs = Static{Mᵣ}();# Mᵢs = Static{Mᵢ}()
    @avx for mᵢ ∈ axes(A,3), mᵣ ∈ 1:Mᵣs, nᵣ ∈ axes(C,3)
         Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[mᵣ,k,mᵢ] * B[nᵣ,k]
        end
        C[mᵣ,mᵢ,nᵣ] += Cₘₙ
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,Nᵣ}},
    A::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}},
    B::AbstractStrideArray{Tuple{Nᵣ,K}},
    ::Val{1}, β
) where {Mᵣ,Mᵢ,K,Nᵣ}
    Mᵣs = Static{Mᵣ}();# Mᵢs = Static{Mᵢ}()
    @avx for mᵢ ∈ axes(A,3), mᵣ ∈ 1:Mᵣs, nᵣ ∈ axes(C,3)
        Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[mᵣ,k,mᵢ] * B[nᵣ,k]
        end
        C[mᵣ,mᵢ,nᵣ] = Cₘₙ + β * C[mᵣ,mᵢ,nᵣ]
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,Nᵣ}},
    A::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}},
    B::AbstractStrideArray{Tuple{Nᵣ,K}},
    α, ::Val{0}
) where {Mᵣ,Mᵢ,K,Nᵣ}
    Mᵣs = Static{Mᵣ}();# Mᵢs = Static{Mᵢ}()
    @avx for mᵢ ∈ axes(A,3), mᵣ ∈ 1:Mᵣs, nᵣ ∈ axes(C,3)
         Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[mᵣ,k,mᵢ] * B[nᵣ,k]
        end
        C[mᵣ,mᵢ,nᵣ] = α * Cₘₙ
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,Nᵣ}},
    A::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}},
    B::AbstractStrideArray{Tuple{Nᵣ,K}},
    α, ::Val{1}
) where {Mᵣ,Mᵢ,K,Nᵣ}
    Mᵣs = Static{Mᵣ}();# Mᵢs = Static{Mᵢ}()
    @avx for mᵢ ∈ axes(A,3), mᵣ ∈ 1:Mᵣs, nᵣ ∈ axes(C,3)
        Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[mᵣ,k,mᵢ] * B[nᵣ,k]
        end
        C[mᵣ,mᵢ,nᵣ] += α * Cₘₙ
    end
    nothing
end
function loopmul!(
    C::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,Nᵣ}},
    A::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}},
    B::AbstractStrideArray{Tuple{Nᵣ,K}},
    α, β
) where {Mᵣ,Mᵢ,K,Nᵣ}
    Mᵣs = Static{Mᵣ}(); #Mᵢs = Static{Mᵢ}()
    @avx for mᵢ ∈ axes(A,3), mᵣ ∈ 1:Mᵣs, nᵣ ∈ axes(C,3)
        Cₘₙ = zero(eltype(C))
        for k ∈ axes(A,2)
            Cₘₙ += A[mᵣ,k,mᵢ] * B[nᵣ,k]
        end
        C[mᵣ,mᵢ,nᵣ] = α * Cₘₙ + β * C[mᵣ,mᵢ,nᵣ]
    end
    nothing
end

function packarray_A!(dest::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}}, src::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,K}}) where {Mᵣ,K,Mᵢ}
    # @inbounds for mᵢ ∈ axes(dest,3), k ∈ axes(dest,2), mᵣ ∈ axes(dest,1)
    @avx for mᵢ ∈ axes(dest,3), k ∈ axes(dest,2), mᵣ ∈ axes(dest,1)
        dest[mᵣ,k,mᵢ] = src[mᵣ,mᵢ,k]
    end
end
function packarray_B!(dest::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ},T}, src::AbstractStrideArray{Tuple{K,Nᵣ,Nᵢ},T}) where {Nᵣ,K,Nᵢ,T}
    # @inbounds for k ∈ axes(dest,2), nᵢ ∈ axes(dest,3), nᵣ ∈ axes(dest,1)
    @avx for k ∈ axes(dest,2), nᵢ ∈ axes(dest,3), nᵣ ∈ axes(dest,1)
        dest[nᵣ,k,nᵢ] = src[k,nᵣ,nᵢ]
    end
end
function packarray_B!(dest::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ},T}, src::AbstractStrideArray{Tuple{K,Nᵣ,Nᵢ},T}, nr) where {Nᵣ,K,Nᵢ,T}
    @avx for nᵢ ∈ axes(src,3), k ∈ axes(dest,2), nᵣᵢ ∈ axes(dest,1)
        dest[nᵣᵢ,k,nᵢ] = src[k,nᵣᵢ,nᵢ]
    end
    if !iszero(nr)
        nᵢ = size(dest,3)
        nᵣₛ = Static{nᵣ}()
        @avx for k ∈ axes(dest,2), nᵣᵢ ∈ 1:nᵣₛ
            dest[nᵣ,k,nᵢ] = nᵣᵢ > nᵣ ? zero(T) : src[k,nᵣ,nᵢ]
        end
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


    ri(x) = reinterpret(Int, x)

roundupnᵣ(x) = (xnr = x + nᵣ; xnr - (xnr % nᵣ))

function jmulh!(
    C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α = Val(1), β = Val(0)
# ) where {Tc, Ta, Tb}
) where {Ta, Tb, Tc}
    mc, kc, nc = matmul_params_val(Tc)
    jmulh!(C, A, B, α, β, mc, kc, nc)
end
function jmulh!(
    C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, ::Val{mc}, ::Val{kc}, ::Val{nc}
# ) where {Tc, Ta, Tb}
) where {Ta, Tb, Tc, mc, kc, nc}
    W = VectorizationBase.pick_vector_width(Tc)
    mᵣW = mᵣ * W
    M, K, N = matmul_sizes(C, A, B)

    num_n_iter = cld(N, nc)
    nreps_per_iter = N ÷ num_n_iter
    nreps_per_iter += nᵣ - 1
    ncrepetitions = nreps_per_iter ÷ nᵣ
    nreps_per_iter = nᵣ * ncrepetitions
    Niter = num_n_iter - 1
    Nrem = N - Niter * nreps_per_iter

    num_m_iter = cld(M, mc)
    mreps_per_iter = M ÷ num_m_iter
    mreps_per_iter += mᵣW - 1
    mcrepetitions = mreps_per_iter ÷ mᵣW
    mreps_per_iter = mᵣW * mcrepetitions
    Miter = num_m_iter - 1
    Mrem = M - Miter * mreps_per_iter
    
    #(iszero(Miter) && iszero(Niter)) && return loopmul!(C, A, B)
    num_k_iter = cld(K, kc)
    Krepetitions, Krem = divrem(K, num_k_iter)
    Krem += Krepetitions
    Kiter = num_k_iter - 1

    # @show mreps_per_iter, Krepetitions, nreps_per_iter
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
                Apacked_krem = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,1,false}(ptrL2, (Krem,mcrepetitions), (Krem*mᵣW,))
                Apmat_krem = PtrArray{Tuple{mᵣW,-1,-1},Ta,3,Tuple{1,mᵣW,-1},2,1,false}(gep(Aptr, (mo*mreps_per_iter, 0)), (mcrepetitions,Krem), Aptr.strides)
                packarray_A!(Apacked_krem, Apmat_krem)
                Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noff))
                Cx = first(Cptr.strides)
                Cpmat = PtrArray{Tuple{mᵣW,-1,nᵣ,-1},Tc,4,Tuple{1,mᵣW,-1,-1},2,2,false}(Cptr_off, (mcrepetitions,ncrepetitions), (Cx,Cx*nᵣ))
                loopmul!(Cpmat, Apacked_krem, Bpacked_krem, α, β)
            end
            # Mrem
            if Mrem > 0
                Apacked_mrem_krem = PtrMatrix(ptrL2, Mrem, Krem, VectorizationBase.align(Mrem, Tc))
                Apmat_mrem_krem = PtrMatrix(gesp(Aptr, (Miter*mreps_per_iter, 0)), Mrem, Krem)
                copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
                Cpmat_mrem = PtrArray{Tuple{-1,nᵣ,-1},Tc,3}(gesp(Cptr, (Miter*mreps_per_iter, Noff)), (Mrem, ncrepetitions))
                loopmul!(Cpmat_mrem, Apacked_mrem_krem, Bpacked_krem, α, β)
            end
            k = Krem
            for ko in 1:Kiter
                # pack kc x nc block of B
                Bpacked = PtrArray{Tuple{nᵣ,-1,-1},Tc,3,Tuple{1,nᵣ,-1},2,1,false}(ptrL3, (Krepetitions,ncrepetitions), (nᵣ * Krepetitions,))
                Bpmat = PtrArray{Tuple{-1,nᵣ,-1},Tb,3}(gesp(Bptr, (k, Noff)), (Krepetitions,ncrepetitions))
                packarray_B!(Bpacked, Bpmat)
                for mo in 0:Miter-1
                    # pack mc x kc block of A
                    Apacked = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,1,false}(ptrL2, (Krepetitions,mcrepetitions), ((mᵣW*kc),))
                    Aptr_off = gep(Aptr, (mo*mreps_per_iter, k))
                    Apmat = PtrArray{Tuple{mᵣW,-1,-1},Ta,3,Tuple{1,mᵣW,-1},2,1,false}(Aptr_off, (mcrepetitions,Krepetitions), Aptr.strides)
                    packarray_A!(Apacked, Apmat)
                    Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noff))
                    Cx = first(Cptr.strides)
                    Cpmat = PtrArray{Tuple{mᵣW,-1,nᵣ,-1},Tc,4,Tuple{1,mᵣW,-1,-1},2,2,false}(Cptr_off, (mcrepetitions,ncrepetitions), (Cx,Cx*nᵣ))
                    loopmul!(Cpmat, Apacked, Bpacked, α, Val(1))
                end
                # Mrem
                if Mrem > 0
                    Apacked_mrem = PtrMatrix(ptrL2, Mrem, Krepetitions, VectorizationBase.align(Mrem,Tc))
                    Apmat_mrem = PtrMatrix(gesp(Aptr, (Miter*mreps_per_iter, k)), Mrem, Krepetitions)
                    copyto!(Apacked_mrem, Apmat_mrem)
                    Cpmat_mrem = PtrArray{Tuple{-1,nᵣ,-1},Tc,3}(gesp(Cptr, (Miter*mreps_per_iter, Noff)), (Mrem, ncrepetitions))
                    loopmul!(Cpmat_mrem, Apacked_mrem, Bpacked, α, Val(1))
                end
                k += Krepetitions
            end
            Noff += nreps_per_iter
        end
        # Nrem
        if Nrem > 0
            # calcnrrem = false
            # Krem
            # pack kc x nc block of B
            ncreprem, ncrepremrem = divrem(Nrem, nᵣ)
            ncrepremc = ncreprem + !(iszero(ncrepremrem))
            lastBstride = nᵣ * Krem
            Bpacked_krem = PtrArray{Tuple{nᵣ,-1,-1},Tc,3,Tuple{1,nᵣ,-1},2,1,false}(ptrL3, (Krem,ncrepremc), (lastBstride,))
            Bpmat_krem = PtrArray{Tuple{-1,nᵣ,-1},Tb,3}(gesp(Bptr, (0, Noff)), (Krem,ncreprem))
            packarray_B!(Bpacked_krem, Bpmat_krem, ncrepremrem)
            Noffrem = Noff + ncreprem*nᵣ
            Bpacked_krem_nrem = PtrMatrix{-1,-1,Tc,1,nᵣ,2,0,false}(gep(ptrL3, lastBstride * ncreprem), (ncrepremrem,Krem), tuple())
            for mo in 0:Miter-1
                # pack mc x kc block of A
                Apacked_krem = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,1,false}(ptrL2, (Krem,mcrepetitions), (Krem*mᵣW,))
                Apmat_krem = PtrArray{Tuple{mᵣW,-1,-1},Ta,3,Tuple{1,mᵣW,-1},2,1,false}(gep(Aptr, (mo*mreps_per_iter, 0)), (mcrepetitions,Krem), Aptr.strides)
                packarray_A!(Apacked_krem, Apmat_krem)
                Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noff))
                Cx = first(Cptr.strides)
                Cpmat = PtrArray{Tuple{mᵣW,-1,nᵣ,-1},Tc,4,Tuple{1,mᵣW,-1,-1},2,2,false}(Cptr_off, (mcrepetitions,ncreprem), (Cx,Cx*nᵣ))
                loopmul!(Cpmat, Apacked_krem, Bpacked_krem, α, β)

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
                copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
                Cpmat_mrem = PtrArray{Tuple{-1,nᵣ,-1},Tc,3}(gesp(Cptr, (Miter*mreps_per_iter, Noff)), (Mrem, ncreprem))
                loopmul!(Cpmat_mrem, Apacked_mrem_krem, Bpacked_krem, α, β)

                if ncrepremrem > 0
                    # if calcnrrem && ncrepremrem > 0
                    Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mreps_per_iter, Noff + ncreprem * nᵣ)), Mrem, ncrepremrem)
                    loopmul!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem', α, β)
                end

            end
            k = Krem
            for ko in 1:Kiter
                # pack kc x nc block of B
                lastBstride = nᵣ * Krepetitions
                Bpacked = PtrArray{Tuple{nᵣ,-1,-1},Tc,3,Tuple{1,nᵣ,-1},2,1,false}(ptrL3, (Krepetitions,ncrepremc), (lastBstride,))
                Bpmat = PtrArray{Tuple{-1,nᵣ,-1},Tb,3}(gesp(Bptr, (k, Noff)), (Krepetitions,ncreprem))
                packarray_B!(Bpacked, Bpmat, ncrepremrem)
                Bpacked_nrem = PtrMatrix{-1,-1,Tc,1,nᵣ,2,0,false}(gep(ptrL3, lastBstride * ncreprem), (ncrepremrem,Krepetitions), tuple())
                for mo in 0:Miter-1
                    # pack mc x kc block of A
                    Apacked = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,(mᵣ*W),-1},2,1,false}(ptrL2, (Krepetitions,mcrepetitions), ((mᵣW*kc),))
                    Aptr_off = gep(Aptr, (mo*mreps_per_iter, k))
                    Apmat = PtrArray{Tuple{mᵣW,-1,-1},Ta,3,Tuple{1,mᵣW,-1},2,1,false}(Aptr_off, (mcrepetitions,Krepetitions), Aptr.strides)
                    packarray_A!(Apacked, Apmat)
                    Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noff))
                    Cx = first(Cptr.strides)
                    Cpmat = PtrArray{Tuple{mᵣW,-1,nᵣ,-1},Tc,4,Tuple{1,mᵣW,-1,-1},2,2,false}(Cptr_off, (mcrepetitions,ncreprem), (Cx,Cx*nᵣ))
                    loopmul!(Cpmat, Apacked, Bpacked, α, Val(1))
                    # if calcnrrem && ncrepremrem > 0
                    if ncrepremrem > 0
                        Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noffrem))
                        Cpmat_nrem = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,2,false}(Cptr_off, (mcrepetitions,ncrepremrem), (Cx,Cx*nᵣ))
                        loopmul!(Cpmat_nrem, Apacked, Bpacked_nrem, α, Val(1))
                    end

                end
                # Mrem
                if Mrem > 0
                    Apacked_mrem = PtrMatrix(ptrL2, Mrem, Krepetitions, VectorizationBase.align(Mrem,Tc))
                    Apmat_mrem = PtrMatrix(gesp(Aptr, (Miter*mreps_per_iter, k)), Mrem, Krepetitions)
                    copyto!(Apacked_mrem, Apmat_mrem)
                    Cpmat_mrem = PtrArray{Tuple{-1,nᵣ,-1},Tc,3}(gesp(Cptr, (Miter*mreps_per_iter, Noff)), (Mrem, ncreprem))
                    loopmul!(Cpmat_mrem, Apacked_mrem, Bpacked, α, Val(1))

                    # if calcnrrem && ncrepremrem > 0
                    if ncrepremrem > 0
                        Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mreps_per_iter, Noff + ncreprem * nᵣ)), Mrem, ncrepremrem)
                        loopmul!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem', α, Val(1))
                    end

                end
                k += Krepetitions
            end
        end
    end # GC.@preserve
    C
 end # function 



@inline function LinearAlgebra.mul!(
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


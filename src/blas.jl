
@inline cld_fast(x, y) = cld(x, y)
@inline function cld_fast(x::I, y) where {I <: Integer}
    x32 = x % UInt32; y32 = y % UInt32
    d = Base.udiv_int(x32, y32)
    (d * y32 == x32 ? d : d + one(UInt32)) % I
end
cld_fast(::StaticInt{N}, ::StaticInt{M}) where {N,M}= (StaticInt{N}() + StaticInt{M}() + One()) ÷ StaticInt{M}()
@inline function divrem_fast(x::I, y) where {I <: Integer}
    x32 = x % UInt32; y32 = y % UInt32
    d = Base.udiv_int(x32, y32)
    r = vsub(x32, vmul(d, y32))
    d % I, r % I
end
@inline divrem_fast(::StaticInt{x}, y::I) where {x, I <: Integer} = divrem_fast(x % I, y)
@inline div_fast(x::I, y::Integer) where {I <: Integer} = Base.udiv_int(x % UInt32, y % UInt32) % I
@inline div_fast(::StaticInt{x}, y::I) where {x, I <: Integer} = Base.udiv_int(x % UInt32, y % UInt32) % I
# @inline div_fast(x::I, ::StaticInt{x}) where {x, I <: Integer} = Base.udiv_int(x % UInt32, y % UInt32) % I
divrem_fast(::StaticInt{N}, ::StaticInt{M}) where {N,M}= divrem(StaticInt{N}(), StaticInt{M}())
div_fast(::StaticInt{N}, ::StaticInt{M}) where {N,M}= StaticInt{N}() ÷ StaticInt{M}()

function Base.copyto!(B::AbstractStrideArray{<:Any,<:Any,<:Any,N}, A::AbstractStrideArray{<:Any,<:Any,<:Any,N}) where {N}
    @avx for I ∈ eachindex(A, B)
        B[I] = A[I]
    end
    B
end
@inline zstridedpointer(A) = VectorizationBase.zero_offsets(stridedpointer(A))
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

# function matmul_params(::Type{T}) where {T}
#    W = VectorizationBase.pick_vector_width(T)
#    L₁ratio = L₁ ÷ (nᵣ * sizeof(T))
#    kc = round(Int, 0.65L₁ratio)
#    mcrep =  5L₂ ÷ (8kc * sizeof(T) * mᵣ * W)
#    ncrep = L₃ ÷ (kc * sizeof(T) * nᵣ)
#    mc = mcrep * mᵣ * W
#    nc = round(Int, 0.4ncrep * nᵣ) #* VectorizationBase.NUM_CORES
#    mc, kc, nc
# end

# assume L₂ is inclusive of L₁
# const INCLUSIVE_L₂ = VectorizationBase.CACHE_COUNT[2] > 0 ? true : nothing
# heuristically assume that if the L₃-per core is at least twice as large as the L₂, that it is inclusive of the L₂
const INCLUSIVE_L₃ = VectorizationBase.CACHE_COUNT[3] > 0 ? ((VectorizationBase.CACHE_SIZE[2] * VectorizationBase.CACHE_COUNT[2] / (VectorizationBase.CACHE_SIZE[3] * VectorizationBase.CACHE_COUNT[3])) < 0.5) : nothing

function matmul_params(::Type{T}) where {T}
    Mᵣ = StaticInt{mᵣ}()
    Nᵣ = StaticInt{nᵣ}()
    W = VectorizationBase.pick_vector_width_val(T)
    MᵣW = Mᵣ * W
    mc = MᵣW * (VectorizationBase.REGISTER_SIZE === 64 ? StaticInt{4}() : StaticInt{9}()) # TODO: make this smarter/less heuristic
    L₁ = something(core_cache_size(T, Val(1)), StaticInt{32768}() ÷ static_sizeof(T))
    L₂ = something(core_cache_size(T, Val(2)), StaticInt{262144}() ÷ static_sizeof(T))
    ΔL₂₁ = L₂ - L₁ # assume caches are inclusive
    kc = ((StaticInt{795}() * ΔL₂₁) ÷ StaticInt{1024}() - StaticInt{4980}()) ÷ mc
    L₃ = if INCLUSIVE_L₃ === nothing
        StaticInt{3145728}() ÷ static_sizeof(T)
    elseif INCLUSIVE_L₃
        something(cache_size(T, Val(3)), StaticInt{3145728}() ÷ static_sizeof(T)) - L₂
    else
        something(cache_size(T, Val(3)), StaticInt{3145728}() ÷ static_sizeof(T))
    end
    nc = ((((StaticInt{132}() * L₃) ÷ StaticInt{125}()) - StaticInt{256651}()) ÷ (kc * Nᵣ)) * Nᵣ
    mc, kc, nc
end
# @generated matmul_params(::Type{T}) where {T} = matmul_params_calc(T, mᵣ, nᵣ)
# @generated function matmul_params_static(::Type{T}) where {T}
#     mc, kc, nc = matmul_params_calc(T)
#     Expr(:tuple, static_expr(mc), static_expr(kc), static_expr(nc))
# end

# function pack_B(Bptr, kc, nc, ::Type{Tb}, koffset::Int, noffset::Int) where {Tb}
#     # Bpacked = PtrMatrix(threadlocal_L3CACHE_pointer(Tb), kc, nc, kc*sizeof(Tb))
#     Bpacked = PtrMatrix(threadlocal_L3CACHE_pointer(Tb), kc, nc, VectorizationBase.align(kc,Tb)*sizeof(Tb))
#     Bpmat = PtrMatrix(gesp(Bptr, (koffset, noffset)), kc, nc)
#     copyto!(Bpacked, Bpmat)
#     Bpacked
# end
# function pack_A(Aptr, mc, kc, ::Type{Ta}, moffset::Int, koffset::Int) where {Ta}
#     # Apacked = PtrArray{Tuple{mᵣW,-1,-1},Ta,3,Tuple{1,mᵣW,-1},2,1,false}(threadlocal_L2CACHE_pointer(Ta), (kc,mcreps), ((mᵣW*kc),))
#     # Apacked = PtrMatrix(threadlocal_L2CACHE_pointer(Ta), mc, kc, mc*sizeof(Ta))
#     Apacked = PtrMatrix(threadlocal_L2CACHE_pointer(Ta), mc, kc, VectorizationBase.align(mc, Ta)*sizeof(Ta))
#     Apmat = PtrMatrix(gesp(Aptr, (moffset, koffset)), mc, kc)
#     # @show (reinterpret(Int, pointer(Apmat)) - reinterpret(Int,pointer(Aptr))) >>> 3
#     # copyto!(Apacked, Apmat)
#     Apacked, Apmat
# end

@generated function dense_dims_subset(::DenseDims{D}, ::StrideRank{R}) where {D,R}
    t = Expr(:tuple)
    for n in eachindex(R)
        push!(t.args, D[n] & (R[n] == 1))
    end
    Expr(:call, Expr(:curly, :DenseDims, t))
end
@inline zrange(N) = Zero():N-One()

"""
Only packs `A`. Primitively does column-major packing: it packs blocks of `A` into a column-major temporary.
"""
function jmulpackAonly!(
    C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, ::StaticInt{mc}, ::StaticInt{kc}, ::StaticInt{nc}, (Ma, Ka, Na) = matmul_axes(C, A, B)
) where {Tc, Ta, Tb, mc, kc, nc}
    W = VectorizationBase.pick_vector_width_val(Tc)
    mᵣW = StaticInt{mᵣ}() * W

    M = static_length(Ma);
    K = static_length(Ka);
    N = static_length(Na);
    
    num_m_iter = cld_fast(M, StaticInt{mc}())
    _mreps_per_iter = div_fast(M, num_m_iter) + mᵣW - One()
    mcrepetitions = _mreps_per_iter ÷ mᵣW
    mreps_per_iter = mᵣW * mcrepetitions
    Miter = num_m_iter - One()
    Mrem = M - Miter * mreps_per_iter
    #(iszero(Miter) && iszero(Niter)) && return loopmul!(C, A, B)
    num_k_iter = cld_fast(K, StaticInt{kc}())
    kreps_per_iter = vadd((div_fast(K, num_k_iter)), 3) & -4
    Krem = vsub(K, vmul(vsub(num_k_iter, One()), kreps_per_iter))
    Kiter = vsub(num_k_iter, One())
    # LCACHEARRAY = core_cache_buffer(Ta, Val(2))
    Aptr = zstridedpointer(A)
    Bptr = zstridedpointer(B)
    Cptr = zstridedpointer(C)
    _Mrem, _mreps_per_iter = promote(Mrem, mreps_per_iter)
    # Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B);
    # GC.@preserve Cb Ab Bb begin
        # # Krem
        # # pack kreps_per_iter x nc block of B
        # Bpacked = PtrArray(Bptr, (Krem, N), dense_dims_subset(dense_dims(B), stride_rank(B)))
        # # Bpacked = PtrArray(B)
        # moffset = 0
        # for mo in 0:Miter
        #     msize = ifelse(mo == Miter, _Mrem, _mreps_per_iter)
        #     # pack mreps_per_iter x kreps_per_iter block of A
            
        #     Asubset = PtrArray(gesp(Aptr, (moffset, Zero())), (msize, Krem), dense_dims_subset(dense_dims(A), stride_rank(A)))

        #     Cpmat = PtrArray(gesp(Cptr, (moffset, Zero())), (msize, N), dense_dims_subset(dense_dims(C), stride_rank(C)))
        #     packaloopmul!(Cpmat, Asubset, Bpacked, α, β, (zrange(msize), zrange(Krem), zrange(N)))
        #     moffset += mreps_per_iter
        # end
        # koffset = Krem
        # for ko in 1:Kiter
        #     # pack kreps_per_iter x nc block of B
        #     Bpacked = PtrArray(gesp(Bptr, (koffset, Zero())), (kreps_per_iter, N), dense_dims_subset(dense_dims(B), stride_rank(B)))
        #     moffset = 0
        #     for mo in 0:Miter
        #         msize = ifelse(mo == Miter, _Mrem, _mreps_per_iter)
        #         # pack mreps_per_iter x kreps_per_iter block of A
        #         Asubset = PtrArray(gesp(Aptr, (moffset, koffset)), (msize, kreps_per_iter), dense_dims_subset(dense_dims(A), stride_rank(A)))
                
        #         Cpmat = PtrArray(gesp(Cptr, (moffset, Zero())), (msize, N), dense_dims_subset(dense_dims(C), stride_rank(C)))
        #         packaloopmul!(Cpmat, Asubset, Bpacked, α, One(), (zrange(msize), zrange(kreps_per_iter), zrange(N)))
        #         moffset += mreps_per_iter
        #     end
        #     koffset += kreps_per_iter
        # end
    koffset = 0
    _Krem, _kreps_per_iter = promote(Krem, kreps_per_iter)
    # @show typeof(C)
    # @show typeof(A)
    # @show typeof(B)
    for ko ∈ 0:Kiter
        ksize = ifelse(ko == 0, _Krem, _kreps_per_iter)
        # _β = ifelse(ko == 0, convert(Tc, β), one(Tc))
        Bpacked = PtrArray(gesp(Bptr, (koffset, Zero())), (ksize, N), dense_dims_subset(dense_dims(B), stride_rank(B)))

        moffset = 0
        for mo in 0:Miter
            msize = ifelse(mo == Miter, _Mrem, _mreps_per_iter)
            # pack mreps_per_iter x kreps_per_iter block of A
            Asubset = PtrArray(gesp(Aptr, (moffset, koffset)), (msize, ksize), dense_dims_subset(dense_dims(A), stride_rank(A)))
            
            Cpmat = PtrArray(gesp(Cptr, (moffset, Zero())), (msize, N), dense_dims_subset(dense_dims(C), stride_rank(C)))
            # packaloopmul!(Cpmat, Asubset, Bpacked, α, _β, (zrange(msize), zrange(ksize), zrange(N)))
            if ko == 0
                packaloopmul!(Cpmat, Asubset, Bpacked, α, β, (zrange(msize), zrange(ksize), zrange(N)))
            else
                packaloopmul!(Cpmat, Asubset, Bpacked, α, One(), (zrange(msize), zrange(ksize), zrange(N)))
            end
            moffset += mreps_per_iter
        end
        koffset += ksize
    end
    # end # GC.@preserve
    nothing
end
"""
Packs both arrays `A` and `B`.
Primitely packs both `A` and `B` into column major temporaries.

Column-major `B` is preferred over row-major, because without packing the stride across `k` iterations of `B` becomes excessive, and without `nᵣ` being a multiple of the cacheline size, we would fail to make use of 100% of the loaded cachelines.
Unfortunately, using column-major `B` does mean that we are starved on integer registers within the macrokernel.

Once `LoopVectorization` adds a few features to make it easy to abstract away tile-major memory layouts, we will switch to those, probably improving performance for larger matrices.
"""
function jmulpackAB!(
    C::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix, α, β, ::StaticInt{mc}, ::StaticInt{kc}, ::StaticInt{nc}, (Ma, Ka, Na) = matmul_axes(C, A, B)
) where {T, mc, kc, nc}
    W = VectorizationBase.pick_vector_width_val(T)
    mᵣW = StaticInt{mᵣ}() * W

    M = static_length(Ma);
    K = static_length(Ka);
    N = static_length(Na);
    
    num_n_iter = cld_fast(N, StaticInt{nc}())
    _nreps_per_iter = div_fast(N, num_n_iter) + StaticInt{nᵣ}() - One()
    ncrepetitions = div_fast(_nreps_per_iter, StaticInt{nᵣ}())
    nreps_per_iter = StaticInt{nᵣ}() * ncrepetitions
    Niter = num_n_iter - One()
    Nrem = N - Niter * nreps_per_iter

    num_m_iter = cld_fast(M, StaticInt{mc}())
    _mreps_per_iter = div_fast(M, num_m_iter) + mᵣW - One()
    mcrepetitions = div_fast(_mreps_per_iter, mᵣW)
    mreps_per_iter = mᵣW * mcrepetitions
    Miter = num_m_iter - One()
    Mrem = M - Miter * mreps_per_iter
    
    num_k_iter = cld_fast(K, StaticInt{kc}())
    kreps_per_iter = ((div_fast(K, num_k_iter)) + 3) & -4
    Krem = K - (num_k_iter - One()) * kreps_per_iter
    Kiter = num_k_iter - One()

    # @show mreps_per_iter, kreps_per_iter, nreps_per_iter
    # @show Miter, Kiter, Niter
    # @show Mrem, Krem, Nrem
    
    Aptr = stridedpointer(A)
    Bptr = stridedpointer(B)
    Cptr = stridedpointer(C)
    # ptrL3 = threadlocal_L3CACHE_pointer(T)
    # ptrL2 = threadlocal_L2CACHE_pointer(T)
    # L2CACHEARRAY = core_cache_buffer(Ta, Val(2))
    # L3CACHEARRAY = core_cache_buffer(Tb, Val(3))
    
    Aptr = zstridedpointer(A)
    Bptr = zstridedpointer(B)
    Cptr = zstridedpointer(C)
    # L3ptr = Base.unsafe_convert(Ptr{Tb}, pointer(BCACHE) + (Threads.threadid()-1)*BSIZE*8)
    L3ptr = Base.unsafe_convert(Ptr{T}, BCACHE)
    noffset = 0
    _Nrem, _nreps_per_iter = promote(Nrem, nreps_per_iter)
    _Mrem, _mreps_per_iter = promote(Mrem, mreps_per_iter)
    _Krem, _kreps_per_iter = promote(Krem, kreps_per_iter)
    # Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B);
    # GC.@preserve Cb Ab Bb BCACHE begin
    GC.@preserve BCACHE begin
        for no in 0:Niter
            # Krem
            # pack kc x nc block of B
            nsize = ifelse(no == Niter, _Nrem, _nreps_per_iter)
            koffset = 0
            for ko ∈ 0:Kiter
                ksize = ifelse(ko == 0, _Krem, _kreps_per_iter)
                # _β = ifelse(ko == 0, convert(T, β), one(T))
                Bsubset2 = PtrArray(gesp(Bptr, (koffset, noffset)), (ksize, nsize), dense_dims_subset(dense_dims(B), stride_rank(B)))
                Bpacked2 = ptrarray0(L3ptr, (ksize, nsize))
                # @show offsets(Bpacked2), offsets(Bsubset2)
                copyto!(Bpacked2, Bsubset2)
                # @show no, ko, (all(isone, Bpacked2),all(isone, Bsubset2)) (noffset, koffset), (kreps_per_iter, nsize) size(Bpacked2) axes(Bpacked2)
                # findall(!isone, Bpacked2)
                # copyto!(view(Bpacked2,:,:), view(Bsubset2,:,:))
                # Bpacked = pack_B(Bptr, kreps_per_iter, nsize, T, koffset, noffset)
                moffset = 0
                for mo in 0:Miter
                    msize = ifelse(mo == Miter, _Mrem, _mreps_per_iter)
                    # pack mreps_per_iter x kreps_per_iter block of A
                    Asubset = PtrArray(gesp(Aptr, (moffset, koffset)), (msize, ksize), dense_dims_subset(dense_dims(A), stride_rank(A)))

                    Cpmat = PtrArray(gesp(Cptr, (moffset, noffset)), (msize, nsize), dense_dims_subset(dense_dims(C), stride_rank(C)))
                    # packaloopmul!(Cpmat, Asubset, Bpacked2, α, _β, (zrange(msize), zrange(ksize), zrange(nsize)))
                    if ko == 0
                        packaloopmul!(Cpmat, Asubset, Bpacked2, α, β, (zrange(msize), zrange(ksize), zrange(nsize)))
                    else
                        packaloopmul!(Cpmat, Asubset, Bpacked2, α, One(), (zrange(msize), zrange(ksize), zrange(nsize)))
                    end
                    moffset += mreps_per_iter
                end
                koffset += ksize

                
            # Bsubset = PtrArray(gesp(Bptr, (Zero(), noffset)), (Krem, nsize), dense_dims_subset(dense_dims(B), stride_rank(B)))
            # Bpacked = ptrarray0(L3ptr, (Krem, nsize))
            # # @show offsets(Bpacked), offsets(Bsubset)
            # copyto!(Bpacked, Bsubset)
            # # @show eltype(Bpacked, Bsubset)
            # # @show no, (all(isone, Bpacked),all(isone, Bsubset)) noffset (Krem,nsize) size(Bpacked) axes(Bpacked)
            # # findall(!isone, Bpacked)
            # # copyto!(view(Bpacked,:,:), view(Bsubset,:,:))
            # # Bpacked = pack_B(Bptr, Krem, nsize, T, 0, noffset)
            # moffset = 0
            # for mo in 0:Miter
            #     msize = ifelse(mo == Miter, _Mrem, _mreps_per_iter)
            #     # pack mreps_per_iter x kreps_per_iter block of A
            #     Asubset = PtrArray(gesp(Aptr, (moffset, Zero())), (msize, Krem), dense_dims_subset(dense_dims(A), stride_rank(A)))

            #     Cpmat = PtrArray(gesp(Cptr, (moffset, noffset)), (msize, nsize), dense_dims_subset(dense_dims(C), stride_rank(C)))
            #     packaloopmul!(Cpmat, Asubset, Bpacked, α, β, (zrange(msize), zrange(Krem), zrange(nsize)))
            #     moffset += mreps_per_iter
            # end
            # koffset = Krem
            # for ko in 1:Kiter
            #     # pack kreps_per_iter x nc block of B
            #     Bsubset2 = PtrArray(gesp(Bptr, (koffset, noffset)), (kreps_per_iter, nsize), dense_dims_subset(dense_dims(B), stride_rank(B)))
            #     Bpacked2 = ptrarray0(L3ptr, (kreps_per_iter, nsize))
            #     # @show offsets(Bpacked2), offsets(Bsubset2)
            #     copyto!(Bpacked2, Bsubset2)
            #     # @show no, ko, (all(isone, Bpacked2),all(isone, Bsubset2)) (noffset, koffset), (kreps_per_iter, nsize) size(Bpacked2) axes(Bpacked2)
            #     # findall(!isone, Bpacked2)
            #     # copyto!(view(Bpacked2,:,:), view(Bsubset2,:,:))
            #     # Bpacked = pack_B(Bptr, kreps_per_iter, nsize, T, koffset, noffset)
            #     moffset = 0
            #     for mo in 0:Miter
            #         msize = ifelse(mo == Miter, _Mrem, _mreps_per_iter)
            #         # pack mreps_per_iter x kreps_per_iter block of A
            #         Asubset = PtrArray(gesp(Aptr, (moffset, koffset)), (msize, kreps_per_iter), dense_dims_subset(dense_dims(A), stride_rank(A)))

            #         Cpmat = PtrArray(gesp(Cptr, (moffset, noffset)), (msize, nsize), dense_dims_subset(dense_dims(C), stride_rank(C)))
            #         packaloopmul!(Cpmat, Asubset, Bpacked2, α, StaticInt{1}(), (zrange(msize), zrange(kreps_per_iter), zrange(nsize)))
            #         moffset += mreps_per_iter
            #     end
            #     koffset += kreps_per_iter
            end
            noffset += nreps_per_iter
        end
    end # GC.@preserve
    nothing
end

@inline contiguousstride1(A) = ArrayInterface.contiguous_axis(A) === ArrayInterface.Contiguous{1}()
@inline firstbytestride(A::AbstractStrideArray) = bytestride(A, One())
# @inline firstbytestride(A::PermutedDimsArray) = LinearAlgebra.stride1(A)
# @inline firstbytestride(A::Adjoint{<:Any,<:AbstractMatrix}) = stride(parent(A), 2)
# @inline firstbytestride(A::Transpose{<:Any,<:AbstractMatrix}) = stride(parent(A), 2)
# @inline firstbytestride(::Any) = typemax(Int)

@inline function vectormultiple(x, ::Type{Tc}, ::Type{Ta}) where {Tc,Ta}
    Wc = VectorizationBase.pick_vector_width_val(Tc) * static_sizeof(Ta) - One()
    iszero(x & Wc)
end
@inline function dontpack(ptrA::Ptr{Ta}, M, K, Xa, ::StaticInt{mc}, ::StaticInt{kc}, ::Type{Tc}) where {mc, kc, Tc, Ta}
    mc_mult = VectorizationBase.AVX512F ? 73 : 53
    (mc_mult > M) || (vectormultiple(Xa, Tc, Ta) && ((M * K) ≤ (mc * kc)) && iszero(reinterpret(Int, ptrA) & (VectorizationBase.REGISTER_SIZE - 1)))
end

"""
  jmul!(C, A, B[, α = 1, β = 0])

Calculates `C = α * (A * B) + β * C` in place.

A single threaded matrix-matrix-multiply implementation.
Supports dynamically and statically sized arrays.

Organizationally, `jmul!` checks the arrays properties to try and dispatch to an appropriate implementation.
If the arrays are small and statically sized, it will dispatch to an inlined multiply.

Otherwise, based on the array's size, whether they are transposed, and whether the columns are already aligned, it decides to not pack at all, to pack only `A`, or to pack both arrays `A` and `B`.
"""
@inline function jmul!(
    C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, ::StaticInt{mc}, ::StaticInt{kc}, ::StaticInt{nc}, (Ma, Ka, Na) = matmul_axes(C, A, B)
) where {Tc, Ta, Tb, mc, kc, nc}
    pA = PtrArray(A)
    pB = PtrArray(B)
    pC = PtrArray(C)
    M = static_length(Ma); K = static_length(Ka); N = static_length(Na);
    Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B);
    GC.@preserve Cb Ab Bb begin
        if VectorizationBase.CACHE_SIZE[2] === nothing || ((nᵣ ≥ N) || (contiguousstride1(pA) && dontpack(pointer(pA), M, K, bytestride(pA,StaticInt{2}()), StaticInt{mc}(), StaticInt{kc}(), Tc)))
            loopmul!(pC, pA, pB, α, β, (Ma,Ka,Na))
        elseif VectorizationBase.CACHE_SIZE[3] === nothing || (((contiguousstride1(pB) && (kc * nc ≥ K * N))) || firstbytestride(pB) ≤ 1600)
            # println("Pack A mul")
            jmulpackAonly!(pC, pA, pB, α, β, StaticInt{mc}(), StaticInt{kc}(), StaticInt{nc}(), (Ma,Ka,Na))
        else
            # println("Pack A and B mul")
            jmulpackAB!(pC, pA, pB, α, β, StaticInt{mc}(), StaticInt{kc}(), StaticInt{nc}(), (Ma,Ka,Na))
        end
    end
    return C
end # function

@inline function jmul!(C::LinearAlgebra.Adjoint{<:Real}, A::AbstractMatrix, B::AbstractMatrix, α, β, ::StaticInt{mc}, ::StaticInt{kc}, ::StaticInt{nc}) where {mc,kc,nc}
    jmul!(C', B', A', α, β, StaticInt{mc}(), StaticInt{kc}(), StaticInt{nc}())
    C
end
@inline function jmul!(
    C::AbstractStrideArray{S,D,T,2,2},
    A::AbstractMatrix,
    B::AbstractMatrix
) where {S, D, T}
    jmul!(C', B', A')
    C
end

struct ThreadRun
    id::UInt32
    nthread::UInt32
end
ThreadRun(i::Int, n::Int) = ThreadRun(i % UInt32, n % UInt32)

struct LoopMulClosure{P,TC,TA,TB,Α,Β,M,K,N}
    C::TC
    A::TA
    B::TB
    α::Α # \Alpha
    β::Β # \Beta
    Maxis::M
    Kaxis::K
    Naxis::N
end
function LoopMulClosure{false}(
    C::TC, A::TA, B::TB, α::Α, β::Β, Maxis::M, Kaxis::K, Naxis::N
) where {TC<:AbstractStridedPointer,TA<:AbstractStridedPointer,TB<:AbstractStridedPointer,Α,Β,M,K,N}
    LoopMulClosure{false,TC,TA,TB,Α,Β,M,K,N}(C, A, B, α, β, Maxis, Kaxis, Naxis)
end
function LoopMulClosure{true}(
    C::TC, A::TA, B::TB, α::Α, β::Β, Maxis::M, Kaxis::K, Naxis::N
) where {TC,TA,TB,Α,Β,M,K,N}
    LoopMulClosure{true,TC,TA,TB,Α,Β,M,K,N}(C, A, B, α, β, Maxis, Kaxis, Naxis)
end
function LoopMulClosure{false}(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β, M, K, N) # if not packing, discard `PtrArray` wrapper
    LoopMulClosure{false}(stridedpointer(C), stridedpointer(A), stridedpointer(B), α, β, M, K, N)
end
# function LoopMulClosure{true}(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β, M, K, N)
#     LoopMulClosure{true}(C, A, B, α, β, M, K, N)
# end
(m::LoopMulClosure{false})() = loopmul!(m.C, m.A, m.B, m.α, m.β, (m.Maxis, m.Kaxis, m.Naxis))
# (m::LoopMulClosure{true})() = jmulpackAonly!(m.C, m.A, m.B, m.α, m.β, (m.Maxis, m.Kaxis, m.Naxis))
function (m::LoopMulClosure{true,TC})() where {S,D,T,TC <: AbstractStrideArray{S,D,T}}
    Mc,Kc,Nc = matmul_params(T)
    jmulpackAonly!(m.C, m.A, m.B, m.α, m.β, Mc, Kc, Nc, (m.Maxis, m.Kaxis, m.Naxis))
end

struct PackAClosure{TC,TA,TB,Α,Β,M,K,N,T}
    # lmc::LoopMulClosure{TC,TA,TB,Α,Β,M,K,N}
    C::TC
    A::TA
    B::TB
    α::Α # \Alpha
    β::Β # \Beta
    Maxis::M
    Kaxis::K
    Naxis::N
    tasks::T
end
# function PackAClosure(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β, M, K, N, tasks)
#     PackAClosure(stridedpointer(C), stridedpointer(A), stridedpointer(B), α, β, M, K, N, tasks)
# end
function (m::PackAClosure{TC})() where {T,TC<:AbstractStridedPointer{T}}
    Mc,Kc,Nc = matmul_params(T)
    # Ma = m.Maxis; Ka = m.Kaxis; Na = m.Naxis
    jmultpackAonly!(m.C, m.A, m.B, m.α, m.β, Mc, Kc, Nc, Ma, Ka, Na, m.tasks)
end

function jmult!(C::AbstractMatrix{T}, A, B, α = One(), β = Zero(), (Ma,Ka,Na) = matmul_axes(C,A,B), nthread = _nthreads()) where {T}
    Mc,Kc,Nc = matmul_params(T)
    jmult!(C, A, B, α, β, Mc, Kc, Nc, (Ma,Ka,Na), nthread)
end

function jmult!(C::AbstractMatrix{T}, A, B, α, β, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}, (Ma,Ka,Na) = matmul_axes(C,A,B), nthread = _nthreads()) where {Mc,Kc,Nc,T}
    Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B)
    GC.@preserve Cb Ab Bb begin
        # Base.unsafe_convert(Ptr{T}, BCACHE)
        _jmult!(zero_offsets(PtrArray(C)), zero_offsets(PtrArray(A)), zero_offsets(PtrArray(B)), α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), (Ma,Ka,Na), nthread, Val{VectorizationBase.CACHE_COUNT[3]}(), nothing)
    end
    return C
end

function gcd_fast(a::T, b::T) where {T<:Base.BitInteger}
    za = trailing_zeros(a)
    zb = trailing_zeros(b)
    k = min(za, zb)
    u = unsigned(abs(a >> za))
    v = unsigned(abs(b >> zb))
    while u != v
        if u > v
            u, v = v, u
        end
        v -= u
        v >>= trailing_zeros(v)
    end
    r = u << k
    r % T
end
function divide_blocks(M, N, ::StaticInt{Mb}, ::StaticInt{Nb}, nspawn) where {Mb,Nb}
    Mfull, Mrem = divrem_fast(M, Mb)
    Mtotal = Mfull + (Mrem > 0)

    Miter = gcd_fast(nspawn, Mtotal)
    nspawn = div_fast(nspawn, Miter)

    
    Nfull, Nrem = divrem_fast(N, Nb)
    Ntotal = Nfull + (Nrem > 0)
    # Niter = gcd_fast(nspawn, Ntotal)
    Niter = cld_fast(Ntotal, cld_fast(Ntotal, nspawn))
    return Miter, Niter
end

function _jmult!(C::AbstractMatrix{T}, A, B, α, β, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}, (Ma,Ka,Na), nthread, ::Val{CC3}, bcache_ptr) where {T,Mc,Kc,Nc,CC3}
    M = static_length(Ma); K = static_length(Ka); N = static_length(Na);

    Mᵣ = StaticInt{mᵣ}(); Nᵣ = StaticInt{nᵣ}()
    W = VectorizationBase.pick_vector_width_val(T)
    MᵣW = Mᵣ*W
    nkern = cld_fast(M * N,  Mᵣ * Nᵣ)

    # Assume 3 μs / spawn cost
    # 15W approximate GFLOPS target
    # 22500 = 3e-6μ/spawn * 15 / 2e-9
    L = StaticInt{22500}() * W
    MKN = M*K*N
    suggested_threads = cld_fast(MKN, L)
    nspawn = min(nthread, suggested_threads)
    if nspawn ≤ 1
        # We convert to `PtrArray`s here to reduce recompilation
        loopmul!(zstridedpointer(C), zstridedpointer(A), zstridedpointer(B), α, β, (Ma,Ka,Na))
        return
    end
    # Approach:
    # Check if we want to pack B
    #    if not, check if we want to pack A
    #       if not, can we divide `N` into `nspawn` pieces? If not, also split `M`.
    #       if we do pack A, we want to maximize re-use of `A`-packs, so can we carve `M` into `nspawn` pieces? If not, also split `N`
    #    if so, check if `CACHE_COUNT[3]` > 1.
    #       if so, subdivide `B` and `C` into `min(CACHE_COUNT[3], nspawn)` pieces, spawn per, and pass `BCACHE` pointer to separate regions
    #    if so but `CACHE_COUNT[3] ≤ 1`, check if we can divide M into `nspawn` pieces
    #       if so, do threading with `packamuls`, one packed-B at a time
    #       if not, also divide `N`, and correspondingly decrease `Nc`
    # TODO: implement packing `B`
    do_not_pack_b = true#(contiguousstride1(B) && (kc * nc ≥ K * N)) | (firstbytestride(B) > 1600)
    tasks = _preallocated_tasks()
    if do_not_pack_b
        do_not_pack_a = VectorizationBase.CACHE_SIZE[2] === nothing || ((nᵣ ≥ N) || (contiguousstride1(A) && dontpack(pointer(A), M, K, bytestride(A,StaticInt{2}()), StaticInt{M}(), StaticInt{Kc}(), T)))
        if do_not_pack_a
            Mblocks, Nblocks = divide_blocks(M, N, MᵣW, Nᵣ, nspawn)
            _nspawn_m1 = Mblocks * Nblocks - One()
            Mbsize = cld_fast(cld_fast(M, Mblocks), W) * W
            Nbsize = cld_fast(N, Nblocks)
            nlower = 0
            nupper = Nbsize-1
            tnum = 0
            for n ∈ 0:Nblocks-1
                nrange = nlower:min(N,nupper)
                mlower = 0
                mupper = Mbsize-1
                _B = stridedpointer(zview(B, :, nrange))
                for m ∈ 0:Mblocks-1
                    mrange = mlower:min(M,mupper)
                    _C = stridedpointer(zview(C, mrange, nrange))
                    _A = stridedpointer(zview(A, mrange, :))
                    if tnum == _nspawn_m1
                        loopmul!(_C, _A, _B, α, β, Zero():static_length(mrange)-One(), Ka, Zero():static_length(nrange)-One())
                        foreach(wait, view(tasks, Base.OneTo(_nspawn_m1)))
                        return
                    end
                    t = Task(LoopMulClosure{false}(_C, _A, _B, α, β, Zero():static_length(mrange)-One(), Ka, Zero():static_length(nrange)-One()))
                    t.sticky = false
                    schedule(t)
                    tasks[(tnum += 1)] = t
                    mlower += Mbsize
                    mupper += Mbsize
                end
                nlower += Nbsize
                nupper += Nbsize
            end
        else# do pack A
            Mreps = cld_fast(M, MᵣW)
            # spawn_per_mrep is how many processes we'll spawn here
            spawn_per_mrep = min(cld_fast(nspawn, Mreps), N) # Safety: don't spawn more than `N`
            if spawn_per_mrep > 1
                nspawn_per = cld(nspawn, spawn_per_mrep)
                spawn_start = spawn_per_mrep
                Nper = cld(N, spawn_per_mrep)
                Nstart = 0
                for i ∈ 1:spawn_per_mrep-1
                    tasks_view = view(tasks, spawn_start:spawn_start+nspawn_per-2) # tasks are to be 1 shorter than number of threads running; "main" task runs the last set.
                    nrange = Nstart:Nstart+Nper-1
                    # t = Task(PackAClosure(zview(C, :, nrange), A, zview(B, :, nrange), α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), Ma, Ka, Zero():Nper, task_view))
                    t = Task(PackAClosure(zview(C, :, nrange), A, zview(B, :, nrange), α, β, Ma, Ka, Zero():Nper, task_view))
                    t.sticky = false
                    schedule(t)
                    tasks[i] = t
                    Nstart += Nper
                end
                tasks_view = view(tasks, 1:nspawn-1)
                nrange = Nstart:N-One()
                jmultpackAonly!(zview(C, :, nrange), A, zview(B, :, nrange), α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), Ma, Ka, Zero():length(nrange)-One(), tasks_view)
                foreach(wait, view(tasks, Base.OneTo(spawn_per_mrep-1))) # wait for spawned `jmultpackAonly!`s
            else
                tasks_view = view(tasks, 1:nspawn-1)
                jmultpackAonly!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), Ma, Ka, Na, tasks_view)
            end
            return
        end
    elseif CC3 > 1
        cc3 = min(nspawn, CC3)
        for c ∈ Base.OneTo(cc3)
            # @spawn begin
                
            # end
        end
        return C
    else
        
    end
end
function jmultpackAonly!(C::AbstractMatrix{T}, A, B, α, β, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}, Maxis, Kaxis, Naxis, tasks) where {T,Mc,Kc,Nc}
    to_spawn = length(tasks)
    total_threads = to_spawn + 1
    
    M = static_length(Maxis)

    W = VectorizationBase.pick_vector_width_val(T)
    _Mblock = cld_fast(M, total_threads)
    Mblock = cld_fast(_Mblock, W) * W
    
    Mstart = 0
    for i ∈ 1:to_spawn
        mrange = Mstart:Mstart + Mblock - 1
        # t = Task(LoopMulClosure{true}(zview(C, mrange, :), zview(A, mrange, :), B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), Zero():Mblock-One(), Kaxis, Naxis))
        t = Task(LoopMulClosure{true}(zview(C, mrange, :), zview(A, mrange, :), B, α, β, Zero():Mblock-One(), Kaxis, Naxis))
        t.sticky = false
        schedule(t)
        tasks[i] = t
        Mstart += Mblock
    end
    mrange = Mstart:M-1
    jmulpackAonly!(zview(C, mrange, :), zview(A, mrange, :), B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), (Zero():length(mrange)-One(), Kaxis, Naxis))
    foreach(wait, tasks)
end

# function jmult!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, ::StaticInt{mc}, ::StaticInt{kc}, ::StaticInt{nc}) where {Tc, Ta, Tb, mc, kc, nc, log2kc}
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
#                 Bpacked_krem = pack_B_krem(Bptr, StaticInt{kc}(), StaticInt{nc}(), Tb, Krem, no)
#                 Base.Threads.@sync begin
#                     for mo in 0:Miter-1
#                         # pack mc x kc block of A
#                         Base.Threads.@spawn begin
#                             # begin
#                             Apacked_krem = pack_A_krem(Aptr, StaticInt{mc}(), Ta, mo, Krem)
#                             Cpmat = PtrMatrix(gesp(Cptr, (mo*mc, no*nc)), mc, nc)
#                             loopmulprefetch!(Cpmat, Apacked_krem, Bpacked_krem, α, β)
#                         end
#                     end
#                     # Mrem
#                     if Mrem > 0
#                         Base.Threads.@spawn begin
#                             # begin
#                             Apacked_mrem_krem = pack_A_mrem_krem(Aptr, StaticInt{mc}(), Ta, Mrem, Krem, Miter)
#                             Cpmat_mrem = PtrMatrix(gesp(Cptr, (Miter*mc, no*nc)), Mrem, nc)
#                             loopmulprefetch!(Cpmat_mrem, Apacked_mrem_krem, Bpacked_krem, α, β)
#                         end
#                     end
#                 end # sync
#                 k = VectorizationBase.unwrap(Krem)
#                 for ko in 1:Kiter
#                     # pack kc x nc block of B
#                     Bpacked = pack_B(Bptr, StaticInt{kc}(), StaticInt{nc}(), Tb, k, no)
#                     # Bprefetch = PtrMatrix{kc,nc}(gesp(Bptr, ((ko+1)*kc, no*nc)))
#                     # copyto!(Bpacked, Bpmat, Bprefetch)
#                     Base.Threads.@sync begin
#                         for mo in 0:Miter-1
#                             Base.Threads.@spawn begin
#                                 # begin
#                                 Apacked = pack_A(Aptr, StaticInt{mc}(), StaticInt{kc}(), Ta, mo, k)
#                                 Cpmat = PtrMatrix(gesp(Cptr, (mo*mc, no*nc)), mc, nc)
#                                 loopmulprefetch!(Cpmat, Apacked, Bpacked, α, StaticInt{1}())
#                             end
#                         end
#                         # Mrem
#                         if Mrem > 0
#                             Base.Threads.@spawn begin
#                                 # begin
#                                 Apacked_mrem = pack_A_mrem(Aptr, StaticInt{mc}(), StaticInt{kc}(), Ta, Mrem, k, Miter)
#                                 Cpmat_mrem = PtrMatrix(gesp(Cptr, (Miter*mc, no*nc)), Mrem, nc)
#                                 loopmulprefetch!(Cpmat_mrem, Apacked_mrem, Bpacked, α, StaticInt{1}())
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
#                 Bpacked_krem_nrem = pack_B_krem_nrem(Bptr, StaticInt{kc}(), StaticInt{nc}(), Tb, Krem, Nrem, Niter)
#                 # Bpacked_krem_nrem = PtrMatrix{-1,-1}(gesp(Bptr, (Kiter*kc, Niter*nc)), Krem, Nrem)
#                 Base.Threads.@sync begin
#                     for mo in 0:Miter-1
#                         Base.Threads.@spawn begin
#                             # begin
#                             # pack mc x kc block of A
#                             Apacked_krem = pack_A_krem(Aptr, StaticInt{mc}(), Ta, mo, Krem)
#                             Cpmat_nrem = PtrMatrix(gesp(Cptr, (mo*mc, Niter*nc)), mc, Nrem)
#                             loopmulprefetch!(Cpmat_nrem, Apacked_krem, Bpacked_krem_nrem, α, β)
#                         end
#                     end
#                     # Mrem
#                     if Mrem > 0
#                         Base.Threads.@spawn begin
#                             # begin
#                             Apacked_mrem_krem = pack_A_mrem_krem(Aptr, StaticInt{mc}(), Ta, Mrem, Krem, Miter)
#                             Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
#                             loopmulprefetch!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem, α, β)
#                         end
#                     end
#                 end # sync
#                 k = VectorizationBase.unwrap(Krem)
#                 for ko in 1:Kiter
#                     # pack kc x nc block of B
#                     Bpacked_nrem = pack_B_nrem(Bptr, StaticInt{kc}(), StaticInt{nc}(), Tb, k, Nrem, Niter)
#                     Base.Threads.@sync begin
#                         for mo in 0:Miter-1
#                             Base.Threads.@spawn begin
#                                 # begin
#                                 # pack mc x kc block of A
#                                 Apacked = pack_A(Aptr, StaticInt{mc}(), StaticInt{kc}(), Ta, mo, k)
#                                 Cpmat_nrem = PtrMatrix(gesp(Cptr, (mo*mc, Niter*nc)), mc, Nrem)
#                                 loopmulprefetch!(Cpmat_nrem, Apacked, Bpacked_nrem, α, StaticInt{1}())
#                             end
#                         end
#                         # Mrem
#                         if Mrem > 0
#                             Base.Threads.@spawn begin
#                                 # begin
#                                 Apacked_mrem = pack_A_mrem(Aptr, StaticInt{mc}(), StaticInt{kc}(), Ta, Mrem, k, Miter)
#                                 Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mc, Niter*nc)), Mrem, Nrem)
#                                 loopmulprefetch!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem, α, StaticInt{1}())
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

maybeinline(::Any, ::Any) = false
function maybeinline(C::AbstractStrideMatrix{<:Any,<:Any,T}, ::AbstractStrideMatrix{<:Any,<:Any,<:Any,1}) where {T}
    M, N = size(C)
    if M isa StaticInt && N isa StaticInt
        static_sizeof(T) * M * N < StaticInt{176}() * StaticInt(mᵣ) * StaticInt{nᵣ}()
    else
        false
    end
end
function maybeinline(C::AbstractStrideMatrix{<:Any,<:Any,T}, ::AbstractStrideMatrix) where {T}
    M, N = size(C)
    if M isa StaticInt && N isa StaticInt
        M * static_sizeof(T) ≤ StaticInt{2}() * StaticInt{VectorizationBase.REGISTER_SIZE}()
    else
        false
    end
end

                                                                       

@inline function jmul!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}) where {Tc, Ta, Tb}
    maybeinline(C, A) && return inlineloopmul!(C, A, B, StaticInt{1}(), StaticInt{0}())
    mc, kc, nc = matmul_params(Tc)
    jmul!(C, A, B, StaticInt{1}(), StaticInt{0}(), mc, kc, nc)
end
@inline function jmul!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α) where {Tc, Ta, Tb}
    maybeinline(C, A) && return inlineloopmul!(C, A, B, α, StaticInt{0}())
    mc, kc, nc = matmul_params(Tc)
    jmul!(C, A, B, α, StaticInt{0}(), mc, kc, nc)
end
@inline function jmul!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β) where {Tc, Ta, Tb}
    maybeinline(C, A) && return inlineloopmul!(C, A, B, α, β)
    mc, kc, nc = matmul_params(Tc)
    jmul!(C, A, B, α, β, mc, kc, nc)
end
# function jmult!(C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}) where {Tc, Ta, Tb}
#     mc, kc, nc = matmul_params_static(Tc)
#     jmult!(C, A, B, mc, kc, nc)
# end





# function packarray_A!(dest::AbstractStrideArray{Tuple{Mᵣ,K,Mᵢ}}, src::AbstractStrideArray{Tuple{Mᵣ,Mᵢ,K}}) where {Mᵣ,K,Mᵢ}
#     # @inbounds for mᵢ ∈ axes(dest,3), k ∈ axes(dest,2), mᵣ ∈ axes(dest,1)
#     @avx for mᵢ ∈ axes(dest,3), k ∈ axes(dest,2), mᵣ ∈ axes(dest,1)
#         dest[mᵣ,k,mᵢ] = src[mᵣ,mᵢ,k]
#     end
# end
# function packarray_B!(dest::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ},T}, src::AbstractStrideArray{Tuple{K,Nᵣ,Nᵢ},T}) where {Nᵣ,K,Nᵢ,T}
#     @avx for k ∈ axes(dest,2), nᵢ ∈ axes(dest,3), nᵣ ∈ axes(dest,1)
#         dest[nᵣ,k,nᵢ] = src[k,nᵣ,nᵢ]
#     end
# end
# function packarray_B!(dest::AbstractStrideArray{Tuple{Nᵣ,K,Nᵢ},T}, src::AbstractStrideArray{Tuple{K,Nᵣ,Nᵢ},T}, nrem) where {Nᵣ,K,Nᵢ,T}
#     nᵢaxis = 1:(size(src,3) - !iszero(nrem))
#     @avx inline=true for nᵢ ∈ nᵢaxis, k ∈ axes(dest,2), nᵣᵢ ∈ axes(dest,1)
#     # for nᵢ ∈ nᵢaxis, k ∈ axes(dest,2), nᵣᵢ ∈ axes(dest,1)
#         dest[nᵣᵢ,k,nᵢ] = src[k,nᵣᵢ,nᵢ]
#     end
#     if !iszero(nrem)
#         nᵢ = size(dest,3)
#         nᵣₛ = Static{nᵣ}()
#         @avx inline=true for k ∈ axes(dest,2), nᵣᵢ ∈ 1:nᵣₛ
#         # for k ∈ axes(dest,2), nᵣᵢ ∈ 1:nᵣₛ
#             dest[nᵣᵢ,k,nᵢ] = nᵣᵢ ≤ nrem ? src[k,nᵣᵢ,nᵢ] : zero(T)
#         end
#         # @inbounds for k ∈ axes(dest,2)
#              # @simd ivdep for nᵣᵢ ∈ 1:nᵣₛ
#                  # dest[nᵣᵢ,k,nᵢ] = nᵣᵢ ≤ nrem ? src[k,nᵣᵢ,nᵢ] : zero(T)
#              # end
#         # end
#     end
# end


#=
function calc_blocksize(iter, step, max, ::StaticInt{round_to}) where {round_to}
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
            ncblock, (ncreps, ncrem) = calc_blocksize(nciter, nc, N, StaticInt{nᵣ}())

            
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
                    mcblock, (mcreps, mcrem) = calc_blocksize(mciter, mc, M, StaticInt{mᵣW}())


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

# function Apack_krem(Aptr::AbstractStridedPointer{Ta}, ptrL2, ::StaticInt{mᵣW}, ::Type{Tc}, mcrepetitions, Krem, mo, mreps_per_iter) where {mᵣW, Ta, Tc}
#     Apacked_krem = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,1,false}(ptrL2, (Krem,mcrepetitions), (Krem*mᵣW,))
#     Apmat_krem = PtrArray{Tuple{mᵣW,-1,-1},Ta,3}(gesp(Aptr, (mo*mreps_per_iter, 0)), (mcrepetitions,Krem))
#     Apacked_krem, Apmat_krem
# end
# function Apack(
#     Aptr::AbstractStridedPointer{Ta}, ptrL2, ::StaticInt{mᵣW}, ::Type{Tc}, Msub, Ksub, m, k
# ) where {mᵣW, Ta, Tc}
#     mreps_per_iter = mᵣW * Msub
#     # Apacked = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,1,false}(ptrL2, (kreps_per_iter,mcrepetitions), ((mᵣW*kc),))
#     # Apmat = PtrArray{Tuple{mᵣW,-1,-1},Ta,3}(gesp(Aptr, (mo*mreps_per_iter, k)), (mcrepetitions,kreps_per_iter))
#     Apacked = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,1,false}(ptrL2, (Ksub,Msub), ((mᵣW*Ksub),))
#     Apmat = PtrArray{Tuple{mᵣW,-1,-1},Ta,3}(gesp(Aptr, (m*mreps_per_iter, k)), (Msub,Ksub))
#     Apacked, Apmat
# end


# roundupnᵣ(x) = (xnr = x + nᵣ; xnr - (xnr % nᵣ))

# function jmulh!(
#     C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α = Val(1), β = Val(0)
# # ) where {Tc, Ta, Tb}
# ) where {Ta, Tb, Tc}
#     mc, kc, nc = matmul_params_static(Tc)
#     jmulh!(C, A, B, α, β, mc, kc, nc)
# end
# function jmulh!(
#     C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, ::StaticInt{mc}, ::StaticInt{kc}, ::StaticInt{nc}, (M, K, N) = matmul_sizes(C, A, B)
# # ) where {Tc, Ta, Tb}
# ) where {Ta, Tb, Tc, mc, kc, nc}
#     W = VectorizationBase.pick_vector_width(Tc)
#     mᵣW = mᵣ * W

#     num_n_iter = cld_fast(N, nc)
#     nreps_per_iter = div_fast(N, num_n_iter)
#     nreps_per_iter += nᵣ - 1
#     ncrepetitions = div_fast(nreps_per_iter, nᵣ)
#     nreps_per_iter = nᵣ * ncrepetitions
#     Niter = num_n_iter - 1
#     Nrem = N - Niter * nreps_per_iter

#     num_m_iter = cld_fast(M, mc)
#     mreps_per_iter = div_fast(M, num_m_iter)
#     mreps_per_iter += mᵣW - 1
#     mcrepetitions = div_fast(mreps_per_iter, mᵣW)
#     mreps_per_iter = mᵣW * mcrepetitions
#     Miter = num_m_iter - 1
#     Mrem = M - Miter * mreps_per_iter
    
#     num_k_iter = cld_fast(K, kc)
#     kreps_per_iter = ((div_fast(K, num_k_iter)) + 3) & -4
#     Krem = K - (num_k_iter - 1) * kreps_per_iter
#     Kiter = num_k_iter - 1

#     # @show mreps_per_iter, kreps_per_iter, nreps_per_iter
#     # @show Miter, Kiter, Niter
#     # @show Mrem, Krem, Nrem
    
#     Aptr = stridedpointer(A)
#     Bptr = stridedpointer(B)
#     Cptr = stridedpointer(C)
#     ptrL3 = threadlocal_L3CACHE_pointer(Tc)
#     ptrL2 = threadlocal_L2CACHE_pointer(Tc)
#     Noff = 0
#     GC.@preserve C A B LCACHEARRAY begin
#         for no in 0:Niter-1
#             # Krem
#             # pack kc x nc block of B
            
#             Bpacked_krem = PtrArray{Tuple{nᵣ,-1,-1},Tc,3,Tuple{1,nᵣ,-1},2,1,false}(ptrL3, (Krem,ncrepetitions), (nᵣ * Krem,))
#             Bpmat_krem = PtrArray{Tuple{-1,nᵣ,-1},Tb,3}(gesp(Bptr, (0, Noff)), (Krem,ncrepetitions))
#             packarray_B!(Bpacked_krem, Bpmat_krem)
#             for mo in 0:Miter-1
#                 # pack mc x kc block of A
#                 # Apacked_krem = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,1,false}(ptrL2, (Krem,mcrepetitions), (Krem*mᵣW,))
#                 # Apmat_krem = PtrArray{Tuple{mᵣW,-1,-1},Ta,3}(gesp(Aptr, (mo*mreps_per_iter, 0)), (mcrepetitions,Krem))
#                 Apacked_krem, Apmat_krem = Apack(Aptr, ptrL2, Val(mᵣW), Tc, mcrepetitions, Krem, mo, 0)
#                 # packarray_A!(Apacked_krem, Apmat_krem)
#                 Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noff))
#                 Cx = first(Cptr.strides)
#                 Cpmat = PtrArray{Tuple{mᵣW,-1,nᵣ,-1},Tc,4,Tuple{1,mᵣW,-1,-1},2,2,false}(Cptr_off, (mcrepetitions,ncrepetitions), (Cx,Cx*nᵣ))
#                 packaloopmul!(Cpmat, Apacked_krem, Apmat_krem, Bpacked_krem, α, β)
#             end
#             # Mrem
#             if Mrem > 0
#                 Apacked_mrem_krem = PtrMatrix(ptrL2, Mrem, Krem, VectorizationBase.align(Mrem, Tc))
#                 Apmat_mrem_krem = PtrMatrix(gesp(Aptr, (Miter*mreps_per_iter, 0)), Mrem, Krem)
#                 # copyto!(Apacked_mrem_krem, Apmat_mrem_krem)
#                 Cpmat_mrem = PtrArray{Tuple{-1,nᵣ,-1},Tc,3}(gesp(Cptr, (Miter*mreps_per_iter, Noff)), (Mrem, ncrepetitions))
#                 packaloopmul!(Cpmat_mrem, Apacked_mrem_krem, Apmat_mrem_krem, Bpacked_krem, α, β)
#             end
#             k = Krem
#             for ko in 1:Kiter
#                 # pack kc x nc block of B
#                 Bpacked = PtrArray{Tuple{nᵣ,-1,-1},Tc,3,Tuple{1,nᵣ,-1},2,1,false}(ptrL3, (kreps_per_iter,ncrepetitions), (nᵣ * kreps_per_iter,))
#                 Bpmat = PtrArray{Tuple{-1,nᵣ,-1},Tb,3}(gesp(Bptr, (k, Noff)), (kreps_per_iter,ncrepetitions))
#                 packarray_B!(Bpacked, Bpmat)
#                 for mo in 0:Miter-1
#                     # pack mc x kc block of A
#                     Apacked, Apmat = Apack(Aptr, ptrL2, Val(mᵣW), Tc, mcrepetitions, kreps_per_iter, mo, k)
#                     Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noff))
#                     Cx = first(Cptr.strides)
#                     Cpmat = PtrArray{Tuple{mᵣW,-1,nᵣ,-1},Tc,4,Tuple{1,mᵣW,-1,-1},2,2,false}(Cptr_off, (mcrepetitions,ncrepetitions), (Cx,Cx*nᵣ))
#                     packaloopmul!(Cpmat, Apacked, Apmat, Bpacked, α, Val(1))
#                 end
#                 # Mrem
#                 if Mrem > 0
#                     Apacked_mrem = PtrMatrix(ptrL2, Mrem, kreps_per_iter, VectorizationBase.align(Mrem,Tc))
#                     Apmat_mrem = PtrMatrix(gesp(Aptr, (Miter*mreps_per_iter, k)), Mrem, kreps_per_iter)
#                     # copyto!(Apacked_mrem, Apmat_mrem)
#                     Cpmat_mrem = PtrArray{Tuple{-1,nᵣ,-1},Tc,3}(gesp(Cptr, (Miter*mreps_per_iter, Noff)), (Mrem, ncrepetitions))
#                     packaloopmul!(Cpmat_mrem, Apacked_mrem, Apmat_mrem, Bpacked, α, Val(1))
#                 end
#                 k += kreps_per_iter
#             end
#             Noff += nreps_per_iter
#         end
#         # Nrem
#         if Nrem > 0
#             # calcnrrem = false
#             # Krem
#             # pack kc x nc block of B
#             ncreprem, ncrepremrem = divrem_fast(Nrem, nᵣ)
#             ncrepremc = ncreprem + !(iszero(ncrepremrem))
#             lastBstride = nᵣ * Krem
#             Bpacked_krem = PtrArray{Tuple{nᵣ,-1,-1},Tc,3,Tuple{1,nᵣ,-1},2,1,false}(ptrL3, (Krem,ncrepremc), (lastBstride,))
#             Bpmat_krem = PtrArray{Tuple{-1,nᵣ,-1},Tb,3}(gesp(Bptr, (0, Noff)), (Krem,ncrepremc))  # Note the last axis extends past array's end!
#             packarray_B!(Bpacked_krem, Bpmat_krem, ncrepremrem) 
#             Noffrem = Noff + ncreprem*nᵣ
#             Bpacked_krem_nrem = PtrMatrix{-1,-1,Tc,1,nᵣ,2,0,false}(gep(ptrL3, lastBstride * ncreprem), (ncrepremrem,Krem), tuple())
#             for mo in 0:Miter-1
#                 # pack mc x kc block of A
#                 Apacked_krem, Apmat_krem = Apack(Aptr, ptrL2, Val(mᵣW), Tc, mcrepetitions, Krem, mo, 0)
#                 Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noff))
#                 Cx = first(Cptr.strides)
#                 Cpmat = PtrArray{Tuple{mᵣW,-1,nᵣ,-1},Tc,4,Tuple{1,mᵣW,-1,-1},2,2,false}(Cptr_off, (mcrepetitions,ncreprem), (Cx,Cx*nᵣ))
#                 packaloopmul!(Cpmat, Apacked_krem, Apmat_krem, Bpacked_krem, α, β)
#                 # @show Apacked_krem
#                 if ncrepremrem > 0
#                     # if calcnrrem && ncrepremrem > 0
#                     Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noffrem))
#                     Cpmat_nrem = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,2,false}(Cptr_off, (mcrepetitions,ncrepremrem), (Cx,Cx*nᵣ))
#                     loopmul!(Cpmat_nrem, Apacked_krem, Bpacked_krem_nrem, α, β)
#                 end
#             end
#             # Mrem
#             if Mrem > 0
#                 Apacked_mrem_krem = PtrMatrix(ptrL2, Mrem, Krem, VectorizationBase.align(Mrem, Tc))
#                 Apmat_mrem_krem = PtrMatrix(gesp(Aptr, (Miter*mreps_per_iter, 0)), Mrem, Krem)
#                 Cpmat_mrem = PtrArray{Tuple{-1,nᵣ,-1},Tc,3}(gesp(Cptr, (Miter*mreps_per_iter, Noff)), (Mrem, ncreprem))
#                 packaloopmul!(Cpmat_mrem, Apacked_mrem_krem, Apmat_mrem_krem, Bpacked_krem, α, β)

#                 if ncrepremrem > 0
#                     # if calcnrrem && ncrepremrem > 0
#                     Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mreps_per_iter, Noff + ncreprem * nᵣ)), Mrem, ncrepremrem)
#                     loopmul!(Cpmat_mrem_nrem, Apacked_mrem_krem, Bpacked_krem_nrem', α, β, (Mrem, Krem, ncrepremrem))
#                 end

#             end
#             k = Krem
#             for ko in 1:Kiter
#                 # pack kc x nc block of B
#                 lastBstride = nᵣ * kreps_per_iter
#                 Bpacked = PtrArray{Tuple{nᵣ,-1,-1},Tc,3,Tuple{1,nᵣ,-1},2,1,false}(ptrL3, (kreps_per_iter,ncrepremc), (lastBstride,))
#                 Bpmat = PtrArray{Tuple{-1,nᵣ,-1},Tb,3}(gesp(Bptr, (k, Noff)), (kreps_per_iter,ncrepremc)) # Note the last axis extends past array's end!
#                 packarray_B!(Bpacked, Bpmat, ncrepremrem)
#                 Bpacked_nrem = PtrMatrix{-1,-1,Tc,1,nᵣ,2,0,false}(gep(ptrL3, lastBstride * ncreprem), (ncrepremrem,kreps_per_iter), tuple())
#                 for mo in 0:Miter-1
#                     # pack mc x kc block of A
#                     Apacked, Apmat = Apack(Aptr, ptrL2, Val(mᵣW), Tc, mcrepetitions, kreps_per_iter, mo, k)
#                     Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noff))
#                     Cx = first(Cptr.strides)
#                     Cpmat = PtrArray{Tuple{mᵣW,-1,nᵣ,-1},Tc,4,Tuple{1,mᵣW,-1,-1},2,2,false}(Cptr_off, (mcrepetitions,ncreprem), (Cx,Cx*nᵣ))
#                     packaloopmul!(Cpmat, Apacked, Apmat, Bpacked, α, Val(1))
#                     # if calcnrrem && ncrepremrem > 0
#                     if ncrepremrem > 0
#                         Cptr_off = gep(Cptr, (mo*mreps_per_iter, Noffrem))
#                         Cpmat_nrem = PtrArray{Tuple{mᵣW,-1,-1},Tc,3,Tuple{1,mᵣW,-1},2,2,false}(Cptr_off, (mcrepetitions,ncrepremrem), (Cx,Cx*nᵣ))
#                         loopmul!(Cpmat_nrem, Apacked, Bpacked_nrem, α, Val(1))
#                     end

#                 end
#                 # Mrem
#                 if Mrem > 0
#                     Apacked_mrem = PtrMatrix(ptrL2, Mrem, kreps_per_iter, VectorizationBase.align(Mrem,Tc))
#                     Apmat_mrem = PtrMatrix(gesp(Aptr, (Miter*mreps_per_iter, k)), Mrem, kreps_per_iter)
#                     Cpmat_mrem = PtrArray{Tuple{-1,nᵣ,-1},Tc,3}(gesp(Cptr, (Miter*mreps_per_iter, Noff)), (Mrem, ncreprem))
#                     packaloopmul!(Cpmat_mrem, Apacked_mrem, Apmat_mrem, Bpacked, α, Val(1))

#                     # if calcnrrem && ncrepremrem > 0
#                     if ncrepremrem > 0
#                         Cpmat_mrem_nrem = PtrMatrix(gesp(Cptr, (Miter*mreps_per_iter, Noff + ncreprem * nᵣ)), Mrem, ncrepremrem)
#                         loopmul!(Cpmat_mrem_nrem, Apacked_mrem, Bpacked_nrem', α, Val(1), (Mrem, kreps_per_iter, ncrepremrem))
#                     end

#                 end
#                 k += kreps_per_iter
#             end
#         end
#     end # GC.@preserve
#     C
#  end # function 

@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractMatrix, B::AbstractMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractStrideMatrix, B::AbstractMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractStrideMatrix, B::AbstractMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractStrideMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractStrideMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)

# @inline function Base.:*(
#     sp::StackPointer,
#     A::AbstractStrideMatrix{<:Any,<:Any,T},
#     B::AbstractStrideMatrix{<:Any,<:Any,T}
# ) where {T}
#     sp, D = PtrArray{T}(sp, (maybestaticsize(A, StaticInt{1}()),maybestaticsize(B, StaticInt{2}())))
#     sp, mul!(D, A, B)
# end
@inline function Base.:*(
    A::AbstractStrideMatrix{<:Any,<:Any,TA},
    B::AbstractStrideMatrix{<:Any,<:Any,TB}
) where {TA,TB}
    TC = promote_type(TA,TB)
    C = StrideArray{TC}(undef, (size(A, StaticInt{1}()),size(B, StaticInt{2}())))
    jmul!(C, A, B)
    C
end

@inline extract_λ(a) = a
@inline extract_λ(a::UniformScaling) = a.λ
@inline function Base.:*(A::AbstractStrideArray{S,D,T}, bλ::Union{Tb,UniformScaling{Tb}}) where {S,D,T,Tb}
    mv = similar(A)
    b = T(extract_λ(bλ))
    @avx for i ∈ eachindex(A)
        mv[i] = A[i] * b
    end
    mv
end
# function Base.:*(
#     sp::StackPointer,
#     A::AbstractFixedSizeArray{S,T,N,X,L},
#     bλ::Union{T,UniformScaling{T}}
# ) where {S,T<:Real,N,X,L}
#     mv = PtrArray{S,T,N,X,L}(pointer(sp,T))
#     b = extract_λ(bλ)
#     @avx for i ∈ eachindex(A)
#         mv[i] = A[i] * b
#     end
#     sp + align(sizeof(T)*L), mv
# end
# @inline function Base.:*(a::T, B::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
#     mv = FixedSizeArray{S,T,N,X,L}(undef)
#     @avx for i ∈ eachindex(B)
#         mv[i] = a * B[i]
#     end
#         # mv
#     ConstantFixedSizeArray(mv)
# end

# @inline function nmul!(
#     D::AbstractMatrix{T},
#     A′::LinearAlgebra.Adjoint{T,<:AbstractMatrix{T}},
#     X::AbstractMatrix{T}
# ) where {T <: BLAS.BlasFloat}
#     BLAS.gemm!('T','N',-one(T),A′.parent,X,zero(T),D)
# end


@inline function LinearAlgebra.mul!(
    C::AbstractStrideMatrix{<:Any,<:Any,T},
    A::LinearAlgebra.Diagonal{T,<:AbstractStrideVector{<:Any,T}},
    B::AbstractMatrix{T}
) where {T}
    M, K, N = matmul_axes(C, A, B)
    MandK = ArrayInterface._pick_range(M, K)
    vA = parent(A)
    @avx for n ∈ N, m ∈ 1:MandK
        C[m,n] = vA[m] * B[m,n]
    end
    C
end
@inline function Base.:*(
    A::LinearAlgebra.Diagonal{T,<:AbstractVector{T}},
    B::AbstractStrideMatrix{<:Any,<:Any,T}
) where {T}
    mul!(similar(B), A, B)
end
# function Base.:*(
#     sp::StackPointer,
#     A::LinearAlgebra.Diagonal{T,<:AbstractStrideVector{M,T}},
#     B::AbstractStrideMatrix{M,N,T}
# ) where {M,N,T}
#     sp, C = similar(sp, B)
#     sp, mul!(C, A, B)
# end





# @static if Base.libllvm_version ≥ v"10.0.0"
#     function llvmmul!(
#         C::AbstractMutableFixedSizeMatrix{M,N,T,1,XC},
#         A::AbstractMutableFixedSizeMatrix{M,K,T,1,XA},
#         B::AbstractMutableFixedSizeMatrix{K,N,T,1,XB}
#     ) where {M,K,N,T,XC,XA,XB}
#         vA = SIMDPirates.vcolumnwiseload(pointer(A), XA, StaticInt{M}(), StaticInt{K}())
#         vB = SIMDPirates.vcolumnwiseload(pointer(B), XB, StaticInt{K}(), StaticInt{N}())
#         vC = SIMDPirates.vmatmul(vA, vB, StaticInt{M}(), StaticInt{K}(), StaticInt{N}())
#         vcolumnwisestore!(pointer(C), vC, XC, StaticInt{M}(), StaticInt{N}())
#         C
#     end
# end


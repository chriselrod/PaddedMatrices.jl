

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


function matmul_params(::Type{T}) where {T}
    Mᵣ = StaticInt{mᵣ}()
    Nᵣ = StaticInt{nᵣ}()
    W = VectorizationBase.pick_vector_width_val(T)
    MᵣW = Mᵣ * W
    mc = MᵣW * (VectorizationBase.REGISTER_SIZE === 64 ? StaticInt{4}() : StaticInt{9}()) # TODO: make this smarter/less heuristic
    # L₁ = something(core_cache_size(T, Val(1)), StaticInt{32768}() ÷ static_sizeof(T))
    L₂ = if CACHE_INCLUSIVITY[2]
        something(core_cache_size(T, Val(2)), StaticInt{262144}() ÷ static_sizeof(T)) - something(core_cache_size(T, Val(1)), StaticInt{32768}() ÷ static_sizeof(T))
    else
        something(core_cache_size(T, Val(2)), StaticInt{262144}() ÷ static_sizeof(T))
    end
    kc = ((StaticInt{795}() * L₂) ÷ StaticInt{1024}() - StaticInt{30420}()) ÷ mc
    L₃ = if CACHE_INCLUSIVITY[3]
        something(cache_size(T, Val(3)), StaticInt{3145728}() ÷ static_sizeof(T)) - L₂
    else
        something(cache_size(T, Val(3)), StaticInt{3145728}() ÷ static_sizeof(T))
    end
    nc = ((((StaticInt{132}() * L₃) ÷ StaticInt{125}()) - StaticInt{256651}()) ÷ (kc * Nᵣ)) * Nᵣ
    mc, kc, nc
end

@generated function dense_dims_subset(::DenseDims{D}, ::StrideRank{R}) where {D,R}
    t = Expr(:tuple)
    for n in eachindex(R)
        push!(t.args, D[n] & (R[n] == 1))
    end
    Expr(:call, Expr(:curly, :DenseDims, t))
end

"""
Only packs `A`. Primitively does column-major packing: it packs blocks of `A` into a column-major temporary.
"""
function jmulpackAonly!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer,
    α, β, ::StaticInt{mc}, ::StaticInt{kc}, ::StaticInt{nc}
) where {T, mc, kc, nc}
    W = VectorizationBase.pick_vector_width_val(Tc)
    mᵣW = StaticInt{mᵣ}() * W
    
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
    Aptr = zero_offsets(A); Bptr = zero_offsets(B); Cptr = zero_offsets(C);
    _Mrem, _mreps_per_iter = promote(Mrem, mreps_per_iter)
    _Krem, _kreps_per_iter = promote(Krem, kreps_per_iter)
    # koffset = 0
    for ko ∈ 0:Kiter
        ksize = ifelse(ko == 0, _Krem, _kreps_per_iter)
        # _β = ifelse(ko == 0, convert(Tc, β), one(Tc))
        # Bpacked = PtrArray(gesp(Bptr, (koffset, Zero())), (ksize, N), dense_dims_subset(dense_dims(B), stride_rank(B)))
        
        # moffset = 0
        let Aptr = Aptr, Cptr = Cptr
            for mo in 0:Miter
                msize = ifelse(mo == Miter, _Mrem, _mreps_per_iter)
                # pack mreps_per_iter x kreps_per_iter block of A
                # Asubset = PtrArray(gesp(Aptr, (moffset, koffset)), (msize, ksize), dense_dims_subset(dense_dims(A), stride_rank(A)))
                # _Aptr = gesp(Aptr, (moffset, koffset))
                # Cpmat = PtrArray(gesp(Cptr, (moffset, Zero())), (msize, N), dense_dims_subset(dense_dims(C), stride_rank(C)))
                # _Cptr = gesp(Cptr, (moffset, Zero()))
                # packaloopmul!(Cpmat, Asubset, Bpacked, α, _β, (CloseOpen(msize), CloseOpen(ksize), CloseOpen(N)))
                if ko == 0
                    packaloopmul!(Cptr, Aptr, Bptr, α, β, (msize, ksize, N))
                else
                    packaloopmul!(Cptr, Aptr, Bptr, α, One(), (msize, ksize, N))
                end
                # moffset += mreps_per_iter
                Aptr = gesp(Aptr, (msize, Zero()))
                Cptr = gesp(Cptr, (msize, Zero()))
            end
        end
        Aptr = gesp(Aptr, (Zero(), ksize))
        Bptr = gesp(Bptr, (ksize, Zero()))
        # koffset += ksize
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
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer,
    α, β, ::StaticInt{mc}, ::StaticInt{kc}, ::StaticInt{nc}, (M, K, N) = matmul_sizes(C, A, B)
) where {T, mc, kc, nc}
    W = VectorizationBase.pick_vector_width_val(T)
    mᵣW = StaticInt{mᵣ}() * W
    
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
        
    Aptr = zero_offsets(A)
    Bptr = zero_offsets(B)
    Cptr = zero_offsets(C)
    # L3ptr = Base.unsafe_convert(Ptr{Tb}, pointer(BCACHE) + (Threads.threadid()-1)*BSIZE*8)
    # bc = _use_bcache()
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
            # koffset = 0
            let Aptr = Aptr, Bptr = Bptr
                for ko ∈ 0:Kiter
                    ksize = ifelse(ko == 0, _Krem, _kreps_per_iter)
                    # _β = ifelse(ko == 0, convert(T, β), one(T))
                    # Bsubset2 = PtrArray(gesp(Bptr, (koffset, noffset)), (ksize, nsize), dense_dims_subset(dense_dims(B), stride_rank(B)))
                    Bsubset2 = PtrArray(Bptr, (ksize, nsize), dense_dims_subset(dense_dims(B), stride_rank(B)))
                    Bpacked2 = ptrarray0(L3ptr, (ksize, nsize))
                    # @show offsets(Bpacked2), offsets(Bsubset2)
                    copyto!(Bpacked2, Bsubset2)
                    let Aptr = Aptr, Cptr = Cptr, Bptr = stridedpointer(Bpacked2)
                        # moffset = 0
                        for mo in 0:Miter
                            msize = ifelse(mo == Miter, _Mrem, _mreps_per_iter)
                            # pack mreps_per_iter x kreps_per_iter block of A
                            # Asubset = PtrArray(gesp(Aptr, (moffset, koffset)), (msize, ksize), dense_dims_subset(dense_dims(A), stride_rank(A)))

                            # Cpmat = PtrArray(gesp(Cptr, (moffset, noffset)), (msize, nsize), dense_dims_subset(dense_dims(C), stride_rank(C)))
                            # packaloopmul!(Cpmat, Asubset, Bpacked2, α, _β, (zrange(msize), zrange(ksize), zrange(nsize)))
                            if ko == 0
                                packaloopmul!(Cptr, Aptr, Bptr, α, β, (msize, ksize, nsize))
                            else
                                packaloopmul!(Cptr, Aptr, Bptr, α, One(), (msize, ksize, nsize))
                            end
                            # moffset += mreps_per_iter
                            Aptr = gesp(Aptr, (msize, Zero()))
                            Cptr = gesp(Cptr, (msize, Zero()))
                        end
                    end
                    # koffset += ksize
                    Aptr = gesp(Aptr, (Zero(), ksize))
                    Bptr = gesp(Bptr, (ksize, Zero()))
                end
            end
            # noffset += nreps_per_iter
            Bptr = gesp(Bptr, (Zero(), nsize))
            Cptr = gesp(Cptr, (Zero(), nsize))
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
    C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, ::StaticInt{mc}, ::StaticInt{kc}, ::StaticInt{nc}, (M, K, N) = matmul_sizes(C, A, B)
) where {Tc, Ta, Tb, mc, kc, nc}
    pA = zstridedpointer(A); pB = zstridedpointer(B); pC = zstridedpointer(C);
    Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B);
    GC.@preserve Cb Ab Bb begin
        if VectorizationBase.CACHE_SIZE[2] === nothing || ((nᵣ ≥ N) || (contiguousstride1(pA) && dontpack(pointer(pA), M, K, bytestride(pA,StaticInt{2}()), StaticInt{mc}(), StaticInt{kc}(), Tc)))
            loopmul!(pC, pA, pB, α, β, (CloseOpen(M),CloseOpen(K),CloseOpen(N)))
        elseif VectorizationBase.CACHE_SIZE[3] === nothing || (((contiguousstride1(pB) && (kc * nc ≥ K * N))) || firstbytestride(pB) ≤ 1600)
            # println("Pack A mul")
            jmulpackAonly!(pC, pA, pB, α, β, StaticInt{mc}(), StaticInt{kc}(), StaticInt{nc}(), (M,K,N))
        else
            # println("Pack A and B mul")
            jmulpackAB!(pC, pA, pB, α, β, StaticInt{mc}(), StaticInt{kc}(), StaticInt{nc}(), (M,K,N))
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

# struct ThreadRun
#     id::UInt32
#     nthread::UInt32
# end
# ThreadRun(i::Int, n::Int) = ThreadRun(i % UInt32, n % UInt32)

struct LoopMulClosure{P,TC,TA,TB,Α,Β,M,K,N}
    C::TC
    A::TA
    B::TB
    α::Α # \Alpha
    β::Β # \Beta
    M::Md
    K::Kd
    N::Nd
end
function LoopMulClosure{P}(
    C::TC, A::TA, B::TB, α::Α, β::Β, M::Md, K::Kd, N::Nd
) where {P,TC<:AbstractStridedPointer,TA<:AbstractStridedPointer,TB<:AbstractStridedPointer,Α,Β,Md,Kd,Nd}
    LoopMulClosure{P,TC,TA,TB,Α,Β,Md,Kd,Nd}(C, A, B, α, β, M, K, N)
end
# function LoopMulClosure{true}(
#     C::TC, A::TA, B::TB, α::Α, β::Β, Maxis::M, Kaxis::K, Naxis::N
# ) where {TC,TA,TB,Α,Β,M,K,N}
#     LoopMulClosure{true,TC,TA,TB,Α,Β,M,K,N}(C, A, B, α, β, Maxis, Kaxis, Naxis)
# end
function LoopMulClosure{P}(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β, M, K, N) where {P} # if not packing, discard `PtrArray` wrapper
    LoopMulClosure{P}(stridedpointer(C), stridedpointer(A), stridedpointer(B), α, β, M, K, N)
end
# function LoopMulClosure{true}(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β, M, K, N)
#     LoopMulClosure{true}(C, A, B, α, β, M, K, N)
# end
(m::LoopMulClosure{false})() = loopmul!(m.C, m.A, m.B, m.α, m.β, (m.M, m.K, m.N))
# (m::LoopMulClosure{true})() = jmulpackAonly!(m.C, m.A, m.B, m.α, m.β, (m.Maxis, m.Kaxis, m.Naxis))
function (m::LoopMulClosure{true,TC})() where {S,D,T,TC <: AbstractStrideArray{S,D,T}}
    Mc,Kc,Nc = matmul_params(T)
    jmulpackAonly!(m.C, m.A, m.B, m.α, m.β, Mc, Kc, Nc, (m.M, m.K, m.N))
end

struct PackAClosure{TC,TA,TB,Α,Β,Md,Kd,Nd,T}
    # lmc::LoopMulClosure{TC,TA,TB,Α,Β,M,K,N}
    C::TC
    A::TA
    B::TB
    α::Α # \Alpha
    β::Β # \Beta
    M::Md
    K::Kd
    N::Nd
    tasks::T
end
# function PackAClosure(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β, M, K, N, tasks)
#     PackAClosure(stridedpointer(C), stridedpointer(A), stridedpointer(B), α, β, M, K, N, tasks)
# end
function (m::PackAClosure{TC})() where {T,TC<:AbstractStridedPointer{T}}
    Mc,Kc,Nc = matmul_params(T)
    jmultpackAonly!(m.C, m.A, m.B, m.α, m.β, Mc, Kc, Nc, m.M, m.K, m.N, m.tasks)
end

function jmult!(C::AbstractMatrix{T}, A, B, α = One(), β = Zero(), (M,K,N) = matmul_sizes(C,A,B), nthread = _nthreads()) where {T}
    Mc,Kc,Nc = matmul_params(T)
    jmult!(C, A, B, α, β, Mc, Kc, Nc, (M,K,N), nthread)
end

function jmult!(
    C::AbstractMatrix{T}, A, B, α, β, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}, (M,K,N) = matmul_sizes(C,A,B), nthread = _nthreads()
) where {Mc,Kc,Nc,T}
    Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B)
    GC.@preserve Cb Ab Bb begin
        # Base.unsafe_convert(Ptr{T}, BCACHE)
        _jmult!(
            zstridedpointer(C), zstridedpointer(A), zstridedpointer(B),
            α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), (M,K,N), nthread
        )
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

function _jmult!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer,
    α, β, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}, (M,K,N), nthread
) where {T,Mc,Kc,Nc}
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
        loopmul!(zero_offsets(C), zero_offsets(A), zero_offsets(B), α, β, (CloseOpen(M),CloseOpen(K),CloseOpen(N)))
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
            Mrem = M - Mbsize * (Mblocks - One())
            Nrem = N - Nbsize * (Nblocks - One())
            tnum = 0
            let _A = A, _B = B, _C = C
                for n ∈ Base.OneTo(Nblocks)
                    nsize = ifelse(n == Nblocks, Nrem, Nbsize) 
                    # nrange = nlower:min(N,nupper)
                    # mlower = 0
                    # mupper = Mbsize-1
                    # _B = stridedpointer(zview(B, :, nrange))
                    let _A = _A, _C = _C
                        for m ∈ Base.OneTo(Mblocks)
                            msize = ifelse(m == Mblocks, Mrem, Mbsize)
                            # mrange = mlower:min(M,mupper)
                            # _C = stridedpointer(zview(C, mrange, nrange))
                            # _A = stridedpointer(zview(A, mrange, :))
                            if tnum == _nspawn_m1
                                loopmul!(_C, _A, _B, α, β, msize, K, nsize)
                                foreach(wait, view(tasks, Base.OneTo(_nspawn_m1)))
                                return
                            end
                            t = Task(LoopMulClosure{false}(_C, _A, _B, α, β, msize, K, nsize))
                            t.sticky = false
                            schedule(t)
                            tasks[(tnum += 1)] = t
                            # mlower += Mbsize
                            # mupper += Mbsize
                            _A = gesp(_A, (msize, Zero()))
                            _C = gesp(_C, (msize, Zero()))
                        end
                    end
                    _B = gesp(_B, (Zero(), nsize))
                    _C = gesp(_C, (Zero(), nsize))
                end
            end
        else# do pack A
            Mreps = cld_fast(M, MᵣW)
            # spawn_per_mrep is how many processes we'll spawn here
            spawn_per_mrep = min(cld_fast(nspawn, Mreps), N) # Safety: don't spawn more than `N`
            if spawn_per_mrep > 1
                nspawn_per = cld(nspawn, spawn_per_mrep)
                spawn_start = spawn_per_mrep
                Nper = cld(N, spawn_per_mrep)
                # Nstart = 0
                task_spawn_iters = spawn_per_mrep - 1
                for i ∈ Base.OneTo(task_spawn_iters)
                    tasks_view = view(tasks, spawn_start:spawn_start+nspawn_per-2) # tasks are to be 1 shorter than number of threads running; "main" task runs the last set.
                    # nrange = Nstart:Nstart+Nper-1
                    # t = Task(PackAClosure(zview(C, :, nrange), A, zview(B, :, nrange), α, β, M, K, Nper, task_view))
                    t = Task(PackAClosure(C, A, B, α, β, M, K, Nper, task_view))
                    t.sticky = false
                    schedule(t)
                    tasks[i] = t
                    # Nstart += Nper
                    spawn_start += nspawn_per - 1
                    B = gesp(B, (Zero(), Nper))
                    C = gesp(C, (Zero(), Nper))
                end
                tasks_view = view(tasks, 1:nspawn-1)
                # nrange = Nstart:N-One()
                Nrem = N - (Nper * task_spawn_iters)
                jmultpackAonly!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), M, K, Nrem, tasks_view)
                foreach(wait, view(tasks, Base.OneTo(task_spawn_iters))) # wait for spawned `jmultpackAonly!`s
            else
                tasks_view = view(tasks, 1:nspawn-1)
                jmultpackAonly!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), M, K, N, tasks_view)
            end
            return
        end
    # elseif CC3 > 1
    #     cc3 = min(nspawn, CC3)
    #     for c ∈ Base.OneTo(cc3)
    #         # @spawn begin
                
    #         # endp
    #     end
    #     return C
    else
        jmultpackAB!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), M, K, N, view(tasks, 1:nspawn))
        return
    end
end
function jmultpackAonly!(
    C::AbstractStridedPointer, A::AbstractStridedPointer, B::AbstractStridedPointer,
    α, β, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}, M, K, N, tasks
) where {Mc,Kc,Nc}
    to_spawn = length(tasks)
    total_threads = to_spawn + 1
    
    M = static_length(Maxis)

    W = VectorizationBase.pick_vector_width_val(T)
    _Mblock = cld_fast(M, total_threads)
    Mblock = cld_fast(_Mblock, W) * W
    
    # Mstart = 0
    for i ∈ 1:to_spawn
        # mrange = Mstart:Mstart + Mblock - 1
        # t = Task(LoopMulClosure{true}(zview(C, mrange, :), zview(A, mrange, :), B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), Zero():Mblock-One(), Kaxis, Naxis))
        # t = Task(LoopMulClosure{true}(zview(C, mrange, :), zview(A, mrange, :), B, α, β, Zero():Mblock-One(), Kaxis, Naxis))
        t = Task(LoopMulClosure{true}(C, A, B, α, β, Mblock, K, N))
        t.sticky = false
        schedule(t)
        tasks[i] = t
        # Mstart += Mblock
        A = gesp(A, (Mblock, Zero()))
        C = gesp(C, (Mblock, Zero()))
    end
    mrange = Mstart:M-1
    Mrem = M - Mblock * to_spawn
    jmulpackAonly!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), Mrem, K, N)
    foreach(wait, tasks)
end

function jmultpackAB!(
    C::AbstractStridedPointer, A::AbstractStridedPointer, B::AbstractStridedPointer,
    α, β, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}, M, K, N, task_view
) where {Mc,Kc,Nc}
    W = VectorizationBase.pick_vector_width_val(T)
    mᵣW = StaticInt{mᵣ}() * W
    
    to_spawn = length(tasks)
    total_threads = to_spawn + 1
    # nspawn = length(task_view) + 1
    # Nsplits = cld_fast(cld_fast(M, mᵣW), nspawn)
    Nsplits = cld_fast(M, mᵣW * total_threads)
    if Nsplits > 1
        _Nsize = cld(N, Nsplits)
        Nsize = cld(N, cld(N, _Nsize))
        Nspawn_per = cld(Nsplits, total_threads)
        tv_start = Nsplits
        Nsplitsm1 = Nsplits-1
        for i in Base.OneTo(Nsplitsm1)
            _taskview = view(task_view, tv_start:tv_start+Nspawn_per-2)
            t = Task(PackABClosure(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), M, K, Nsize, _taskview))
            t.sticky = false
            schedule(t)
            task_view[i] = t
            tv_start += Nspawn_per - 1
            B = gesp(C, (Zero(), Nsize))
            C = gesp(C, (Zero(), Nsize))
        end
        Nrem = N - Nsize * Nsplitsm1
        _jmultpackAB!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), M, K, Nrem, _taskview)
        foreach(wait, view(task_view, Base.OneTo(Nsplitsm1)))
        return
    end

    atomicsync = Ref(UInt}(zero(UInt))
    p = Base.unsafe_convert(Ptr{UInt}, atomicsync)
    bc = _use_bcache()
    bc_ptr = Base.unsafe_convert(typeof(pointer(C)), pointer(bc))
    # Mblock = cld_fast(cld_fast(M, total_threads), MᵣW) * Mᵣ*W
    GC.@preserve atomicsync begin
        Mblock = cld_fast(cld_fast(M, total_threads), W) * W
        for i ∈ Base.OneTo(to_spawn)
            t = Task(SyncClosure(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), Mblock, K, N, bc_ptr, i % UInt, to_spawn % UInt))
            t.sticky = false
            schedule(t)
            # task_view[i] = t # shouldn't have to wait, should sync in there
            A = gesp(A, (Mblock, Zero()))
            C = gesp(C, (Mblock, Zero()))
        end
    end
    Mrem = M - to_spawn * Mblock
    sync_mul!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), Mrem, K, N, p, bc_ptr, zero(UInt), to_spawn % UInt)
    _free_bcache!(bc)
    return
end

struct SyncClosure{Mc,Kc,Nc,TC,TA,TB,Α,Β,Md,Kd,Nd,C}
    C::TC
    A::TA
    B::TB
    α::Α
    β::Β
    M::Md
    K::Kd
    N::Nd
    p::Ptr{UInt}
    bc::C
    id::UInt
    last_id::UInt
end

function SyncClosure{Mc,Kc,Nc}(
    C::TC, A::TA, B::TB, α::Α, β::Β, M::Md, K::Kd, N::Nd, p::Ptr{UInt}, bc::C, id::UInt, last_id::UInt
) where {Mc,Kc,Nc,TC,TA,TB,Α,Β,Md,Kd,Nd,C}
    SyncClosure{Mc,Kc,Nc,TC,TA,TB,Α,Β,Md,Kd,Nd,C}(C, A, B, α, β, M, K, N, p, bc, id, last_id)    
end

function (sc::SyncClosure{Mc,Kc,Nc})() where {Mc,Kc,Nc}
    sync_mul!(sc.C, sc.A, sc.B, sc.α, sc.β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), sc.M, sc.K, sc.N, sc.p, sc.bc, sc.id, sc.last_id)
end

function sync_mul!(
    C::AbstractStridedPointer, A::AbstractStridedPointer, B::AbstractStridedPointer,
    α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), Mblock, K, N, atomicp::Ptr{UInt}, bc::Ptr, id::UInt, last_id::UInt
)
    num_n_iter = cld_fast(N, StaticInt{nc}())
    _nreps_per_iter = div_fast(N, num_n_iter) + StaticInt{nᵣ}() - One()
    ncrepetitions = div_fast(_nreps_per_iter, StaticInt{nᵣ}())
    nreps_per_iter = StaticInt{nᵣ}() * ncrepetitions
    Niter = num_n_iter - One()
    Nrem = N - Niter * nreps_per_iter

    total_ids = last_id + one(UInt)
    flag = (one(UInt) << total_ids) - one(UInt)
    idflag = one(UInt) << id
    Npackblock_full = cld_fast(nreps_per_iter, total_ids)
    Npackblock_rem = nreps_per_iter - Npackblock_full * last_id
    Npackblockrem_full = cld_fast(Nrem, total_ids)
    Npackblockrem_rem = Nrem - Npackblock_full * last_id
    pack_offset = Npackblock_full*id
    pack_view = pack_offset:min(pack_offset + Npackblock_full, nreps_per_iter)
    pack_offset_rem = Npackblockrem_full*id
    pack_view_rem = pack_offset:min(pack_offset_rem + Npackblockrem_full, Nrem)
    # pack_offset is the offset of region this one is packing
    # Npackblock is the block size being packed
    
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
        
    # noffset = 0
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
            _pack_view = no == Niter ? pack_view_rem : pack_view
            # koffset = 0
            let B = B, C = C
                for ko ∈ 0:Kiter
                    ksize = ifelse(ko == 0, _Krem, _kreps_per_iter)
                    Bsubset2 = PtrArray(gesp(Bptr, (koffset, noffset)), (ksize, nsize), dense_dims_subset(dense_dims(B), stride_rank(B)))
                    Bpacked2 = ptrarray0(L3ptr, (ksize, nsize))
                    
                    copyto!(view(Bpacked2, :, _pack_view), view(Bsubset2, :, _pack_view))
                    _atomic_or!(atomicp, idflag)
                    while _atomic_load(atomicp) != flag
                        pause()
                    end
                    # moffset = 0
                    let A = A, B = stridedpointer(Bpacked2), C = C
                        for mo in 0:Miter
                            msize = ifelse(mo == Miter, _Mrem, _mreps_per_iter)
                            Asubset = PtrArray(gesp(Aptr, (moffset, koffset)), (msize, ksize), dense_dims_subset(dense_dims(A), stride_rank(A)))

                            Cpmat = PtrArray(gesp(Cptr, (moffset, noffset)), (msize, nsize), dense_dims_subset(dense_dims(C), stride_rank(C)))
                            if ko == 0
                                packaloopmul!(Cpmat, Asubset, Bpacked2, α, β, (zrange(msize), zrange(ksize), zrange(nsize)))
                            else
                                packaloopmul!(C, A, B, α, One(), (zrange(msize), zrange(ksize), zrange(nsize)))
                            end
                            A = gesp(A, (mreps_per_iter, Zero()))
                            C = gesp(C, (mreps_per_iter, Zero()))
                            # moffset += mreps_per_iter
                        end
                    end
                    _atomic_xor!(atomicp, idflag)
                    A = gesp(A, (Zero(), ksize))
                    B = gesp(A, (ksize, Zero()))
                    while _atomic_load(atomicp) != zero(UInt)
                        pause()
                    end
                    # koffset += ksize
                end
            end
            B = gesp(B, (Zero(), nreps_per_iter))
            C = gesp(C, (Zero(), nreps_per_iter))
            # noffset += nreps_per_iter
        end
    end # GC.@preserve
    nothing
end

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


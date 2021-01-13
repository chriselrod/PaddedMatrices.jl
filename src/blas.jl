

@inline cld_fast(x, y) = cld(x, y)
@inline function cld_fast(x::I, y) where {I <: Integer}
    # ux = unsigned(x); uy = unsigned(y)
    # d = Base.udiv_int(ux, uy)
    d = div_fast(x, y)
    (d + (d * unsigned(y) != unsigned(x))) % I
    # ifelse(d * uy == ux, d, d + one(d)) % I
end
cld_fast(::StaticInt{N}, ::StaticInt{M}) where {N,M}= (StaticInt{N}() + StaticInt{M}() + One()) ÷ StaticInt{M}()
@inline function divrem_fast(x::I, y) where {I <: Integer}
    ux = unsigned(x); uy = unsigned(y)
    d = Base.udiv_int(ux, uy)
    r = ux - d * uy
    d % I, r % I
end
@inline divrem_fast(::StaticInt{x}, y::I) where {x, I <: Integer} = divrem_fast(x % I, y)
@inline div_fast(x::I, y::Integer) where {I <: Integer} = Base.udiv_int(unsigned(x), unsigned(y)) % I
@inline div_fast(::StaticInt{x}, y::I) where {x, I <: Integer} = Base.udiv_int(unsigned(x), unsigned(y)) % I
# @inline div_fast(x::I, ::StaticInt{x}) where {x, I <: Integer} = Base.udiv_int(x % UInt32, y % UInt32) % I
divrem_fast(::StaticInt{N}, ::StaticInt{M}) where {N,M}= divrem(StaticInt{N}(), StaticInt{M}())
div_fast(::StaticInt{N}, ::StaticInt{M}) where {N,M}= StaticInt{N}() ÷ StaticInt{M}()
@generated function div_fast(x::I, ::StaticInt{M}) where {I<:Integer,M}
    if VectorizationBase.ispow2(M)
        lm = VectorizationBase.intlog2(M)
        Expr(:block, Expr(:meta,:inline), :(x >>> $lm))
    else
        Expr(:block, Expr(:meta,:inline), :(div_fast(x, $(I(M)))))
    end
end

function Base.copyto!(B::AbstractStrideArray{<:Any,<:Any,<:Any,N}, A::AbstractStrideArray{<:Any,<:Any,<:Any,N}) where {N}
    @avx for I ∈ eachindex(A, B)
        B[I] = A[I]
    end
    B
end

@generated _max(::StaticInt{N}, ::StaticInt{M}) where {N,M} = :(StaticInt{$(max(N,M))}())
const MᵣW_mul_factor = VectorizationBase.REGISTER_SIZE === 64 ? StaticInt{4}() : StaticInt{9}()


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
    α, β, M,K,N, ::StaticInt{mc}, ::StaticInt{kc}, ::StaticInt{nc}
) where {T, mc, kc, nc}
    W = VectorizationBase.pick_vector_width_val(T)
    mᵣW = StaticInt{mᵣ}() * W

    Miter = cld_fast(M, StaticInt{mc}())
    Mbsize, Mrem, Mremfinal, Mblocks = split_m(M, Miter, W)
    Mblock_Mrem, Mblock_ = promote(Mbsize + W, Mbsize)

    Kiter = cld_fast(K, StaticInt{kc}())
    Kblock, Krem = divrem_fast(K, Kiter)
    Kblock_Krem, Kblock_ = promote(Kblock + One(), Kblock)
    
    Aptr = zero_offsets(A); Bptr = zero_offsets(B); Cptr = zero_offsets(C);
    # @show Miter, Kiter, M, K, N
    for ko ∈ CloseOpen(Kiter)
        ksize = ifelse(ko < Krem, Kblock_Krem, Kblock_)
        let Aptr = Aptr, Cptr = Cptr
            for mo in CloseOpen(Miter)
                msize = ifelse(mo == (Miter - 1), Mremfinal, ifelse(mo < Mrem, Mblock_Mrem, Mblock_))
                if ko == 0
                    packaloopmul!(Cptr, Aptr, Bptr, α, β, msize, ksize, N)
                else
                    packaloopmul!(Cptr, Aptr, Bptr, α, One(), msize, ksize, N)
                end
                Aptr = gesp(Aptr, (msize, Zero()))
                Cptr = gesp(Cptr, (msize, Zero()))
            end
        end
        Aptr = gesp(Aptr, (Zero(), ksize))
        Bptr = gesp(Bptr, (ksize, Zero()))
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
    α, β, M, K, N, ::StaticInt{mc}, ::StaticInt{kc}, ::StaticInt{nc}
) where {T, mc, kc, nc}
    W = VectorizationBase.pick_vector_width_val(T)
    mᵣW = StaticInt{mᵣ}() * W
    
    num_n_iter = cld_fast(N, StaticInt{nc}())
    _nreps_per_iter = div_fast(N, num_n_iter) + StaticInt{nᵣ}() - One()
    nreps_per_iter = div_fast(_nreps_per_iter, StaticInt{nᵣ}()) * StaticInt{nᵣ}()
    Niter = num_n_iter - One()
    Nrem = N - Niter * nreps_per_iter

    num_m_iter = cld_fast(M, StaticInt{mc}())
    _mreps_per_iter = div_fast(M, num_m_iter) + mᵣW - One()
    mcrepetitions = div_fast(_mreps_per_iter, mᵣW)
    mreps_per_iter = mᵣW * mcrepetitions
    Miter = num_m_iter - One()
    Mrem = M - Miter * mreps_per_iter
    if (Mrem < mᵣW) & (Miter > 0)
        Miter -= one(Miter)
        Mrem += mreps_per_iter
    end

    num_k_iter = cld_fast(K, StaticInt{kc}())
    kreps_per_iter = ((div_fast(K, num_k_iter)) + 3) & -4
    Krem = K - (num_k_iter - One()) * kreps_per_iter
    Kiter = num_k_iter - One()
        
    Aptr = zero_offsets(A)
    Bptr = zero_offsets(B)
    Cptr = zero_offsets(C)
    # L3ptr = Base.unsafe_convert(Ptr{Tb}, pointer(BCACHE) + (Threads.threadid()-1)*BSIZE*8)
    # bc = _use_bcache()
    # L3ptr = Base.unsafe_convert(Ptr{T}, BCACHE)
    bcache = _use_bcache()
    L3ptr = Base.unsafe_convert(Ptr{T}, bcache)
    noffset = 0
    _Nrem, _nreps_per_iter = promote(Nrem, nreps_per_iter)
    _Mrem, _mreps_per_iter = promote(Mrem, mreps_per_iter)
    _Krem, _kreps_per_iter = promote(Krem, kreps_per_iter)
    # Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B);
    # GC.@preserve Cb Ab Bb BCACHE begin
    # @show T Niter, _Nrem, _nreps_per_iter Kiter, _Krem, _kreps_per_iter 
    # return _free_bcache!(bcache)
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
                    # Bsubset2 = PtrArray(Bptr, (ksize, nsize), dense_dims_subset(dense_dims(B), stride_rank(B)))
                    Bsubset2 = PtrArray(Bptr, (ksize, nsize), none_dense(Val{2}()))
                    Bpacked2 = ptrarray0(L3ptr, (ksize, nsize))
                    # @show offsets(Bpacked2), offsets(Bsubset2)
                    copyto!(Bpacked2, Bsubset2)
                    let Aptr = Aptr, Cptr = Cptr, Bptr = zstridedpointer(Bpacked2)
                        # moffset = 0
                        for mo in 0:Miter
                            msize = ifelse(mo == Miter, _Mrem, _mreps_per_iter)
                            # pack mreps_per_iter x kreps_per_iter block of A
                            # Asubset = PtrArray(gesp(Aptr, (moffset, koffset)), (msize, ksize), dense_dims_subset(dense_dims(A), stride_rank(A)))

                            # Cpmat = PtrArray(gesp(Cptr, (moffset, noffset)), (msize, nsize), dense_dims_subset(dense_dims(C), stride_rank(C)))
                            # packaloopmul!(Cpmat, Asubset, Bpacked2, α, _β, (zrange(msize), zrange(ksize), zrange(nsize)))
                            if ko == 0
                                packaloopmul!(Cptr, Aptr, Bptr, α, β, msize, ksize, nsize)
                            else
                                packaloopmul!(Cptr, Aptr, Bptr, α, One(), msize, ksize, nsize)
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
    _free_bcache!(bcache)
    nothing
end

@inline contiguousstride1(A) = ArrayInterface.contiguous_axis(A) === ArrayInterface.Contiguous{1}()
@inline contiguousstride1(A::AbstractStridedPointer{T,N,1}) where {T,N} = true
@inline firstbytestride(A::AbstractStridedPointer) = VectorizationBase.bytestrides(A)[One()]
# @inline firstbytestride(A::AbstractStrideArray) = bytestride(A, One())
# @inline firstbytestride(A::PermutedDimsArray) = LinearAlgebra.stride1(A)
# @inline firstbytestride(A::Adjoint{<:Any,<:AbstractMatrix}) = stride(parent(A), 2)
# @inline firstbytestride(A::Transpose{<:Any,<:AbstractMatrix}) = stride(parent(A), 2)
# @inline firstbytestride(::Any) = typemax(Int)

@inline function vectormultiple(bytex, ::Type{Tc}, ::Type{Ta}) where {Tc,Ta}
    Wc = VectorizationBase.pick_vector_width_val(Tc) * static_sizeof(Ta) - One()
    iszero(bytex & (VectorizationBase.REGISTER_SIZE - 1))
end
@inline function dontpack(pA::AbstractStridedPointer{Ta}, M, K, ::StaticInt{mc}, ::StaticInt{kc}, ::Type{Tc}) where {mc, kc, Tc, Ta}
    (contiguousstride1(pA) && 
         ((((VectorizationBase.AVX512F ? 9 : 13) * VectorizationBase.pick_vector_width(Tc)) ≥ M) ||
          (vectormultiple(bytestride(pA, StaticInt{2}()), Tc, Ta) && ((M * K) ≤ (mc * kc)) && iszero(reinterpret(Int, pointer(pA)) & (VectorizationBase.REGISTER_SIZE - 1)))))
end

@inline function matmul_serial(A::AbstractMatrix, B::AbstractMatrix)
    m = size(A, StaticInt{1}())
    p = size(B, StaticInt{2}())
    C = StrideArray{promote_type(eltype(A),eltype(B))}(undef, (m,p))
    matmul_serial!(C, A, B)
    return C
end

@inline function matmul_serial!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    maybeinline(C, A) && return inlineloopmul!(C, A, B, One(), Zero())
    matmul_serial!(C, A, B, One(), Zero(), ArrayInterface.is_column_major(C))
end
@inline function matmul_serial!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α)
    maybeinline(C, A) && return inlineloopmul!(C, A, B, α, Zero())
    matmul_serial!(C, A, B, α, Zero(), ArrayInterface.is_column_major(C))
end
@inline function matmul_serial!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β)
    maybeinline(C, A) && return inlineloopmul!(C, A, B, α, β)
    matmul_serial!(C, A, B, α, β, ArrayInterface.is_column_major(C))
end
@inline matmul_serial!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β, ::Val{false}) = (matmul_serial!(C', B', A', α, β, nothing); return C)
@inline matmul_serial!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β, ::Val{true}) = matmul_serial!(C, A, B, α, β, nothing)

"""
  matmul_serial!(C, A, B[, α = 1, β = 0])

Calculates `C = α * (A * B) + β * C` in place.

A single threaded matrix-matrix-multiply implementation.
Supports dynamically and statically sized arrays.

Organizationally, `matmul_serial!` checks the arrays properties to try and dispatch to an appropriate implementation.
If the arrays are small and statically sized, it will dispatch to an inlined multiply.

Otherwise, based on the array's size, whether they are transposed, and whether the columns are already aligned, it decides to not pack at all, to pack only `A`, or to pack both arrays `A` and `B`.
"""
@inline function matmul_serial!(
    C::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix, α, β, MKN::Union{Nothing,Tuple{Vararg{Integer,3}}}
) where {T}
    pA = zstridedpointer(A); pB = zstridedpointer(B); pC = zstridedpointer(C);
    Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B);
    (M,K,N) = MKN === nothing ? matmul_sizes(C, A, B) : MKN
    Mc, Kc, Nc = matmul_params(T)
    GC.@preserve Cb Ab Bb begin
        if VectorizationBase.CACHE_SIZE[2] === nothing ||  (nᵣ ≥ N) || dontpack(pA, M, K, Mc, Kc, T)
            loopmul!(pC, pA, pB, α, β, M, K, N)
            return C
        else
            jmul_singlethread_pack!(pC, pA, pB, α, β, M, K, N, Mc, Kc, Nc)
            return C
        end
    end
    return C
end # function

function jmul_singlethread_pack!(pC, pA, pB, α, β, M,K,N, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}) where {Mc,Kc,Nc}
    if VectorizationBase.CACHE_SIZE[3] === nothing || (contiguousstride1(pB) ? (Kc * Nc ≥ K * N) : (firstbytestride(pB) ≤ 1600))
        # println("Pack A mul")
        jmulpackAonly!(pC, pA, pB, α, β, M, K, N, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}())
    else
        # println("Pack A and B mul")
        jmulpackAB!(pC, pA, pB, α, β, M, K, N, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}())
    end
    nothing
end

@inline function matmul!(
    C::AbstractStrideArray{S,D,T,2,2},
    A::AbstractMatrix,
    B::AbstractMatrix
) where {S, D, T}
    matmul!(C', B', A')
    C
end


"""
    matmul(A, B)

Multiply matrices `A` and `B`.
"""
@inline function matmul(A, B)
    M = size(A, StaticInt{1}())
    N = size(B, StaticInt{2}())
    C = StrideArray{promote_type(eltype(A),eltype(B))}(undef, (M, N))
    matmul!(C, A, B)
    return C
end

"""
    matmul!(C, A, B[, α, β, max_threads])

Calculates `C = α * A * B + β * C` in place, overwriting the contents of `A`.
It may use up to `max_threads` threads. It will not use threads when nested in other threaded code.
"""
@inline function matmul!(C, A, B)
    maybeinline(C, A) && return inlineloopmul!(C, A, B, One(), Zero())
    matmul!(C, A, B, One(), Zero(), nothing, ArrayInterface.is_column_major(C))
end
@inline function matmul!(C, A, B, α)
    maybeinline(C, A) && return inlineloopmul!(C, A, B, α, Zero())
    matmul!(C, A, B, α, Zero(), nothing, ArrayInterface.is_column_major(C))
end
@inline function matmul!(C, A, B, α, β)
    maybeinline(C, A) && return inlineloopmul!(C, A, B, α, β)
    matmul!(C, A, B, α, β, nothing, ArrayInterface.is_column_major(C))
end
@inline function matmul!(C, A, B, α, β, nthread)
    maybeinline(C, A) && return inlineloopmul!(C, A, B, α, β)
    matmul!(C, A, B, α, β, nthread, ArrayInterface.is_column_major(C))
end
@inline matmul!(C::AbstractMatrix, A, B, α, β, nthread, ::Val{false}) = (matmul!(C', B', A', α, β, nthread, nothing); return C)
@inline matmul!(C::AbstractMatrix, A, B, α, β, nthread, ::Val{true}) = matmul!(C, A, B, α, β, nthread, nothing)


@inline function dontpack(pA::AbstractStridedPointer{Ta}, M, K, ::StaticInt{mc}, ::StaticInt{kc}, ::Type{Tc}, nspawn) where {mc, kc, Tc, Ta}
    # MᵣW = VectorizationBase.pick_vector_width_val(Tc) * StaticInt{mᵣ}()
    # TODO: perhaps consider K vs kc by themselves?
    (contiguousstride1(pA) && ((M * K) ≤ (mc * kc) * nspawn >>> 1))
end

@inline function matmul!(C::AbstractMatrix{T}, A, B, α, β, nthread, matmuldims) where {T}#::Union{Nothing,Tuple{Vararg{Integer,3}}}) where {T}
    # Don't allow nested `jmult!`, and if we were padded `nthread` of 1, don't thread

    M, K, N = matmuldims === nothing ? matmul_sizes(C, A, B) : matmuldims
    W = VectorizationBase.pick_vector_width_val(T)
    # Assume 3 μs / spawn cost
    # 15W approximate GFLOPS target
    # 22500 = 3e-6μ/spawn * 15 / 2e-9
    Mc, Kc, Nc = matmul_params(T)
    pA = zstridedpointer(A); pB = zstridedpointer(B); pC = zstridedpointer(C);
    Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B);
    
    GC.@preserve Cb Ab Bb begin
        # if N_too_small | (MKN < (StaticInt{5451776}() ÷ static_sizeof(T)))
        N_too_small = nᵣ ≥ N
        # I'm really just being silly with these `@goto`s. Do they help?
        N_too_small && @goto LOOPMUL
        MKN = M*K*N
        do_not_thread = (MKN < (StaticInt{110592}() * W))
        if do_not_thread
            dontpack(pA, M, K, Mc, Kc, T) && @goto LOOPMUL
            jmulpackAonly!(pC, pA, pB, α, β, M, K, N, Mc, Kc, Nc)
            return C
        end
        # Not taking the fast path
        # But maybe we don't want to thread anyway
        nt = _nthreads()
        _nthread = nthread === nothing ? nt : min(_nthreads(), nthread)
        # Maybe this is nested, or we have ≤ 1 threads
        if (_nthread ≤ 1) || (!iszero(ccall(:jl_in_threaded_region, Cint, ())))
            jmul_singlethread_pack!(pC, pA, pB, α, β, M, K, N, Mc, Kc, Nc)
            return C
        end
        # We are threading, but how many threads?
        # nspawn = min(_nthread, div_fast(M + N, StaticInt{4}() * W))
        # nspawn = clamp(div_fast(M + N, StaticInt{4}() * W), 1, NUM_CORES)
        nspawn = clamp(div_fast((M + N)*StaticInt{11}(), StaticInt{64}() * W), 1, NUM_CORES)
        # L = StaticInt{22500}() * W
        # nspawn = min(_nthread, cld_fast(MKN, L))
        _matmul!(pC, pA, pB, α, β, nspawn, M, K, N, Mc, Kc, Nc)
        return C
        @label LOOPMUL
        loopmul!(pC, pA, pB, α, β, M, K, N)
        return C
    end
end

function jmultsplitn!(C::AbstractStridedPointer{T}, A, B, α, β, ::StaticInt{Mc}, nspawn, M, K, N, ::Val{PACK}) where {T, Mc, PACK}
    Mᵣ = StaticInt{mᵣ}(); Nᵣ = StaticInt{nᵣ}();
    W = VectorizationBase.pick_vector_width_val(T)
    MᵣW = Mᵣ*W

    _Mblocks, _Nblocks = divide_blocks(M, cld_fast(N, Nᵣ), nspawn, W)
    Mbsize, Mrem, Mremfinal, Mblocks = split_m(M, _Mblocks, W)
    Nblocks = min(N, _Nblocks)
    Nbsize, Nrem = divrem(N, Nblocks)
    
    _nspawn = Mblocks * Nblocks
    _nspawn_m1 = _nspawn - One()
    Mbsize_Mrem, Mbsize_ = promote(Mbsize + W, Mbsize)
    Nbsize_Nrem, Nbsize_ = promote(Nbsize + One(), Nbsize)
    
    tnum = 0
    # @show Mblocks, Nblocks, (Mbsize,Mrem), (Nbsize,Nrem)
    let _A = A, _B = B, _C = C
        for n ∈ Base.OneTo(Nblocks)
            nsize = ifelse(n ≤ Nrem, Nbsize_Nrem, Nbsize_)
            let _A = _A, _C = _C
                for m ∈ Base.OneTo(Mblocks)
                    msize = ifelse(m == Mblocks, Mremfinal, ifelse(m ≤ Mrem, Mbsize_Mrem, Mbsize_))
                    if tnum == _nspawn_m1
                        wait(runfunc(LoopMulClosure{PACK}(_C, _A, _B, α, β, msize, K, nsize), _nspawn))
                        waitonmultasks(CloseOpen(_nspawn))
                        return
                    end
                    runfunc!(LoopMulClosure{PACK}(_C, _A, _B, α, β, msize, K, nsize), (tnum += 1))
                    _A = gesp(_A, (msize, Zero()))
                    _C = gesp(_C, (msize, Zero()))
                end
            end
            _B = gesp(_B, (Zero(), nsize))
            _C = gesp(_C, (Zero(), nsize))
        end
    end
end

function _matmul!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer,
    α, β, nspawn, M, K, N, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}
) where {T,Mc,Kc,Nc}
    Mᵣ = StaticInt{mᵣ}(); Nᵣ = StaticInt{nᵣ}();
    W = VectorizationBase.pick_vector_width_val(T)
    MᵣW = Mᵣ*W
    # nkern = cld_fast(M * N,  MᵣW * Nᵣ)
    # @show nspawn
    # @show VectorizationBase.CACHE_SIZE[2] === nothing ||  (nᵣ ≥ N) || 
    #     dontpack(A, M, K, StaticInt{Mc}(), StaticInt{Kc}(), T) || # do not pack A
    #     (N * M ≤ (Nc * MᵣW) * nspawn) # do we just want to split aggressively?
    # @show (contiguousstride1(B) && (Kc * Nc ≥ K * N)) | (firstbytestride(B) > 1600) # do not pack B
    # @show M + MᵣW ≤ MᵣW * nspawn # do we want to split `N`
    #
    # Approach:
    # Check if we don't want to pack A,
    #    if not, aggressively subdivide
    # if so, check if we don't want to pack B
    #    if not, check if we want to thread `N` loop anyway
    #       if so, divide `M` first, then use ratio of desired divisions / divisions along `M` to calc divisions along `N`
    #       if not, only thread along `M`. These don't need syncing, as we're not packing `B`
    #    if so, `jmultpackAB!`
    #
    # MᵣW * (MᵣW_mul_factor - One()) # gives a smaller Mc, then
    # if 2M/nspawn is less than it, we don't don't `A`
    # First check is: do we just want to split aggressively?
    if VectorizationBase.CACHE_SIZE[2] === nothing ||  # do not pack A
        dontpack(A, M, K, StaticInt{Mc}(), StaticInt{Kc}(), T, nspawn) || (W ≥ M) || (nᵣ*nspawn ≥ N)
        # `nᵣ*nspawn ≥ N` is needed at the moment to avoid accidentally splitting `N` to be `< nᵣ` while packing
        # Should probably handle that with a smarter splitting function...
        jmultsplitn!(C, A, B, α, β, StaticInt{Mc}(), nspawn, M, K, N, Val{false}())
    elseif contiguousstride1(B) ? ((Kc * Nc - StaticInt{200_000}()) ≥ K * N) : (firstbytestride(B) ≤ 1600)
        jmultsplitn!(C, A, B, α, β, StaticInt{Mc}(), nspawn, M, K, N, Val{true}())
    else
        jmultpackAB!(
            C, A, B, α, β, M, K, N, CloseOpen(nspawn),
            StaticFloat{W₁Default}(), StaticFloat{W₂Default}(), StaticFloat{R₁Default}(), StaticFloat{R₂Default}(), Val{CACHE_COUNT[3]}()
        )
    end
end

# If tasks is [0,1,2,3] (e.g., `CloseOpen(0,4)`), it will wait on `MULTASKS[i]` for `i = [1,2,3]`.
function waitonmultasks(tasks::CloseOpen)
    tid = Int(first(tasks))
    while (tid += 1) < tasks.upper
        @inbounds wait(MULTASKS[tid])
    end
end

function jmultpackAB!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer, α, β, M, K, N,
    tasks::CloseOpen, ::StaticFloat{W₁}, ::StaticFloat{W₂}, ::StaticFloat{R₁}, ::StaticFloat{R₂}, ::Val{CC}
) where {T,W₁,W₂,R₁,R₂,CC}
    @assert CC > 1
    
    W = VectorizationBase.pick_vector_width_val(T)
    mᵣW = StaticInt{mᵣ}() * W

    to_spawn = length(tasks)
    # to_spawn = to_spawn + 1
    # nspawn = length(task_view) + 1
    # Nsplits = cld_fast(cld_fast(M, mᵣW), nspawn)
    _Nsplits = cld_fast(M, mᵣW * to_spawn)
    _Nsplits > 1 || return jmultpackAB!(C, A, B, α, β, M, K, N, tasks, StaticFloat{W₁}(), StaticFloat{W₂}(), StaticFloat{R₁}(), StaticFloat{R₂}(), Val{1}())
    Nsplits = min(_Nsplits, CC)

    _Nsize, Nrem = divrem_fast(N, Nsplits)
    Nsize_Nrem, Nsize_ = promote(Nsize + One(), Nsize)

    Nspawn_per = cld_fast(Nsplits, to_spawn)
    task_start = 0
    for i ∈ CloseOpen(Nsplits)
        task_next = min(task_start + Nspawn_per, tasks.upper)
        _taskview = CloseOpen(task_start, task_next)
        task_start = task_next
        nsize = ifelse(i < Nrem, Nsize_Nrem, Nsize_)
        jmultpackAB!(C, A, B, α, β, M, K, nsize, _taskview, StaticFloat{W₁}(), StaticFloat{W₂}(), StaticFloat{R₁}(), StaticFloat{R₂}(), Val{1}())
        B = gesp(C, (Zero(), nsize))
        C = gesp(C, (Zero(), nsize))
    end
end
function jmultpackAB!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer, α, β, M, K, N,
    tasks::CloseOpen, ::StaticFloat{W₁}, ::StaticFloat{W₂}, ::StaticFloat{R₁}, ::StaticFloat{R₂}, ::Val{1}
) where {T,W₁,W₂,R₁,R₂}
    W = VectorizationBase.pick_vector_width_val(T)
    mᵣW = StaticInt{mᵣ}() * W

    to_spawn = length(tasks)
    atomicsync = Ref{NTuple{9,UInt}}()
    p = Base.unsafe_convert(Ptr{UInt}, atomicsync)
    _atomic_min!(p, zero(UInt)); _atomic_min!(p + 8sizeof(UInt), zero(UInt))
    # unsafe_store!(p, zero(UInt), 1); unsafe_store!(p, zero(UInt), 9)
    Mbsize, Mrem, Mremfinal, _to_spawn = split_m(M, to_spawn, W) # M is guaranteed to be > W because of `W ≥ M` condition for `jmultsplitn!`...
    Mblock_Mrem, Mblock_ = promote(Mbsize + W, Mbsize)
    u_to_spawn = _to_spawn % UInt
    ft = first(tasks)
    tid = Int(ft)
    # @show Mbsize, Mrem, Mremfinal, _to_spawn
    # return
    bc = _use_bcache()
    bc_ptr = Base.unsafe_convert(typeof(pointer(C)), pointer(bc))
    # Mblock = cld_fast(cld_fast(M, to_spawn), MᵣW) * Mᵣ*W
    GC.@preserve atomicsync begin
        for i ∈ CloseOpen(One(), _to_spawn) # ...thus the fact that `CloseOpen()` iterates at least once is okay.
            Mblock = ifelse(i ≤ Mrem, Mblock_Mrem, Mblock_)
            runfunc!(SyncClosure{W₁,W₂,R₁,R₂}(C, A, B, α, β, Mblock, K, N, p, bc_ptr, i % UInt, u_to_spawn), (tid += 1))
            A = gesp(A, (Mblock, Zero()))
            C = gesp(C, (Mblock, Zero()))
        end
        wait(runfunc(SyncClosure{W₁,W₂,R₁,R₂}(C, A, B, α, β, Mremfinal, K, N, p, bc_ptr, zero(UInt), u_to_spawn), (tid += 1)))
        waitonmultasks(CloseOpen(ft, tid))
    end
    _free_bcache!(bc)
    return
end

function sync_mul!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer, α, β, M, K, N, atomicp::Ptr{UInt}, bc::Ptr, id::UInt, total_ids::UInt,
    ::StaticFloat{W₁}, ::StaticFloat{W₂}, ::StaticFloat{R₁}, ::StaticFloat{R₂}
) where {T, W₁, W₂, R₁, R₂}

    (Mblock, Mrem, Mremfinal, Miter), (Kblock, Kblock_Krem, Krem, Kiter), (Nblock, Nblock_Nrem, Nrem, Niter) =
        solve_block_sizes(T, M, K, N, StaticFloat{W₁}(), StaticFloat{W₂}(), StaticFloat{R₁}(), StaticFloat{R₂}())
    Mblock_Mrem = Mblock + VectorizationBase.pick_vector_width_val(T)
    
    last_id = total_ids - one(UInt)
    atomics = atomicp + 8sizeof(UInt)
    sync_iters = zero(UInt64)

    Npackb_r_div, Npackb_r_rem = divrem_fast(Nblock_Nrem, total_ids)
    Npackb_r_block_rem, Npackb_r_block_ = promote(Npackb_r_div + One(), Npackb_r_div)
    
    Npackb___div, Npackb___rem = divrem_fast(Nblock, total_ids)
    Npackb___block_rem, Npackb___block_ = promote(Npackb___div + One(), Npackb___div)

    pack_r_offset = Npackb_r_div * id + min(id, Npackb_r_rem)
    pack_r_view = CloseOpen(pack_r_offset, pack_r_offset + ifelse(id < Npackb_r_rem, Npackb_r_block_rem, Npackb_r_block_))
    pack___offset = Npackb___div * id + min(id, Npackb___rem)
    pack___view = CloseOpen(pack___offset, pack___offset + ifelse(id < Npackb___rem, Npackb___block_rem, Npackb___block_))

    GC.@preserve BCACHE begin
        for no in CloseOpen(Niter)
            # Krem
            # pack kc x nc block of B
            nfull = no < Nrem
            nsize = ifelse(nfull, Nblock_Nrem, Nblock)
            pack_view = ifelse(nfull, pack_r_view, pack___view)
            let A = A, B = B#, C = C
                for ko ∈ CloseOpen(Kiter)
                    ksize = ifelse(ko < Krem, Kblock_Krem, Kblock)

                    Bsubset2 = PtrArray(B, (ksize, nsize), none_dense(Val{2}()))
                    Bpacked2 = ptrarray0(bc, (ksize, nsize))
                    copyto!(zview(Bpacked2, :, pack_view), zview(Bsubset2, :, pack_view))
                    # synchronize before starting the multiplication, to ensure `B` is packed
                    sync_iters += total_ids
                    _mv = _atomic_add!(atomicp, one(UInt))
                    while _mv < sync_iters
                        pause()
                        _mv = _atomic_max!(atomicp, zero(UInt))
                    end
                    # multiply
                    let A = A, B = zstridedpointer(Bpacked2), C = C
                        for mo in CloseOpen(Miter)
                            # @show mo, ko, no
                            msize = ifelse(mo == (Miter-1), Mremfinal, ifelse(mo < Mrem, Mblock_Mrem, Mblock))
                            if ko == 0
                                packaloopmul!(C, A, B, α,     β, msize, ksize, nsize)
                            else
                                packaloopmul!(C, A, B, α, One(), msize, ksize, nsize)
                            end
                            A = gesp(A, (msize, Zero()))
                            C = gesp(C, (msize, Zero()))
                        end
                    end
                    A = gesp(A, (Zero(), ksize))
                    B = gesp(B, (ksize, Zero()))
                    # synchronize on completion so we wait until every thread is done with `Bpacked` before beginning to overwrite it
                    _mv = _atomic_add!(atomics, one(UInt))
                    while _mv < sync_iters
                        pause()
                        _mv = _atomic_max!(atomics, zero(UInt))
                    end
                end
            end
            B = gesp(B, (Zero(), nsize))
            C = gesp(C, (Zero(), nsize))
        end
    end # GC.@preserve
    nothing
end

maybeinline(::Any, ::Any) = false
function maybeinline(C::AbstractStrideMatrix{Tuple{StaticInt{M}, StaticInt{N}},<:Any,T}, ::AbstractStrideMatrix{<:Any,<:Any,<:Any,1}) where {T,M,N}
    static_sizeof(T) * StaticInt{M}() * StaticInt{N}() < StaticInt{176}() * StaticInt(mᵣ) * StaticInt{nᵣ}()
end
function maybeinline(C::AbstractStrideMatrix{Tuple{StaticInt{M}, StaticInt{N}},<:Any,T}, ::AbstractStrideMatrix) where {T,M,N}
    StaticInt{M}() * static_sizeof(T) ≤ StaticInt{2}() * StaticInt{VectorizationBase.REGISTER_SIZE}()
end

@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::StridedMatrix, B::StridedMatrix) = matmul!(C, A, B)
@inline LinearAlgebra.mul!(C::StridedMatrix, A::AbstractStrideMatrix, B::StridedMatrix) = matmul!(C, A, B)
@inline LinearAlgebra.mul!(C::StridedMatrix, A::StridedMatrix, B::AbstractStrideMatrix) = matmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractStrideMatrix, B::StridedMatrix) = matmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::StridedMatrix, B::AbstractStrideMatrix) = matmul!(C, A, B)
@inline LinearAlgebra.mul!(C::StridedMatrix, A::AbstractStrideMatrix, B::AbstractStrideMatrix) = matmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractStrideMatrix, B::AbstractStrideMatrix) = matmul!(C, A, B)

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
    matmul!(C, A, B)
    C
end

@inline extract_λ(a) = a
@inline extract_λ(a::UniformScaling) = a.λ
@inline function Base.:*(A::AbstractStrideArray{S,D,T}, bλ::Union{Tb,UniformScaling{Tb}}) where {S,D,T<:VectorizationBase.NativeTypes,Tb <: Real}
    mv = similar(A)
    b = T(extract_λ(bλ))
    @avx for i ∈ eachindex(A)
        mv[i] = A[i] * b
    end
    mv
end

function LinearAlgebra.mul!(
    C::AbstractStrideMatrix{<:Any,<:Any,T},
    A::LinearAlgebra.Diagonal{T,<:AbstractStrideVector{<:Any,T}},
    B::StridedMatrix{T}
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


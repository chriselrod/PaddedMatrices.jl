

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
    # r = vsub(ux, vmul(d, uy))
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
@inline zstridedpointer(A) = VectorizationBase.zero_offsets(stridedpointer(A))

@generated _max(::StaticInt{N}, ::StaticInt{M}) where {N,M} = :(StaticInt{$(max(N,M))}())
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
    _kc = ((StaticInt{795}() * L₂) ÷ StaticInt{1024}() - StaticInt{30420}()) ÷ mc
    kc = _max(_kc, StaticInt{120}())
    L₃ = if CACHE_INCLUSIVITY[3]
        something(cache_size(T, Val(3)), StaticInt{3145728}() ÷ static_sizeof(T)) - L₂
    else
        something(cache_size(T, Val(3)), StaticInt{3145728}() ÷ static_sizeof(T))
    end
    _nc = ((((StaticInt{132}() * L₃) ÷ StaticInt{125}()) - StaticInt{256651}()) ÷ (kc * Nᵣ)) * Nᵣ
    nc = _max(_nc, StaticInt{400}())
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
    α, β, ::StaticInt{mc}, ::StaticInt{kc}, ::StaticInt{nc}, (M,K,N)
) where {T, mc, kc, nc}
    W = VectorizationBase.pick_vector_width_val(T)
    mᵣW = StaticInt{mᵣ}() * W
    num_m_iter = cld_fast(M, StaticInt{mc}())
    _mreps_per_iter = div_fast(M, num_m_iter) + mᵣW - One()
    mcrepetitions = _mreps_per_iter ÷ mᵣW
    mreps_per_iter = mᵣW * mcrepetitions
    Miter = num_m_iter - One()
    Mrem = M - Miter * mreps_per_iter
    if (Mrem < mᵣW) & (Miter > 0)
        Miter -= one(Miter)
        Mrem += mreps_per_iter
    end
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
    # @show Miter, Kiter, M, K, N
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
    α, β, ::StaticInt{mc}, ::StaticInt{kc}, ::StaticInt{nc}, (M, K, N)# = matmul_sizes(C, A, B)
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
@inline function dontpack(ptrA::Ptr{Ta}, M, K, Xa, ::StaticInt{mc}, ::StaticInt{kc}, ::Type{Tc}) where {mc, kc, Tc, Ta}
    mc_mult = ((VectorizationBase.AVX512F ? 9 : 13) * VectorizationBase.pick_vector_width(Tc)) ≥ M
    mc_mult || (vectormultiple(Xa, Tc, Ta) && ((M * K) ≤ (mc * kc)) && iszero(reinterpret(Int, ptrA) & (VectorizationBase.REGISTER_SIZE - 1)))
end

@inline function jmul(A::AbstractMatrix, B::AbstractMatrix)
    m = size(A, StaticInt{1}())
    p = size(B, StaticInt{2}())
    C = StrideArray{promote_type(eltype(A),eltype(B))}(undef, (m,p))
    jmul!(C, A, B)
    return C
end

@inline function jmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    maybeinline(C, A) && return inlineloopmul!(C, A, B, One(), Zero())
    jmul!(C, A, B, One(), Zero(), stride_rank(C))
end
@inline function jmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α)
    maybeinline(C, A) && return inlineloopmul!(C, A, B, α, Zero())
    jmul!(C, A, B, α, Zero(), stride_rank(C))
end
@inline function jmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β)
    maybeinline(C, A) && return inlineloopmul!(C, A, B, α, β)
    jmul!(C, A, B, α, β, stride_rank(C))
end
jmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β, ::StrideRank{(2,1)}) = (jmul!(C', B', A', α, β, nothing); return C)
jmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β, ::StrideRank) = jmul!(C, A, B, α, β, nothing)

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
    C::AbstractMatrix{Tc}, A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}, α, β, MKN::Union{Nothing,Tuple{Vararg{Integer,3}}}
) where {Tc, Ta, Tb}
    pA = zstridedpointer(A); pB = zstridedpointer(B); pC = zstridedpointer(C);
    Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B);
    (M,K,N) = MKN === nothing ? matmul_sizes(C, A, B) : MKN
    Mc, Kc, Nc = matmul_params(Tc)
    GC.@preserve Cb Ab Bb begin
        if VectorizationBase.CACHE_SIZE[2] === nothing || ((nᵣ ≥ N) || (contiguousstride1(pA) && dontpack(pointer(pA), M, K, bytestride(pA,StaticInt{2}()), Mc, Kc, Tc)))
            loopmul!(pC, pA, pB, α, β, (CloseOpen(M),CloseOpen(K),CloseOpen(N)))
        elseif VectorizationBase.CACHE_SIZE[3] === nothing || (((contiguousstride1(pB) && (Kc * Nc ≥ K * N))) || firstbytestride(pB) ≤ 1600)
            # println("Pack A mul")
            jmulpackAonly!(pC, pA, pB, α, β, Mc, Kc, Nc, (M,K,N))
        else
            # println("Pack A and B mul")
            jmulpackAB!(pC, pA, pB, α, β, Mc, Kc, Nc, (M,K,N))
        end
    end
    return C
end # function

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

struct LoopMulClosure{P,TC,TA,TB,Α,Β,Md,Kd,Nd}
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
    LoopMulClosure{P}(zstridedpointer(C), zstridedpointer(A), zstridedpointer(B), α, β, M, K, N)
end
# function LoopMulClosure{true}(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β, M, K, N)
#     LoopMulClosure{true}(C, A, B, α, β, M, K, N)
# end
(m::LoopMulClosure{false})() = loopmul!(m.C, m.A, m.B, m.α, m.β, (CloseOpen(m.M), CloseOpen(m.K), CloseOpen(m.N)))
# (m::LoopMulClosure{true})() = jmulpackAonly!(m.C, m.A, m.B, m.α, m.β, (m.Maxis, m.Kaxis, m.Naxis))
function (m::LoopMulClosure{true,TC})() where {T,TC <: AbstractStridedPointer{T}}
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

@inline function jmult(A, B)
    m = size(A, StaticInt{1}())
    p = size(B, StaticInt{2}())
    C = StrideArray{promote_type(eltype(A),eltype(B))}(undef, (m,p))
    jmult!(C, A, B)
    return C
end

@inline function jmult!(C, A, B)
    maybeinline(C, A) && return inlineloopmul!(C, A, B, One(), Zero())
    jmult!(C, A, B, One(), Zero(), nothing, stride_rank(C))
end
@inline function jmult!(C, A, B, α)
    maybeinline(C, A) && return inlineloopmul!(C, A, B, α, Zero())
    jmult!(C, A, B, α, Zero(), nothing, stride_rank(C))
end
@inline function jmult!(C, A, B, α, β)
    maybeinline(C, A) && return inlineloopmul!(C, A, B, α, β)
    jmult!(C, A, B, α, β, nothing, stride_rank(C))
end
@inline function jmult!(C, A, B, α, β, nthread)
    maybeinline(C, A) && return inlineloopmul!(C, A, B, α, β)
    jmult!(C, A, B, α, β, nthread, stride_rank(C))
end
jmult!(C::AbstractMatrix, A, B, α, β, nthread, ::StrideRank{(2, 1)}) = (jmult!(C', B', A', α, β, nthread, nothing); return C)
jmult!(C::AbstractMatrix, A, B, α, β, nthread, ::StrideRank) = jmult!(C, A, B, α, β, nthread, nothing)

function jmult!(C::AbstractMatrix{T}, A, B, α, β, nthread, matmuldims) where {T}#::Union{Nothing,Tuple{Vararg{Integer,3}}}) where {T}
    # Don't allow nested `jmult!`, and if we were padded `nthread` of 1, don't thread
    nt = _nthreads()
    _nthread = nthread === nothing ? nt : min(_nthreads(), nthread)

    (!iszero(ccall(:jl_in_threaded_region, Cint, ())) || (_nthread ≤ 1)) && return jmul!(C, A, B, α, β, matmuldims)

    M, K, N = matmuldims === nothing ? matmul_sizes(C, A, B) : matmuldims
    W = VectorizationBase.pick_vector_width_val(T)
    # Assume 3 μs / spawn cost
    # 15W approximate GFLOPS target
    # 22500 = 3e-6μ/spawn * 15 / 2e-9
    L = StaticInt{22500}() * W
    MKN = M*K*N
    suggested_threads = cld_fast(MKN, L)
    if suggested_threads ≤ 1
        # We convert to `PtrArray`s here to reduce recompilation
        loopmul!(zstridedpointer(C), zstridedpointer(A), zstridedpointer(B), α, β, (CloseOpen(M),CloseOpen(K),CloseOpen(N)))
        return C
    elseif MKN * static_sizeof(T) < 5451776  # Only start threading beyond about 88x88 * 88x88 for `Float64`, or 111x111 * 111x111 for `Float32`
        return jmul!(C, A, B, α, β, (M, K, N))
    end
    nspawn = min(_nthread, suggested_threads)
    Mc, Kc, Nc = matmul_params(T)
    Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B)
    GC.@preserve Cb Ab Bb begin
        _jmult!(
            zstridedpointer(C), zstridedpointer(A), zstridedpointer(B),
            α, β, nspawn, (M,K,N), Mc, Kc, Nc
        )
    end
    return C
end

@inline function gcd_fast(a::T, b::T) where {T<:Base.BitInteger}
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
@inline function divide_blocks(M, N, ::StaticInt{Mb}, ::StaticInt{Nb}, nspawn) where {Mb,Nb}
    Mfull, Mrem = divrem_fast(M, Mb)
    Mtotal = Mfull + (Mrem > 0)

    Nfull, Nrem = divrem_fast(N, Nb)
    Ntotal = Nfull + (Nrem > 0)
    
    divide_blocks(Mtotal, Ntotal, nspawn)
end
@inline function divide_blocks(Mtotal, Ntotal, _nspawn)
    Miter = gcd_fast(_nspawn, Mtotal)
    nspawn = div_fast(_nspawn, Miter)
    # Niter = gcd_fast(nspawn, Ntotal)
    Niter = cld_fast(Ntotal, cld_fast(Ntotal, nspawn))
    return Miter, Niter
end

function jmultdonotpacka!(C::AbstractStridedPointer{T}, A, B, α, β, nspawn, (M,K,N)) where {T}
    Mᵣ = StaticInt{mᵣ}(); Nᵣ = StaticInt{nᵣ}();
    W = VectorizationBase.pick_vector_width_val(T)
    MᵣW = Mᵣ*W

    _Mtotal = cld_fast(M, MᵣW)
    
    Mblocks, Nblocks = divide_blocks(_Mtotal, cld_fast(N, Nᵣ), nspawn)
    _nspawn = Mblocks * Nblocks
    _nspawn_m1 = _nspawn - One()
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
                        wait(runfunc(LoopMulClosure{false}(_C, _A, _B, α, β, msize, K, nsize), _nspawn))
                        # loopmul!(_C, _A, _B, α, β, (CloseOpen(msize), CloseOpen(K), CloseOpen(nsize)))
                        waitonmultasks(CloseOpen(Zero(), _nspawn))
                        return
                    end
                    runfunc!(LoopMulClosure{false}(_C, _A, _B, α, β, msize, K, nsize), (tnum += 1))
                    _A = gesp(_A, (msize, Zero()))
                    _C = gesp(_C, (msize, Zero()))
                end
            end
            _B = gesp(_B, (Zero(), nsize))
            _C = gesp(_C, (Zero(), nsize))
        end
    end
end
function jmultsplitn!(C::AbstractStridedPointer{T}, A, B, α, β, nspawn, (M,K,N), ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}) where {T, Mc, Kc, Nc}
    Mᵣ = StaticInt{mᵣ}(); #Nᵣ = StaticInt{nᵣ}();
    W = VectorizationBase.pick_vector_width_val(T)
    MᵣW = Mᵣ*W

    _Mtotal = cld_fast(M, MᵣW)
    # split_n = nspawn > _Mtotal
    _Niter = cld_fast(nspawn, _Mtotal)
    _Nper = cld(N, _Niter)
    
    nspawn_per = cld_fast(nspawn, _Niter)
    spawn_start = 0#spawn_per_mrep
    Nper = _Nper
    Nrem = N - (Nper * (_Niter - 1))
    for i ∈ Base.OneTo(_Niter)
        Nsize = ifelse(i == _Niter, Nrem, Nper)
        next_start = ifelse(i == _Niter, nspawn, spawn_start + nspawn_per)
        task_view = CloseOpen(spawn_start, next_start)
        jmultpackAonly!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), M, K, Nsize, task_view)
        spawn_start = next_start
        B = gesp(B, (Zero(), Nper))
        C = gesp(C, (Zero(), Nper))
    end
    # task_view = CloseOpen(spawn_start, nspawn)
    # jmultpackAonly!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), M, K, Nrem, task_view)
    nothing
end

function _jmult!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer,
    α, β, nspawn, (M,K,N), ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}
) where {T,Mc,Kc,Nc}
    Mᵣ = StaticInt{mᵣ}(); Nᵣ = StaticInt{nᵣ}();
    W = VectorizationBase.pick_vector_width_val(T)
    MᵣW = Mᵣ*W
    # nkern = cld_fast(M * N,  MᵣW * Nᵣ)

    # Approach:
    # Check if we don't want to pack A,
    #    if not, aggressively subdivide
    # if so, check if we don't want to pack B
    #    if not, check if we want to thread `N` loop anyway
    #       if so, divide `M` first, then use ratio of desired divisions / divisions along `M` to calc divisions along `N`
    #       if not, only thread along `M`. These don't need syncing, as we're not packing `B`
    #    if so, `jmultpackAB!`
    if VectorizationBase.CACHE_SIZE[2] === nothing ||
        ((nᵣ ≥ N) || (contiguousstride1(A) && (dontpack(pointer(A), M, K, bytestride(A,StaticInt{2}()), StaticInt{Mc}(), StaticInt{Kc}(), T)))) || # do not pack A
        (N * M ≤ (Nc * MᵣW) * nspawn) # do we just want to split aggressively?
        jmultdonotpacka!(C, A, B, α, β, nspawn, (M,K,N))
    elseif (contiguousstride1(B) && (Kc * Nc ≥ K * N)) | (firstbytestride(B) > 1600) # do not pack B
        if M + MᵣW ≤ MᵣW * nspawn # do we want to split `N`
            jmultsplitn!(C, A, B, α, β, nspawn, (M,K,N), StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}())
        else
            jmultpackAonly!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), M, K, N, CloseOpen(0, nspawn))
        end
    else
        jmultpackAB!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), M, K, N, CloseOpen(0, nspawn), Val{CACHE_COUNT[3]}())
    end
end

# If tasks is [0,1,2,3] (e.g., `CloseOpen(0,4)`), it will wait on `MULTASKS[i]` for `i = [1,2,3]`.
function waitonmultasks(tasks::CloseOpen)
    tid = Int(first(tasks))
    while (tid += 1) < tasks.upper
        @inbounds wait(MULTASKS[tid])
    end
end

# _front(r::AbstractUnitRange) = CloseOpen(first(r), last(r))
"""
Runs on threadids `tasks .+ 1`
"""
function jmultpackAonly!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer,
    α, β, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}, M, K, N, tasks::CloseOpen
) where {T,Mc,Kc,Nc}
    # e.g., tasks = CloseOpen(0,4) = [0,1,2,3] runs on [1,2,3,4]
    to_spawn = length(tasks)
    
    W = VectorizationBase.pick_vector_width_val(T)
    _Mblock = cld_fast(M, to_spawn)
    Mblock = cld_fast(_Mblock, W) * W
    _Miter = M ÷ Mblock
    _Mrem = M - Mblock * _Miter
    if iszero(_Mrem)
        Mrem = Mblock
        Miter = _Miter - One()
    else
        Mrem = _Mrem
        Miter = _Miter
    end
    tid = ft = first(tasks)
    for _ ∈ Base.OneTo(Miter)
        runfunc!(LoopMulClosure{true}(C, A, B, α, β, Mblock, K, N), (tid += 1))
        A = gesp(A, (Mblock, Zero()))
        C = gesp(C, (Mblock, Zero()))        
    end
    wait(runfunc(LoopMulClosure{true}(C, A, B, α, β, Mrem, K, N), (tid += 1)))
    # jmulpackAonly!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), (Mrem, K, N))
    waitonmultasks(CloseOpen(ft, tid))
end

# open_upper(r::CloseOpen) = r.upper
function jmultpackAB!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer,
    α, β, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}, M, K, N, tasks::CloseOpen, ::Val{CC}
) where {T,Mc,Kc,Nc,CC}
    @assert CC > 1
    
    W = VectorizationBase.pick_vector_width_val(T)
    mᵣW = StaticInt{mᵣ}() * W

    to_spawn = length(tasks)
    # to_spawn = to_spawn + 1
    # nspawn = length(task_view) + 1
    # Nsplits = cld_fast(cld_fast(M, mᵣW), nspawn)
    Nsplits = min(cld_fast(M, mᵣW * to_spawn), CC)
    Nsplits > 1 || return jmultpackAB!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), M, K, N, tasks, Val{1}())
    
    _Nsize = cld_fast(N, Nsplits)
    Nsize = cld_fast(N, cld_fast(N, _Nsize))
    Nspawn_per = cld_fast(Nsplits, to_spawn)
    task_start = 0
    Nsplitsm1 = Nsplits - One()
    Nrem = N - Nsize * Nsplitsm1
    for i ∈ Base.OneTo(Nsplitsm1)#CloseOpen(Nsplits)
        task_next = min(task_start + Nspawn_per, tasks.upper)
        _taskview = CloseOpen(task_start, task_next)
        task_start = task_next
        nsize = i == Nsplitsm1 ? Nrem : Nsize
        jmultpackAB!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), M, K, nsize, _taskview, Val{1}())
        
        B = gesp(C, (Zero(), Nsize))
        C = gesp(C, (Zero(), Nsize))
    end
end
function jmultpackAB!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer,
    α, β, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}, M, K, N, tasks::CloseOpen, ::Val{1}
) where {T,Mc,Kc,Nc}
    W = VectorizationBase.pick_vector_width_val(T)
    mᵣW = StaticInt{mᵣ}() * W

    to_spawn = length(tasks)
    atomicsync = Ref{NTuple{9,UInt}}()
    p = Base.unsafe_convert(Ptr{UInt}, atomicsync)
    _atomic_store!(p, zero(UInt)); _atomic_store!(p + 8sizeof(UInt), zero(UInt))
    # unsafe_store!(p, zero(UInt), 1); unsafe_store!(p, zero(UInt), 9)
    
    bc = _use_bcache()
    bc_ptr = Base.unsafe_convert(typeof(pointer(C)), pointer(bc))
    # Mblock = cld_fast(cld_fast(M, to_spawn), MᵣW) * Mᵣ*W
    GC.@preserve atomicsync begin
        Mblock = cld_fast(cld_fast(M, to_spawn), W) * W
        _Miter = M ÷ Mblock
        _Mrem = M - Mblock * _Miter
        if iszero(_Mrem)
            Mrem = Mblock
            Miter = _Miter - One()
        else
            Mrem = _Mrem
            Miter = _Miter
        end
        tid = ft = first(tasks)
        _to_spawn = (Miter + 1) % UInt
        for i ∈ Base.OneTo(Miter)
            runfunc!(SyncClosure{Mc,Kc,Nc}(C, A, B, α, β, Mblock, K, N, p, bc_ptr, i % UInt, _to_spawn), (tid += 1))
            A = gesp(A, (Mblock, Zero()))
            C = gesp(C, (Mblock, Zero()))
        end
        wait(runfunc(SyncClosure{Mc,Kc,Nc}(C, A, B, α, β, Mrem, K, N, p, bc_ptr, zero(UInt), _to_spawn), (tid += 1)))
        waitonmultasks(CloseOpen(ft, tid))
    end
    _free_bcache!(bc)
    return
end

# struct PackABClosure{Mc,Kc,Nc,TC,TA,TB,Α,Β,Md,Kd,Nd,V}
#     C::TC
#     A::TA
#     B::TB
#     α::Α
#     β::Β
#     M::Md
#     K::Kd
#     N::Nd
#     tv::V
# end
# function PackABClosure{Mc,Kc,Nc}(
#     C::TC, A::TA, B::TB, α::Α, β::Β, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}, M::Md, K::Kd, N::Nd, _taskview::V
# ) where {Mc,Kc,Nc,TC,TA,TB,Α,Β,Md,Kd,Nd,V}
#     PackABClosure{Mc,Kc,Nc,TC,TA,TB,Α,Β,Md,Kd,Nd,V}(C, A, B, α, β, M, K, N, _taskview)
# end
# function (m::PackABClosure{Mc,Kc,Nc})() where {Mc,Kc,Nc}
#     jmultpackAB!(m.C, m.A, m.B, m.α, m.β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), m.M, m.K, m.N, m.tv, Val{1}())
# end

struct SyncClosure{Mc,Kc,Nc,TC,TA,TB,Α,Β,Md,Kd,Nd,CA}
    C::TC
    A::TA
    B::TB
    α::Α
    β::Β
    M::Md
    K::Kd
    N::Nd
    p::Ptr{UInt}
    bc::Ptr{CA}
    id::UInt
    last_id::UInt
end

function SyncClosure{Mc,Kc,Nc}(
    C::TC, A::TA, B::TB, α::Α, β::Β, M::Md, K::Kd, N::Nd, p::Ptr{UInt}, bc::Ptr{CA}, id::UInt, last_id::UInt
) where {Mc,Kc,Nc,TC,TA,TB,Α,Β,Md,Kd,Nd,CA}
    SyncClosure{Mc,Kc,Nc,TC,TA,TB,Α,Β,Md,Kd,Nd,CA}(C, A, B, α, β, M, K, N, p, bc, id, last_id)    
end

function (sc::SyncClosure{Mc,Kc,Nc})() where {Mc,Kc,Nc}
    sync_mul!(sc.C, sc.A, sc.B, sc.α, sc.β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), sc.M, sc.K, sc.N, sc.p, sc.bc, sc.id, sc.last_id)
end

function sync_mul!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer,
    α, β, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}, M, K, N, atomicp::Ptr{UInt}, bc::Ptr, id::UInt, total_ids::UInt
) where {T, Mc, Kc, Nc}
    num_n_iter = cld_fast(N, StaticInt{Nc}())
    _nreps_per_iter = div_fast(N, num_n_iter) + StaticInt{nᵣ}() - One()
    nreps_per_iter = div_fast(_nreps_per_iter, StaticInt{nᵣ}()) * StaticInt{nᵣ}()
    Niter = num_n_iter - One()
    Nrem = N - Niter * nreps_per_iter

    last_id = total_ids - one(UInt)
    # total_ids = last_id + one(UInt)
    # flag = (one(UInt) << total_ids) - one(UInt)
    # xchngflag = flag | (id << (4sizeof(UInt)))
    # idflag = one(UInt) << id
    
    # cmp_id = id << (4sizeof(UInt))
    # cmp_id_next = (id + one(UInt))  << (4sizeof(UInt))
    # cmp_id = iszero(id) ? total_ids : id << (4sizeof(UInt))
    # cmp_id_next = id == last_id ? zero(UInt64) : (id+one(UInt)) << (4sizeof(UInt))
    atomics = atomicp + 8sizeof(UInt)
    sync_iters = zero(UInt64)
    # ids are 0, 1, 2, 3; last_id = 3
    # cmp_ids init as = id
    # on each iter
    # cmp_id += total_ids
    # cmp_ids = 4, 5, 6, 7
    # everyone atomic adds, so we have atomicp[] == 4
    # then _atomic_cas_cmp!(atomicp, cmp_ids, cmp_id_next)
    # so now cmp_ids == 8;
    # later, to sync finishing
    # _atomic_cas_cmp!(atomicp, cmp_id_next, cmp_ids)
    
    Npackblock_full = cld_fast(nreps_per_iter, total_ids)
    Npackblock_rem = nreps_per_iter - Npackblock_full * last_id
    Npackblockrem_full = cld_fast(Nrem, total_ids)
    Npackblockrem_rem = Nrem - Npackblock_full * last_id
    pack_offset = Npackblock_full*id
    pack_view = CloseOpen(pack_offset, min((pack_offset + Npackblock_full) % UInt, nreps_per_iter % UInt))
    pack_offset_rem = Npackblockrem_full*id
    pack_view_rem = CloseOpen(pack_offset_rem, min((pack_offset_rem + Npackblockrem_full) % UInt, Nrem % UInt))
    # @show id, pack_view, pack_view_rem, id%Int
    # pack_offset is the offset of region this one is packing
    # Npackblock is the block size being packed
    mᵣW = StaticInt{mᵣ}() * VectorizationBase.pick_vector_width_val(T)
    num_m_iter = cld_fast(M, StaticInt{Mc}())
    _mreps_per_iter = div_fast(M, num_m_iter) + mᵣW - One()
    mcrepetitions = div_fast(_mreps_per_iter, mᵣW)
    mreps_per_iter = mᵣW * mcrepetitions
    Miter = num_m_iter - One()
    Mrem = M - Miter * mreps_per_iter
    if (Mrem < mᵣW) & (Miter > 0)
        Miter -= one(Miter)
        Mrem += mreps_per_iter
    end
    
    num_k_iter = cld_fast(K, StaticInt{Kc}())
    kreps_per_iter = ((div_fast(K, num_k_iter)) + 3) & -4
    Krem = K - (num_k_iter - One()) * kreps_per_iter
    Kiter = num_k_iter - One()
        
    _Nrem, _nreps_per_iter = promote(Nrem, nreps_per_iter)
    _Mrem, _mreps_per_iter = promote(Mrem, mreps_per_iter)
    _Krem, _kreps_per_iter = promote(Krem, kreps_per_iter)
    # stuck = 0; max_iter = 199_999_999;
    # @show Miter, Kiter, Niter, id%Int mreps_per_iter, kreps_per_iter, nreps_per_iter, id%Int Mrem, Krem, Nrem, id%Int
    # @show (Miter, Kiter, Niter) .* (mreps_per_iter, kreps_per_iter, nreps_per_iter) .+ (Mrem, Krem, Nrem), id%Int
    # return
    GC.@preserve BCACHE begin
        for no in 0:Niter
            # Krem
            # pack kc x nc block of B
            nsize = ifelse(no == Niter, _Nrem, _nreps_per_iter)
            _pack_view = no == Niter ? pack_view_rem : pack_view
            let A = A, B = B#, C = C
                for ko ∈ 0:Kiter
                    ksize = ifelse(ko == 0, _Krem, _kreps_per_iter)
                    Bsubset2 = PtrArray(B, (ksize, nsize), none_dense(Val{2}()))
                    Bpacked2 = ptrarray0(bc, (ksize, nsize))
                    copyto!(zview(Bpacked2, :, _pack_view), zview(Bsubset2, :, _pack_view))
                    
                    # flagcmp = _atomic_or!(atomicp, idflag)
                    # # atomic_fence()
                    # while flagcmp != flag
                    #     # @show _atomic_load(atomicp), id, last_id, flag
                    #     flagcmp = _atomic_max!(atomicp, zero(UInt))
                    #     pause()
                    #     ((stuck += 1) > max_iter) && error("$id stuck for $stuck iterations")
                    # end
                    # @show 0, id, time_ns()
                    sync_iters += total_ids
                    _mv = _atomic_add!(atomicp, one(UInt))
                    while _mv < sync_iters
                        pause()
                        # _mv = _atomic_min!(atomicp, sync_iters)
                        _mv = _atomic_max!(atomicp, zero(UInt))
                        # mv = _mv % Int; sv = sync_iters % Int
                        # @show 0, (id%Int), threadid()-1, mv, sv
                        # ((stuck += 1) > max_iter) && error("$id stuck for $stuck iterations")
                        # ccall(:jl_wakeup_thread, Cvoid, (Int16,), 0 % Int16)
                        # ccall(:jl_wakeup_thread, Cvoid, (Int16,), 1 % Int16)
                        # ccall(:jl_wakeup_thread, Cvoid, (Int16,), 2 % Int16)
                        # ccall(:jl_wakeup_thread, Cvoid, (Int16,), 3 % Int16)
                    end
                    # @show 1, id, time_ns()
                    # cmp_id += total_ids
                    # cmp_id_next += total_ids
                    # atomic_fence()
                    # while _atomic_load(atomicp) < sync_iters
                    # while _atomic_min!(atomicp, sync_iters) < sync_iters
                    #     # atomic_fence()
                    #     mv = _atomic_min!(atomicp, sync_iters) % Int; sv = sync_iters%Int
                    #     @show 0, (id%Int), threadid()-1, mv, sv
                    #     pause()
                    #     # sleep(1e-3)
                    #     ((stuck += 1) > max_iter) && error("$id stuck for $stuck iterations")
                    # end
                    # while !_atomic_cas_cmp!(atomicp, cmp_id, cmp_id_next)
                    #     # cl = cmp_id % Int32; cu = (cmp_id >> 32) % Int32
                    #     # aplu = _atomic_load(atomicp); apl = aplu % Int32; apu = (aplu >> 32)%Int32
                    #     # @show 0, id%Int, apl, apu, cl, cu
                    #     # @assert _atomic_load(atomicp) ≤ cmp_id "atomicp too large for $id, !($(_atomic_load(atomicp)) ≤ $cmp_id"
                    #     pause()
                    #     # ((stuck += 1) > max_iter) && error("$id stuck for $stuck iterations")
                    #     # sleep(1e-2)
                    # end
                    let A = A, B = zstridedpointer(Bpacked2), C = C
                        for mo in 0:Miter
                            # @show mo, ko, no
                            msize = ifelse(mo == Miter, _Mrem, _mreps_per_iter)
                            if ko == 0
                                packaloopmul!(C, A, B, α,     β, (msize, ksize, nsize))
                            else
                                packaloopmul!(C, A, B, α, One(), (msize, ksize, nsize))
                            end
                            A = gesp(A, (mreps_per_iter, Zero()))
                            C = gesp(C, (mreps_per_iter, Zero()))
                            # moffset += mreps_per_iter
                        end
                    end
                    # @assert _atomic_xor!(atomicp, idflag) != zero(UInt)
                    # 
                    A = gesp(A, (Zero(), ksize))
                    B = gesp(B, (ksize, Zero()))
                    # atomic_fence()
                    # flagcmp = _atomic_xor!(atomicp, idflag)
                    # while flagcmp != zero(UInt)
                    #     # @show _atomic_load(atomicp), id, last_id
                    #     flagcmp = _atomic_max!(atomicp, zero(UInt))
                    #     pause()
                    #     ((stuck += 1) > max_iter) && error("$id stuck for $stuck iterations")
                    # end

                    _mv = _atomic_add!(atomics, one(UInt))
                    # while _atomic_load(atomics) < sync_iters
                    while _mv < sync_iters
                        pause()
                        # _mv = _atomic_min!(atomics, sync_iters)
                        _mv = _atomic_max!(atomics, zero(UInt))
                        # atomic_fence()
                        # mv = _atomic_min!(atomicp, sync_iters) % Int; sv = sync_iters%Int
                        # @show 1, (id%Int), threadid()-1, mv, sv
                        # pause()
                        # # sleep(1e-3)
                        # ((stuck += 1) > max_iter) && error("$id stuck for $stuck iterations")
                        # ccall(:jl_wakeup_thread, Cvoid, (Int16,), 0 % Int16)
                        # ccall(:jl_wakeup_thread, Cvoid, (Int16,), 1 % Int16)
                        # ccall(:jl_wakeup_thread, Cvoid, (Int16,), 2 % Int16)
                        # ccall(:jl_wakeup_thread, Cvoid, (Int16,), 3 % Int16)
                    end
                    # atomic_fence()
                    # while !_atomic_cas_cmp!(atomics, cmp_id, cmp_id_next)
                        # cl = cmp_id % Int32; cu = (cmp_id >> 32) % Int32
                        # aslu = _atomic_load(atomics); asl = aslu % Int32; asu = (aslu >> 32)%Int32
                        # @show 1, id%Int, asl, asu, cl, cu
                        # @assert _atomic_load(atomics) ≤ cmp_id "atomics too large, !($(_atomic_load(atomics)) ≤ $cmp_id"
                        # pause()
                        # ((stuck += 1) > max_iter) && error("$id stuck for $stuck iterations")
                        # sleep(1e-2)
                    # end
                    # stuck = 0
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


@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::StridedMatrix, B::StridedMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::StridedMatrix, A::AbstractStrideMatrix, B::StridedMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::StridedMatrix, A::StridedMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::AbstractStrideMatrix, B::StridedMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::StridedMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)
@inline LinearAlgebra.mul!(C::StridedMatrix, A::AbstractStrideMatrix, B::AbstractStrideMatrix) = jmul!(C, A, B)
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
@inline function Base.:*(A::AbstractStrideArray{S,D,T}, bλ::Union{Tb,UniformScaling{Tb}}) where {S,D,T<:VectorizationBase.NativeTypes,Tb <: Real}
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


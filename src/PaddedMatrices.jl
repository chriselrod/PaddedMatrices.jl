

module PaddedMatrices

using VectorizationBase, ArrayInterface,
    SLEEFPirates, VectorizedRNG,
    LoopVectorization, LinearAlgebra,
    Random, Base.Threads#, StackPointers#,
    # SpecialFunctions # Perhaps there is a better way to support erf?

using VectorizationBase: align, gep, AbstractStridedPointer, AbstractSIMDVector, vnoaliasstore!, staticm1,
    static_sizeof, lazymul, vmul, vadd, vsub, StridedPointer, gesp, zero_offsets, NUM_CORES, CACHE_INCLUSIVITY, pause
using LoopVectorization: maybestaticsize, mᵣ, nᵣ, preserve_buffer, CloseOpen
using ArrayInterface: StaticInt, Zero, One, OptionallyStaticUnitRange, size, strides, offsets, indices,
    static_length, static_first, static_last, axes,
    dense_dims, DenseDims, stride_rank, StrideRank
# using Threads: @spawn
# import ReverseDiffExpressionsBase:
    # RESERVED_INCREMENT_SEED_RESERVED!, ∂getindex,
    # alloc_adjoint, uninitialized, initialized, isinitialized
# import LoopVectorization: isdense

# using Parameters: @unpack

export @StrideArray, @gc_preserve, # @Constant,
    AbstractStrideArray, AbstractStrideVector, AbstractStrideMatrix,
    StrideArray, StrideVector, StrideMatrix,
    PtrArray,# PtrVector, PtrMatrix,
    # ConstantArray, ConstantVector, ConstantMatrix, allocarray,
    jmul!, mul!, *ˡ, StaticInt
# LazyMap, 



include("l3_cache_buffer.jl")
include("type_declarations.jl")
include("size_and_strides.jl")
include("adjoints.jl")
include("stridedpointers.jl")
include("indexing.jl")
include("initialization.jl")
include("views.jl")
include("rand.jl")
include("kernels.jl")
include("blas.jl")
include("broadcast.jl")
include("miscellaneous.jl")


# Commented, because I don't want this to be the only doc string.
# """
# To find a mode, define methods for `logdensity` and logdensity_and_gradient!` dispatching on obj, and evaluating at the position `q`.

# logdensity(obj, q, [::StackPointer])
# ∂logdensity!(∇, obj, q, [::StackPointer])

# These must return a value (eg, a logdensity). logdensity_and_gradient! should store the gradient in ∇.
# """
function logdensity end
function ∂logdensity! end


const BCACHE = Float64[]
const MAX_THREADS = min(64, NUM_CORES)
const MULTASKS = Vector{Task}[]
const BSYNCHRONIZERS = Atomic{UInt}[]
_nthreads() = min(MAX_THREADS, nthreads())
_preallocated_tasks() = MULTASKS[threadid()]

function __init__()
    # resize!(ACACHE, ASIZE * Threads.nthreads())
    # resize!(BCACHE, BSIZE * Threads.nthreads())
    resize!(BCACHE, BSIZE * BCACHE_COUNT)
    nthread = nthreads()
    resize!(MULTASKS, nthread)
    resize!(BSYNCHRONIZERS, nthread)
    @threads for t ∈ Base.OneTo(nthread)
        MULTASKS[t] = Vector{Task}(undef, _nthreads()-1)
        BSYNCHRONIZERS[t] = Atomic{UInt}(zero(UInt))
    end
# #    set_zero_subnormals(true)
#     page_size = ccall(:jl_getpagesize, Int, ())
    # resize!(LCACHEARRAY, ((L₂ + L₃) >>> 3) * Threads.nthreads() + page_size)
    # resize!
#     LCACHE[] = VectorizationBase.align(pointer(LCACHEARRAY), page_size)
#     @assert iszero(reinterpret(UInt, LCACHE[]) % page_size)
#   #  resize!(BCACHE, (L₃ >>> 3) )#* Threads.nthreads())
#     # @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" @eval using PaddedMatricesForwardDiff
end

# include("precompile.jl")
# _precompile_()



end # module

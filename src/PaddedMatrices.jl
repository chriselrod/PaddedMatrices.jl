

module PaddedMatrices

using VectorizationBase, ArrayInterface,
    SLEEFPirates, VectorizedRNG,
    LoopVectorization, LinearAlgebra,
    Random#, StackPointers#,
    # SpecialFunctions # Perhaps there is a better way to support erf?

using VectorizationBase: align, gep, AbstractStridedPointer, AbstractSIMDVector, vnoaliasstore!, staticm1,
    static_sizeof, lazymul, vmul, vadd, vsub, StridedPointer, gesp
using LoopVectorization: maybestaticsize, mᵣ, nᵣ, preserve_buffer
using ArrayInterface: StaticInt, Zero, One, OptionallyStaticUnitRange, size, strides, offsets, indices,
    static_length, static_first, static_last, axes,
    dense_dims, DenseDims, stride_rank, StrideRank
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

# include("zeroinitialized.jl")
# include("stack_pointers.jl")
# include("mappedtypes.jl")
# include("lazy_maps.jl")

include("miscellaneous.jl")
# include("gradsupport.jl")
# include("linear_algebra.jl")


"""
To find a mode, define methods for `logdensity` and logdensity_and_gradient!` dispatching on obj, and evaluating at the position `q`.

logdensity(obj, q, [::StackPointer])
∂logdensity!(∇, obj, q, [::StackPointer])

These must return a value (eg, a logdensity). logdensity_and_gradient! should store the gradient in ∇.
"""
function logdensity end
function ∂logdensity! end




# function logdensity(x)
#     s = zero(first(x))
#     @avx for i ∈ eachindex(x)
#         s += x[i]*x[i]
#     end
#     -0.5s
# end
# function ∂logdensity!(∂x, x)
#     s = zero(first(x))
#     @avx for i ∈ eachindex(x)
#         s += x[i]*x[i]
#         ∂x[i] = -x[i]
#     end
#     -0.5s
# end

# include("precompile.jl")
# _precompile_()

# Initialize on package load, else precompile will store excessively sized objects
# const L1CACHE = Float64[]
# const L3CACHE = Float64[]



# function cache_sizes()
#     L₁, L₂, L₃, L₄ = VectorizationBase.CACHE_SIZE
#     # # L₃ ÷= VectorizationBase.NUM_CORES # L₃ is shared, L₁ and L₂ are note
#     # align.((L₁, L₂, L₃), ccall(:jl_getpagesize, Int, ()))
# end
# const L₁, L₂, L₃, L₄ = cache_sizes()
# const LCACHEARRAY = Float64[]
# const LCACHE = Ref{Ptr{Float64}}()

# function threadlocal_L2CACHE_pointer(::Type{T} = Float64, threadid = Threads.threadid()) where {T}
#     Base.unsafe_convert(Ptr{T}, LCACHE[]) + (L₂ + L₃) * (threadid - 1)
# end
# function threadlocal_L3CACHE_pointer(::Type{T} = Float64, threadid = Threads.threadid()) where {T}
#     Base.unsafe_convert(Ptr{T}, LCACHE[]) + (L₂ + L₃) * (threadid - 1) + L₂
# end
#=


const ACACHE = Float64[]
const BCACHE = Float64[]
A_pointer(::Type{T}) where {T} = Base.unsafe_convert(Ptr{T}, pointer(ACACHE))
function A_pointer(::Type{T}, ::Val{M}, ::Val{K}, Miter, m, k) where {T, M, K}
    ptrA = Base.unsafe_convert(Ptr{T}, pointer(ACACHE))
    MKst = M*K*sizeof(T)
    ptrA + MKst * (m + (Miter+1) * k)
end
function resize_Acache!(::Type{T}, ::Val{M}, ::Val{K}, Miter, Kiter) where {T, M, K}
    # sizeof(T) * M should be a multiple of REGISTER_SIZE, therefore the `>>> 3` will be exact division by 8
    L = (sizeof(T) * M * K * (Miter + 1) * (Kiter + 1)) >>> 3
    L > length(ACACHE) && resize!(ACACHE, L)
    nothing
end
# function B_pointer(::Type{T} = Float64, threadid = Threads.threadid()) where {T}
    # Base.unsafe_convert(Ptr{T}, pointer(BCACHE)) + L₃ * (threadid - 1)
# end
function B_pointer(::Type{T} = Float64) where {T}
    Base.unsafe_convert(Ptr{T}, pointer(BCACHE))# + L₃ * (threadid - 1)
end
=#
# const ACACHE = Float64[]
# const ASIZE = something(Int(core_cache_size(Float64, Val(2))), 163840);
const BCACHE = Float64[]
const BSIZE = Int(something(core_cache_size(Float64, Val(3)), 393216));
function __init__()
    # resize!(ACACHE, ASIZE * Threads.nthreads())
    resize!(BCACHE, BSIZE * Threads.nthreads())
# #    set_zero_subnormals(true)
#     page_size = ccall(:jl_getpagesize, Int, ())
    # resize!(LCACHEARRAY, ((L₂ + L₃) >>> 3) * Threads.nthreads() + page_size)
    # resize!
#     LCACHE[] = VectorizationBase.align(pointer(LCACHEARRAY), page_size)
#     @assert iszero(reinterpret(UInt, LCACHE[]) % page_size)
#   #  resize!(BCACHE, (L₃ >>> 3) )#* Threads.nthreads())
#     # @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" @eval using PaddedMatricesForwardDiff
end




end # module

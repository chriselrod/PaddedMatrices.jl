

module PaddedMatrices

using VectorizationBase, SIMDPirates,
    SLEEFPirates, VectorizedRNG,
    LoopVectorization, LinearAlgebra,
    Random, StackPointers#,
    # SpecialFunctions # Perhaps there is a better way to support erf?

using VectorizationBase: Static, StaticUnitRange, align, gep, offset, AbstractStructVec, AbstractStridedPointer, AbstractSIMDVector, vnoaliasstore!
using LoopVectorization: maybestaticsize
# import ReverseDiffExpressionsBase:
    # RESERVED_INCREMENT_SEED_RESERVED!, ∂getindex,
    # alloc_adjoint, uninitialized, initialized, isinitialized
# import LoopVectorization: isdense

# using Parameters: @unpack

export @FixedSize, # @Constant,
    AbstractStrideArray, AbstractStrideVector, AbstractStrideMatrix,
    StrideArray, StrideVector, StrideMatrix,
    FixedSizeArray, FixedSizeVector, FixedSizeMatrix,
    PtrArray, PtrVector, PtrMatrix,
    # ConstantArray, ConstantVector, ConstantMatrix,
    LazyMap, muladd!, mul!, *ˡ, Static


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

include("stack_pointers.jl")
include("mappedtypes.jl")
include("lazy_maps.jl")
# include("gradsupport.jl")
# include("linear_algebra.jl")

function logdensity(x)
    s = zero(first(x))
    @avx for i ∈ eachindex(x)
        s += x[i]*x[i]
    end
    -0.5s
end
function ∂logdensity!(∂x, x)
    s = zero(first(x))
    @avx for i ∈ eachindex(x)
        s += x[i]*x[i]
        ∂x[i] = -x[i]
    end
    -0.5s
end

# include("precompile.jl")
# _precompile_()

# Initialize on package load, else precompile will store excessively sized objects
# const L1CACHE = Float64[]
# const L3CACHE = Float64[]
function cache_sizes()
    L₁, L₂, L₃ = VectorizationBase.CACHE_SIZE
    # L₃ ÷= VectorizationBase.NUM_CORES # L₃ is shared, L₁ and L₂ are not
    align.((L₁, L₂, L₃))
end
const L₁, L₂, L₃ = cache_sizes()

const LCACHEARRAY = Float64[]
const LCACHE = Ref{Ptr{Float64}}()

function threadlocal_L2CACHE_pointer(::Type{T} = Float64, threadid = Threads.threadid()) where {T}
    Base.unsafe_convert(Ptr{T}, LCACHE[]) + (L₂ + L₃) * (threadid - 1)
end
function threadlocal_L3CACHE_pointer(::Type{T} = Float64, threadid = Threads.threadid()) where {T}
    Base.unsafe_convert(Ptr{T}, LCACHE[]) + (L₂ + L₃) * (threadid - 1) + L₂
end
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

function __init__()
#    set_zero_subnormals(true)
    page_size = ccall(:jl_getpagesize, Int, ())
    resize!(LCACHEARRAY, ((L₂ + L₃ * VectorizationBase.NUM_CORES) >>> 3) * Threads.nthreads() + page_size)
    LCACHE[] = VectorizationBase.align(pointer(LCACHEARRAY), page_size)
    @assert iszero(reinterpret(UInt, LCACHE[]) % page_size)
  #  resize!(BCACHE, (L₃ >>> 3) )#* Threads.nthreads())
    # @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" @eval using PaddedMatricesForwardDiff
end




end # module

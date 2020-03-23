

module PaddedMatrices

using VectorizationBase, SIMDPirates,
    SLEEFPirates, VectorizedRNG,
    LoopVectorization, LinearAlgebra,
    Random, StackPointers#,
    # SpecialFunctions # Perhaps there is a better way to support erf?

using VectorizationBase: Static, StaticUnitRange, align, gep, offset, AbstractStructVec, AbstractStridedPointer, AbstractSIMDVector
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
    LazyMap, muladd!, mul!, *ˡ


include("type_declarations.jl")
include("size_and_strides.jl")
include("adjoints.jl")
include("stridedpointers.jl")
include("indexing.jl")
include("initialization.jl")
include("views.jl")
include("rand.jl")
include("blas.jl")
include("broadcast.jl")

include("stack_pointers.jl")
include("mappedtypes.jl")
include("lazy_maps.jl")
include("gradsupport.jl")
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
const L2CACHE = Float64[]
const L3CACHE = Float64[]

function pack_B!(B::AbstractMatrix{T}, threadid = 1) where {T}
    
end

function cache_sizes()
    L₁, L₂, L₃ = VectorizationBase.CACHE_SIZE
    L₃ ÷= VectorizationBase.NUM_CORES # L₃ is shared, L₁ and L₂ are not
    L₁, L₂, L₃
end

function __init__()
    set_zero_subnormals(true)
    L1, L2, L3 = cache_sizes()
    nthread = Threads.nthreads()
    # resize!(L1CACHE, ((VectorizationBase.align(L1) >>> 3) * nthread))
    resize!(L2CACHE, ((VectorizationBase.align(L2) >>> 3) * nthread))
    resize!(L3CACHE, ((VectorizationBase.align(L3) >>> 3) * nthread))
    # @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" @eval using PaddedMatricesForwardDiff
end




end # module

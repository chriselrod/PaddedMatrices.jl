

module PaddedMatrices

using VectorizationBase, SIMDPirates,
    SLEEFPirates, VectorizedRNG,
    LoopVectorization, LinearAlgebra,
    Random, StackPointers#,
    # SpecialFunctions # Perhaps there is a better way to support erf?

using VectorizationBase: Static, StaticUnitRange, align, gep
# import ReverseDiffExpressionsBase:
    # RESERVED_INCREMENT_SEED_RESERVED!, ∂getindex,
    # alloc_adjoint, uninitialized, initialized, isinitialized
# import LoopVectorization: isdense

# using Parameters: @unpack

export @Mutable, # @Constant,
    AbstractStrideArray, AbstractStrideVector, AbstractStrideMatrix,
    StrideArray, StrideVector, StrideMatrix,
    FixedSizeArray, FixedSizeVector, FixedSizeMatrix,
    PtrArray, PtrVector, PtrMatrix,
    # ConstantArray, ConstantVector, ConstantMatrix,
    LazyMap, muladd!, mul!, *ˡ


include("type_declarations.jl")
include("size_and_strides.jl")
include("adjoints.jl")
include("indexing.jl")
include("initialization.jl")





# include("padded_array.jl")
# include("mutable_fs_padded_array.jl")
include("blas.jl")
include("linear_algebra.jl")
include("rand.jl")
include("utilities.jl")
include("broadcast.jl")
include("getindex.jl")
include("lazy_maps.jl")
include("gradsupport.jl")
include("zeroinitialized.jl")


# include("precompile.jl")
# _precompile_()

function __init__()
    set_zero_subnormals(true)
    # @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" @eval using PaddedMatricesForwardDiff
end




end # module

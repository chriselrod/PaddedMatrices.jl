module PaddedMatrices

using VectorizationBase, Base.Cartesian

abstract type AbstractPaddedArray{T,N} <: AbstractArray{T,N}


Base.IndexStyle(::AbstractPaddedArray) = IndexCartesian()
@noinline ThrowBoundsError() = throw(BoundsError())


end # module

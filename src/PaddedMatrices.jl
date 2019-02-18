module PaddedMatrices

using VectorizationBase, Base.Cartesian

abstract type AbstractPaddedArray{T,N} <: AbstractArray{T,N}
abstract type AbstractStaticPaddedArray{S,T,N,P,L} <: AbstractPaddedArray{T,N}

const AbstractPaddedVector{T} = AbstractPaddedArray{T,1}
const AbstractPaddedMatrix{T} = AbstractPaddedArray{T,2}
const AbstractStaticPaddedVector{S,T,P,L} = AbstractStaticPaddedArray{Tuple{S},T,1,P,L}
const AbstractStaticPaddedMatrix{M,N,T,P,L} = AbstractStaticPaddedArray{Tuple{M,N},T,2,P,L}

Base.IndexStyle(::AbstractPaddedArray) = IndexCartesian()
@noinline ThrowBoundsError() = throw(BoundsError())


end # module

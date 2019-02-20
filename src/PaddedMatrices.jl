module PaddedMatrices

using VectorizationBase, SIMDPirates, Base.Cartesian, UnsafeArrays

abstract type AbstractPaddedArray{T,N} <: AbstractArray{T,N} end
abstract type AbstractFixedSizePaddedArray{S,T,N,P,L} <: AbstractPaddedArray{T,N} end
abstract type AbstractMutableFixedSizePaddedArray{S,T,N,P,L} <: AbstractFixedSizePaddedArray{S,T,N,P,L} end
abstract type AbstractConstantFixedSizePaddedArray{S,T,N,P,L} <: AbstractFixedSizePaddedArray{S,T,N,P,L} end

const AbstractPaddedVector{T} = AbstractPaddedArray{T,1}
const AbstractPaddedMatrix{T} = AbstractPaddedArray{T,2}
const AbstractFixedSizePaddedVector{M,T,P,L} = AbstractFixedSizePaddedArray{Tuple{M},T,1,P,L}
const AbstractFixedSizePaddedMatrix{M,N,T,P,L} = AbstractFixedSizePaddedArray{Tuple{M,N},T,2,P,L}
const AbstractMutableFixedSizePaddedVector{M,T,P,L} = AbstractMutableFixedSizePaddedArray{Tuple{M},T,1,P,L}
const AbstractMutableFixedSizePaddedMatrix{M,N,T,P,L} = AbstractMutableFixedSizePaddedArray{Tuple{M,N},T,2,P,L}
const AbstractConstantFixedSizePaddedVector{M,T,P,L} = AbstractConstantFixedSizePaddedArray{Tuple{M},T,1,P,L}
const AbstractConstantFixedSizePaddedMatrix{M,N,T,P,L} = AbstractConstantFixedSizePaddedArray{Tuple{M,N},T,2,P,L}

Base.IndexStyle(::AbstractPaddedArray) = IndexCartesian()
@noinline ThrowBoundsError() = throw(BoundsError())
@noinline ThrowBoundsError(str) = throw(BoundsError(str))

@inline full_length(::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L} = L
@inline full_length(::Type{<: AbstractFixedSizePaddedArray{S,T,N,P,L}}) where {S,T,N,P,L} = L
@inline full_length(A::AbstractPaddedArray) = length(A.data)
@inline full_length(::NTuple{N}) where {N} = N
@inline full_length(::Type{<:NTuple{N}}) where {N} = N

@inline is_sized(::NTuple) = true
@inline is_sized(::Type{<:NTuple}) = true
@inline is_sized(::Number) = true
@inline is_sized(::Type{<:Number}) = true
@inline type_length(::NTuple{N}) where {N} = N
@inline type_length(::Type{<:NTuple{N}}) where {N} = N
@inline type_length(::Number) = 1
@inline type_length(::Type{<:Number}) = 1
@inline is_sized(::AbstractStaticPaddedVector) = true
@inline is_sized(::Type{<:AbstractStaticPaddedVector}) = true
@inline is_sized(::AbstractStaticPaddedMatrix) = true
@inline is_sized(::Type{<:AbstractStaticPaddedMatrix}) = true
@inline type_length(::AbstractStaticPaddedVector{N}) where {N} = N
@inline type_length(::Type{<:AbstractStaticPaddedVector{N}}) where {N} = N
@inline type_length(::AbstractStaticPaddedMatrix{M,N}) where {M,N} = M*N
@inline type_length(::Type{<:AbstractStaticPaddedMatrix{M,N}}) where {M,N} = M*N
@inline is_sized(::Any) = false



end # module

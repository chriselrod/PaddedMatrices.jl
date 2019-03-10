module PaddedMatrices

using VectorizationBase, SIMDPirates,
        Base.Cartesian, UnsafeArrays,
        SLEEFPirates, VectorizedRNG,
        LoopVectorization, LinearAlgebra, Random

export @CFixedSize, @MFixedSize,
    ConstantFixedSizePaddedArray,
    ConstantFixedSizePaddedVector,
    ConstantFixedSizePaddedMatrix,
    MutableFixedSizePaddedArray,
    MutableFixedSizePaddedVector,
    MutableFixedSizePaddedMatrix


struct Static{N} end
Base.@pure Static(N) = Static{N}()

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

@inline LoopVectorization.stride_row(::AbstractFixedSizePaddedMatrix{M,N,T,P}) where {M,N,T,P} = P

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
# @inline is_sized(::AbstractFixedSizePaddedVector) = true
# @inline is_sized(::Type{<:AbstractFixedSizePaddedVector}) = true
@inline is_sized(::AbstractFixedSizePaddedArray) = true
@inline is_sized(::Type{<:AbstractFixedSizePaddedArray}) = true
# @inline type_length(::AbstractFixedSizePaddedVector{N}) where {N} = N
# @inline type_length(::Type{<:AbstractFixedSizePaddedVector{N}}) where {N} = N
# @inline type_length(::AbstractFixedSizePaddedMatrix{M,N}) where {M,N} = M*N
# @inline type_length(::Type{<:AbstractFixedSizePaddedMatrix{M,N}}) where {M,N} = M*N
@generated function type_length(::AbstractFixedSizePaddedArray{S}) where {S}
    SV = S.parameters
    L = 1
    for n ∈ 1:length(SV)
        L *= SV[n]
    end
    L
end
@generated function type_length(::Type{<:AbstractFixedSizePaddedArray{S}}) where {S}
    SV = S.parameters
    L = 1
    for n ∈ 1:length(SV)
        L *= SV[n]
    end
    L
end
@inline is_sized(::Any) = false

"""
Returns one based index.
"""
function sub2ind_expr(S, P)
    N = length(S)
    N == 1 && return :(i[1])
    ex = :(i[$N] - 1)
    for i ∈ (N - 1):-1:2
        ex = :(i[$i] - 1 + $(S[i]) * $ex)
    end
    :(i[1] + $P * $ex)
end

include("padded_array.jl")
include("mutable_fs_padded_array.jl")
include("const_fs_padded_array.jl")
include("kernels.jl")
include("blas.jl")
include("linear_algebra.jl")
include("rand.jl")
include("utilities.jl")

end # module

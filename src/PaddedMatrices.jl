module PaddedMatrices

using VectorizationBase, SIMDPirates,
        Base.Cartesian, UnsafeArrays,
        SLEEFPirates, VectorizedRNG,
        LoopVectorization, LinearAlgebra, Random

using MacroTools: @capture, prettify, postwalk
using LoopVectorization: @vvectorize

export @Constant, @Mutable,
    ConstantFixedSizePaddedArray,
    ConstantFixedSizePaddedVector,
    ConstantFixedSizePaddedMatrix,
    MutableFixedSizePaddedArray,
    MutableFixedSizePaddedVector,
    MutableFixedSizePaddedMatrix,
    DynamicPaddedVector,
    DynamicPaddedMatrix,
    DynamicPaddedArray,
    PtrVector, PtrMatrix, PtrArray


@noinline ThrowBoundsError() = throw(BoundsError())
@noinline ThrowBoundsError(str) = throw(BoundsError(str))

struct Static{N} end
Base.@pure Static(N) = Static{N}()
static_type(::Static{N}) where {N} = N
static_type(::Type{Static{N}}) where {N} = N
(::Base.Colon)(i::Int64,::Static{N}) where {N} = i:N
tonumber(::Static{N}) where {N} = N
@inline function Base.getindex(::Static{N}, i) where {N}
    @boundscheck i > N && ThrowBoundsError()
    i
end

abstract type AbstractPaddedArray{T,N} <: AbstractArray{T,N} end
abstract type AbstractFixedSizePaddedArray{S,T,N,P,L} <: AbstractPaddedArray{T,N} end
abstract type AbstractMutableFixedSizePaddedArray{S,T,N,P,L} <: AbstractFixedSizePaddedArray{S,T,N,P,L} end
abstract type AbstractConstantFixedSizePaddedArray{S,T,N,P,L} <: AbstractFixedSizePaddedArray{S,T,N,P,L} end

const AbstractPaddedArrayOrAdjoint{T,N} = Union{AbstractPaddedArray{T,N},<:Adjoint{T,<:AbstractPaddedArray{T,N}}}

const AbstractPaddedVector{T} = AbstractPaddedArray{T,1}
const AbstractPaddedMatrix{T} = AbstractPaddedArray{T,2}
const AbstractFixedSizePaddedVector{M,T,P,L} = AbstractFixedSizePaddedArray{Tuple{M},T,1,P,L}
const AbstractFixedSizePaddedMatrix{M,N,T,P,L} = AbstractFixedSizePaddedArray{Tuple{M,N},T,2,P,L}
const AbstractMutableFixedSizePaddedVector{M,T,P,L} = AbstractMutableFixedSizePaddedArray{Tuple{M},T,1,P,L}
const AbstractMutableFixedSizePaddedMatrix{M,N,T,P,L} = AbstractMutableFixedSizePaddedArray{Tuple{M,N},T,2,P,L}
const AbstractConstantFixedSizePaddedVector{M,T,P,L} = AbstractConstantFixedSizePaddedArray{Tuple{M},T,1,P,L}
const AbstractConstantFixedSizePaddedMatrix{M,N,T,P,L} = AbstractConstantFixedSizePaddedArray{Tuple{M,N},T,2,P,L}

struct StaticUnitRange{L,S} <: AbstractFixedSizePaddedVector{L,Int,L,L} end
Base.getindex(::StaticUnitRange{L,S}, i::Integer) where {L,S} = Int(i+S-1)
Base.size(::StaticUnitRange{L}) where {L} = (L,)
Base.length(::StaticUnitRange{L}) where {L} = L
Base.IndexStyle(::Type{<:StaticUnitRange}) = Base.IndexLinear()
@generated StaticUnitRange(::Val{Start}, ::Val{Stop}) where {Start,Stop} = StaticUnitRange{Stop-Start+1,Start}()
macro StaticRange(rq)
    @assert rq.head == :call
    @assert rq.args[1] == :(:)
    :(StaticUnitRange(Val{$(rq.args[2])}(), Val{$(rq.args[3])}()))
end
LinearAlgebra.checksquare(::AbstractFixedSizePaddedMatrix{M,M}) where {M} = M
LinearAlgebra.checksquare(::AbstractFixedSizePaddedMatrix) = DimensionMismatch("Matrix is not square.")
                                
@inline LoopVectorization.stride_row(::LinearAlgebra.Adjoint{T,V}) where {T,V <: AbstractVector{T}} = 1
@inline LoopVectorization.stride_row(::AbstractFixedSizePaddedArray{S,T,N,P}) where {S,T,N,P} = P

Base.IndexStyle(::Type{<:AbstractPaddedArray}) = IndexCartesian()
Base.IndexStyle(::Type{<:AbstractPaddedVector}) = IndexLinear()
@generated function Base.IndexStyle(::Type{<:AbstractFixedSizePaddedArray{S,T,N,P}}) where {S,T,N,P}
    # If it is a vector, of if the array doesn't have padding, then it is IndexLinear().
    N == 1 && return IndexLinear()
    S.parameters[1] == P && return IndexLinear()
    IndexCartesian()
end


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
@inline param_type_length(::NTuple{N}) where {N} = N
@inline param_type_length(::Type{<:NTuple{N}}) where {N} = N
@inline type_length(::Number) = 1
@inline type_length(::Type{<:Number}) = 1
@inline param_type_length(::Number) = 1
@inline param_type_length(::Type{<:Number}) = 1
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
@generated param_type_length(::AbstractFixedSizePaddedArray{S}) where {S} = prod(SV.parameters)
@generated param_type_length(::Type{<:AbstractFixedSizePaddedArray{S}}) where {S} = prod(S.parameters)
@inline is_sized(::Any) = false

"""
Converts Cartesian one-based index into linear one-based index.
Just subtract 1 for a zero based index.
"""
function sub2ind_expr(S, P,  N = length(S))
    N == 1 && return :(@inbounds i[1])
    ex = :(i[$N] - 1)
    for i ∈ (N - 1):-1:2
        ex = :(i[$i] - 1 + $(S[i]) * $ex)
    end
    :(@inbounds i[1] + $P * $ex)
end
@generated function sub2ind(s::NTuple{N}, i::NTuple{N}, P = size(s,1)) where {N}
    N == 1 && return :(@inbounds i[1])
    ex = :(i[$N] - 1)
    for j ∈ (N-1):-1:2
        ex = :(i[$j] - 1 + s[$j] * $ex)
    end
    :(@inbounds i[1] + P * $ex)
end

## WHAT IS THIS FOR?
## ADD DOCS FOR STUBS
function vload! end

function __init__()
    set_zero_subnormals(true)
end

include("stack_pointer.jl")
include("padded_array.jl")
include("mutable_fs_padded_array.jl")
include("const_fs_padded_array.jl")
include("kernels.jl")
include("blas.jl")
include("array_of_vecs_funcs.jl")
include("linear_algebra.jl")
include("rand.jl")
include("utilities.jl")
include("seed_increments.jl")
include("broadcast.jl")

end # module

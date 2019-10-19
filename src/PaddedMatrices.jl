


module PaddedMatrices

using VectorizationBase, SIMDPirates,
    SLEEFPirates, VectorizedRNG,
    LoopVectorization, LinearAlgebra,
    Random, MacroTools, StackPointers


import ReverseDiffExpressionsBase:
    RESERVED_INCREMENT_SEED_RESERVED,
    RESERVED_MULTIPLY_SEED_RESERVED,
    ∂getindex

using Parameters: @unpack
using MacroTools: @capture, prettify, postwalk

export @Constant, @Mutable,
    ConstantFixedSizeArray,
    ConstantFixedSizeVector,
    ConstantFixedSizeMatrix,
    FixedSizeArray,
    FixedSizeVector,
    FixedSizeMatrix,
    DynamicPaddedVector,
    DynamicPaddedMatrix,
    DynamicPaddedArray,
    PtrVector, DynamicPtrVector,
    PtrMatrix, DynamicPtrMatrix,
    PtrArray, DynamicPtrArray


    


@noinline ThrowBoundsError() = throw(BoundsError())
@noinline ThrowBoundsError(str) = throw(BoundsError(str))

# Like MacroTools.prettify, but it doesn't alias gensyms. This is because doing so before combining expressions could cause problems.
simplify_expr(ex; lines = false, macro_module::Union{Nothing,Module} = nothing) =
  ex |> (macro_module isa Module ? x -> macroexpand(macro_module,x) : identity) |> (lines ? identity : MacroTools.striplines) |> MacroTools.flatten |> MacroTools.unresolve |> MacroTools.resyntax

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

abstract type AbstractPaddedArray{T,N} <: DenseArray{T,N} end # AbstractArray{T,N} end
abstract type AbstractFixedSizeArray{S<:Tuple,T,N,X<:Tuple,L} <: AbstractPaddedArray{T,N} end
abstract type AbstractMutableFixedSizeArray{S,T,N,X,L} <: AbstractFixedSizeArray{S,T,N,X,L} end
abstract type AbstractConstantFixedSizeArray{S,T,N,X,L} <: AbstractFixedSizeArray{S,T,N,X,L} end

const AbstractPaddedArrayOrAdjoint{T,N} = Union{AbstractPaddedArray{T,N},<:Adjoint{T,<:AbstractPaddedArray{T,N}}}

const AbstractPaddedVector{T} = AbstractPaddedArray{T,1}
const AbstractPaddedMatrix{T} = AbstractPaddedArray{T,2}
const AbstractFixedSizeVector{M,T,L} = AbstractFixedSizeArray{Tuple{M},T,1,Tuple{1},L}
const AbstractFixedSizeMatrix{M,N,T,P,L} = AbstractFixedSizeArray{Tuple{M,N},T,2,Tuple{1,P},L}
const AbstractMutableFixedSizeVector{M,T,L} = AbstractMutableFixedSizeArray{Tuple{M},T,1,Tuple{1},L}
const AbstractMutableFixedSizeMatrix{M,N,T,P,L} = AbstractMutableFixedSizeArray{Tuple{M,N},T,2,Tuple{1,P},L}
const AbstractConstantFixedSizeVector{M,T,L} = AbstractConstantFixedSizeArray{Tuple{M},T,1,Tuple{1},L}
const AbstractConstantFixedSizeMatrix{M,N,T,P,L} = AbstractConstantFixedSizeArray{Tuple{M,N},T,2,Tuple{1,P},L}

maybe_static_size(::AbstractFixedSizeArray{S}) where {S} = Static{S}()
maybe_static_size(A::AbstractArray) = size(A)

LinearAlgebra.checksquare(::AbstractFixedSizeMatrix{M,M}) where {M} = M
LinearAlgebra.checksquare(::AbstractFixedSizeMatrix) = DimensionMismatch("Matrix is not square.")
                                
@inline LoopVectorization.stride_row(::LinearAlgebra.Adjoint{T,V}) where {T,V <: AbstractVector{T}} = 1
@generated function LoopVectorization.stride_row(::AbstractFixedSizeArray{S,T,N,P}) where {S,T,N,P}
    quote
        $(Expr(:meta, :inline))
        $((P.parameters[2])::Int)
    end
end

Base.IndexStyle(::Type{<:AbstractPaddedArray}) = IndexCartesian()
Base.IndexStyle(::Type{<:AbstractPaddedVector}) = IndexLinear()
@generated function Base.IndexStyle(::Type{<:AbstractFixedSizeArray{S,T,N,P}}) where {S,T,N,P}
    # If it is a vector, of if the array doesn't have padding, then it is IndexLinear().
    N == 1 && return IndexLinear()
    (S.parameters[1])::Int == (P.parameters[2])::Int && return IndexLinear()
    IndexCartesian()
end


@inline full_length(::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L} = L
@inline full_length(::Type{<: AbstractFixedSizeArray{S,T,N,X,L}}) where {S,T,N,X,L} = L
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
# @inline is_sized(::AbstractFixedSizeVector) = true
# @inline is_sized(::Type{<:AbstractFixedSizeVector}) = true
@inline is_sized(::AbstractFixedSizeArray) = true
@inline is_sized(::Type{<:AbstractFixedSizeArray}) = true
# @inline type_length(::AbstractFixedSizeVector{N}) where {N} = N
# @inline type_length(::Type{<:AbstractFixedSizeVector{N}}) where {N} = N
# @inline type_length(::AbstractFixedSizeMatrix{M,N}) where {M,N} = M*N
# @inline type_length(::Type{<:AbstractFixedSizeMatrix{M,N}}) where {M,N} = M*N
@noinline function simple_vec_prod(sv::Core.SimpleVector)
    p = 1
    for n in 1:length(sv)
        p *= (sv[n])::Int
    end
    p
end
@generated type_length(::AbstractFixedSizeArray{S}) where {S} = simple_vec_prod(S.parameters)
@generated type_length(::Type{<:AbstractFixedSizeArray{S}}) where {S} = simple_vec_prod(S.parameters)
@generated param_type_length(::AbstractFixedSizeArray{S}) where {S} = simple_vec_prod(S.parameters)
@generated param_type_length(::Type{<:AbstractFixedSizeArray{S}}) where {S} = simple_vec_prod(S.parameters)
@inline is_sized(::Any) = false

"""
Converts Cartesian one-based index into linear one-based index.
Just subtract 1 for a zero based index.
"""
@noinline function sub2ind_expr(X::Core.SimpleVector)#, N::Int = length(X)) 
    x1 = first(X)::Int; N = length(X)
    if N == 1
        return x1 == 1 ? :(@inbounds i[1] - 1) : :(@inbounds (i[1] - 1) * $x1 )
    end
    :( @inbounds +($(x1 == 1 ? :(i[1] - 1) : :((i[1] - 1)*$x1)),  $([:((i[$n]-1)*$((X[n])::Int)) for n in 2:N]...)) )
end
@generated function sub2ind(
    s::NTuple{N},
    i::Union{<: NTuple{N}, CartesianIndex{N}},
    P = size(s,1)
) where {N}
    N == 1 && return :(@inbounds i[1])
    ex = :(i[$N] - 1)
    for j ∈ (N-1):-1:2
        ex = :(i[$j] - 1 + s[$j] * $ex)
    end
    :(@inbounds i[1] + P * $ex)
end
@generated function sub2ind(::AbstractFixedSizeArray{S,T,N,X}, i) where {S,T,N,X}
    sub2ind_expr(X.parameters)
end

function evaluate_integers(expr)
    p::Int = 0
    MacroTools.postwalk(expr) do x
        if @capture(x, +(args__))
            p = 0
            args2 = Any[]
            for a ∈ args
                if a isa Integer
                    p += a
                else
                    push!(args2, a)
                end
            end
            return length(args2) == 0 ? p : :(+($p, $(args2...)))
        elseif @capture(x, *(args__))
            p = 1
            args2 = Any[]
            for a ∈ args
                if a isa Integer
                    p *= a
                else
                    push!(args2, a)
                end
            end
            return length(args2) == 0 ? p : :(*($p, $(args2...)))
        else
            return x
        end
    end
end

## WHAT IS THIS FOR?
## ADD DOCS FOR STUBS
function vload! end

function calc_padding(nrow::Int, T)
    W = VectorizationBase.pick_vector_width(T)
    W > nrow ? VectorizationBase.nextpow2(nrow) : VectorizationBase.align(nrow, T)
end
@noinline function calc_strides(SV::Core.SimpleVector, T)
    P = calc_padding(first(SV)::Int, T)::Int
    X = [ 1 ]
    for n in 2:length(SV)-1
        push!(X, P)
        P *= (SV[n])::Int
    end
    push!(X, P)
    X
end
# calc_padding(nrow::Int, T) = VectorizationBase.ispow2(nrow) ? nrow : VectorizationBase.align(nrow, T)

# include("stack_pointer.jl")
include("padded_array.jl")
include("mutable_fs_padded_array.jl")
include("const_fs_padded_array.jl")
include("kernels.jl")
include("blas.jl")
include("array_of_vecs_funcs.jl")
include("linear_algebra.jl")
include("rand.jl")
include("utilities.jl")
include("broadcast.jl")
include("getindex.jl")

@def_stackpointer_fallback vexp ∂materialize DynamicPtrVector DynamicPtrMatrix DynamicPtrArray 
function __init__()
    @add_stackpointer_method vexp ∂materialize DynamicPtrVector DynamicPtrMatrix DynamicPtrArray 
    set_zero_subnormals(true)
    # @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" @eval using PaddedMatricesForwardDiff
end


end # module

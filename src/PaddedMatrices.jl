


module PaddedMatrices

using VectorizationBase, SIMDPirates,
    SLEEFPirates, VectorizedRNG,
    LoopVectorization, LinearAlgebra,
    Random, StackPointers,
    SpecialFunctions # Perhaps there is a better way to support erf?

using VectorizationBase: Static, StaticUnitRange, align, gep
import ReverseDiffExpressionsBase:
    RESERVED_INCREMENT_SEED_RESERVED!, ∂getindex,
    alloc_adjoint, uninitialized, initialized, isinitialized
import LoopVectorization: isdense

using Parameters: @unpack

export @Mutable, # @Constant,
    AbstractFixedSizeArray,
    AbstractFixedSizeVector,
    AbstractFixedSizeMatrix,
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
    PtrArray, DynamicPtrArray,
    LazyMap, Static, StaticOneTo, muladd!, mul!, *ˡ


@noinline ThrowBoundsError() = throw("Bounds Error")
@noinline ThrowBoundsError(str) = throw(str)
@noinline ThrowBoundsError(A, i) = throw(BoundsError(A, i))

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

const AbstractTransposedFixedSizeMatrix{M,N,T} = Union{LinearAlgebra.Transpose{T,<:AbstractFixedSizeArray{Tuple{N,M},T}},LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{Tuple{N,M},T}}}
const AbstractFixedMatrix{M,N,T} = Union{AbstractTransposedFixedSizeMatrix{M,N,T},AbstractFixedSizeArray{Tuple{M,N},T}}


LoopVectorization.maybestaticlength(::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L} = Static{L}()
LoopVectorization.maybestaticsize(::AbstractFixedSizeArray{<:Tuple{M,Vararg}}, ::Val{1}) where {M} = Static{M}()
LoopVectorization.maybestaticsize(::AbstractFixedSizeArray{<:Tuple{M,N,Vararg}}, ::Val{2}) where {M,N} = Static{N}()
LoopVectorization.maybestaticsize(::AbstractFixedSizeArray{<:Tuple{M,N,K,Vararg}}, ::Val{3}) where {M,N,K} = Static{K}()
@generated LoopVectorization.maybestaticsize(::AbstractFixedSizeArray{S}, ::Val{I}) where {S,I} = Static{S.parameters[I]}()

LinearAlgebra.checksquare(::AbstractFixedSizeMatrix{M,M}) where {M} = M
LinearAlgebra.checksquare(::AbstractFixedSizeMatrix) = DimensionMismatch("Matrix is not square.")
                                
Base.IndexStyle(::Type{<:AbstractPaddedArray}) = IndexCartesian()
Base.IndexStyle(::Type{<:AbstractPaddedVector}) = IndexLinear()
@generated function Base.IndexStyle(::Type{<:AbstractFixedSizeArray{S,T,N,X}}) where {S,T,N,X}
    x = 1
    for n ∈ 1:N
        x == (X.parameters[n])::Int || return IndexCartesian()
        x *= (S.parameters[n])::Int
    end
    IndexLinear()
end


# FIXME: Need to clean up this mess
@inline val_length(::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L} = Val{L}()
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
@inline type_length(x::AbstractArray) = length(x)
@inline param_type_length(::Number) = 1
@inline param_type_length(::Type{<:Number}) = 1

@generated function type_length(nt::NT) where {NT <: NamedTuple}
    P = first(NT.parameters)
    q = quote s = 0 end
    for p ∈ P
        push!(q.args, :(s += type_length(nt.$p)))
    end
    q
end


@inline is_sized(::AbstractFixedSizeArray) = true
@inline is_sized(::Type{<:AbstractFixedSizeArray}) = true
@noinline function simple_vec_prod(sv::Core.SimpleVector)
    p = 1
    for n ∈ 1:length(sv)
        p *= (sv[n])::Int
    end
    p
end
@noinline function isdense(S::Core.SimpleVector, X::Core.SimpleVector)
    N = length(S)
    # N == 1 && return true # shortcut not valid in general, as non-unit stride is possible
    p = 1
    for n ∈ 1:N-1
        p *= (S[n])::Int
    end
    p == (last(X)::Int)
end
@generated function isdense(::Type{<:AbstractFixedSizeArray{S,T,N,X}}) where {S,T,N,X}
    isdense(S.parameters, X.parameters)
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
function sum_inds(args)
    p = 0
    ex = Expr(:call, :+)
    for a ∈ args
        if a isa Integer
            p += a
        else
            push!(ex.args, a)
        end
    end
    length(ex.args) == 1 && return p
    push!(ex.args, p)
    ex
end
function mul_inds(args)
    p = 1
    ex = Expr(:call, :*)
    for a ∈ args
        if a isa Integer
            p *= a
        else
            push!(ex.args, a)
        end
    end
    length(ex.args) == 1 && return p
    push!(ex.args, p)
    ex
end

## WHAT IS THIS FOR?
## ADD DOCS FOR STUBS
# function vload! end

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
# include("const_fs_padded_array.jl")
# include("kernels.jl")
include("blas.jl")
# include("elementwise.jl")
# include("array_of_vecs_funcs.jl")
include("linear_algebra.jl")
include("rand.jl")
include("utilities.jl")
include("broadcast.jl")
include("getindex.jl")
include("lazy_maps.jl")
include("gradsupport.jl")
include("zeroinitialized.jl")

function pointer_array_type(::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    PtrArray{S,T,N,X,L,false}
end
macro temporary_similar(A)
    :(pointer_array_type($A)(SIMDPirates.alloca(val_length($A), eltype($A))))
end

@def_stackpointer_fallback vexp ∂materialize DynamicPtrVector DynamicPtrMatrix DynamicPtrArray 

# include("precompile.jl")
# _precompile_()

function __init__()
    @add_stackpointer_method vexp ∂materialize DynamicPtrVector DynamicPtrMatrix DynamicPtrArray 
    set_zero_subnormals(true)
    # @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" @eval using PaddedMatricesForwardDiff
end




end # module




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

abstract type AbstractStrideArray{S,T,N,X,SN,XN,V,L} <: DenseArray{T,N} end

struct StrideArray{S,T,N,X,SN,XN,L} <: AbstractStrideArray{S,T,N,X,SN,XN,false,L}
    data::Vector{T}
    size::NTuple{SN,Int32}
    stride::NTuple{XN,Int32}
end
mutable struct FixedSizeArray{S,T,N,X,L} <: AbstractStrideArray{S,T,N,X,0,0,false,L}
    data::NTuple{L,Core.VecElement{T}}
end
struct ConstantArray{S,T,N,X,L} <: AbstractStrideArray{S,T,N,X,0,0,false,L}
    data::NTuple{L,Core.VecElement{T}}
end
struct PtrArray{S,T,N,X,SN,XN,V,L} <: AbstractStrideArray{S,T,N,X,SN,XN,V,L}
    data::Ptr{T}
    size::NTuple{SN,Int32}
    stride::NTuple{XN,Int32}
end

const AbstractStrideVector{M,T,X1,SN,XN,V,L} = AbstractStrideArray{Tuple{M},T,1,Tuple{X1},SN,XN,V,L}
const AbstractStrideMatrix{M,N,T,X1,X2,SN,XN,V,L} = AbstractStrideArray{Tuple{M,N},T,2,Tuple{X1,X2},SN,XN,V,L}
const StrideVector{M,T,X1,SN,XN,L} = StrideArray{Tuple{M},T,1,Tuple{X1},SN,XN,L}
const StrideMatrix{M,N,T,X1,X2,SN,XN,L} = StrideArray{Tuple{M,N},T,2,Tuple{X1,X2},SN,XN,L}
const FixedSizeVector{M,T,X1,L} = FixedSizeArray{Tuple{M},T,1,Tuple{X1},L}
const FixedSizeMatrix{M,N,T,X1,X2,L} = FixedSizeArray{Tuple{M,N},T,2,Tuple{X1,X2},L}
const ConstantVector{M,T,X1,L} = ConstantArray{Tuple{M},T,1,Tuple{X1},L}
const ConstantMatrix{M,N,T,X1,X2,L} = ConstantArray{Tuple{M,N},T,2,Tuple{X1,X2},L}
const PtrVector{M,T,X1,SN,XN,V,L} = PtrArray{Tuple{M},T,1,Tuple{X1},SN,XN,V,L}
const PtrMatrix{M,N,T,X1,X2,SN,XN,V,L} = PtrArray{Tuple{M,N},T,2,Tuple{X1,X2},SN,XN,V,L}
const AbstractFixedSizeArray{S,T,N,X,V,L} = AbstractStrideArray{S,T,N,X,0,0,V,L}

@noinline ThrowBoundsError() = throw("Bounds Error")
@noinline ThrowBoundsError(str) = throw(str)
@noinline ThrowBoundsError(A, i) = throw(BoundsError(A, i))

LoopVectorization.maybestaticlength(A::AbstractStrideArray{S,T,N,X,SN,XN,V,-1}) where {S,T,N,X,SN,XN,V} = length(A)
LoopVectorization.maybestaticlength(::AbstractStrideArray{S,T,N,X,SN,XN,V,L}) where {S,T,N,X,SN,XN,V,L} = Static{L}()
LoopVectorization.maybestaticsize(::AbstractStrideArray{<:Tuple{M,Vararg}}, ::Val{1}) where {M} = Static{M}()
LoopVectorization.maybestaticsize(::AbstractStrideArray{<:Tuple{M,N,Vararg}}, ::Val{2}) where {M,N} = Static{N}()
LoopVectorization.maybestaticsize(::AbstractStrideArray{<:Tuple{M,N,K,Vararg}}, ::Val{3}) where {M,N,K} = Static{K}()
LoopVectorization.maybestaticsize(::AbstractStrideArray{<:Tuple{M,N,K,L,Vararg}}, ::Val{4}) where {M,N,K,L} = Static{L}()
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{-1,Vararg}}, ::Val{1}) = @inbounds A.size[1]
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,-1,Vararg}}, ::Val{2}) where {M} = @inbounds A.size[1]
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{-1,-1,Vararg}}, ::Val{2}) where {M} = @inbounds A.size[2]
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,N,-1,Vararg}}, ::Val{3}) where {M,N} = size(A,3)
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,N,K,-1,Vararg}}, ::Val{4}) where {M,N,K} = size(A,4)
@generated function LoopVectorization.maybestaticsize(A::AbstractFixedSizeArray{S}, ::Val{I}) where {S,I}
    M = (S.parameters[I])::Int
    M == -1 ? :(size(A, $I)) : Static{M}()
end


@generated Base.size(A::AbstractFixedSizeArray{S}) where {S} = Expr(:tuple, S.parameters...)
@generated Base.strides(A::AbstractFixedSizeArray{S,T,N,X}) where {S,T,N,X} = Expr(:tuple, X.parameters...)
tup_sv_quote(T) = tup_sv_quote(T.parameters)
function tup_sv_quote(T::Core.SimpleVector, s)
    t = Expr(:tuple)
    N = = length(T)
    i = 0
    for n ∈ 1:N
        Tₙ = (T[n])::Int
        if Tₙ == -1
            i += 0
            push!(t.args, Expr(:ref, Expr(:(.), :A, QuoteNode(s)), i))
        else
            push!(t.args, Tₙ)
        end
    end
    Expr(:block, Expr(:meta, :inline), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), t))
end
@generated Base.size(A::AbstractStrideArray{S}) where {S} = tup_sv_quote(S, :size)
@generated Base.strides(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X} = tup_sv_quote(X, :stride)

LinearAlgebra.checksquare(::AbstractFixedMatrix{M,M}) where {M} = M
LinearAlgebra.checksquare(A::AbstractFixedMatrix{M,-1}) where {M} = ((@assert M == @inbounds A.size[1]); M)
LinearAlgebra.checksquare(A::AbstractFixedMatrix{-1,M}) where {M} = ((@assert M == @inbounds A.size[1]); M)
function LinearAlgebra.checksquare(A::AbstractFixedMatrix{-1,-1})
    M, N = @inbounds A.size[1], A.size[2]
    @assert M == N
    M
end
LinearAlgebra.checksquare(::AbstractFixedMatrix{M,N}) where {M,N} = DimensionMismatch("Matrix is not square: dimensions are ($M,$N).")
                                
Base.IndexStyle(::Type{<:AbstractStrideArray}) = IndexCartesian()
Base.IndexStyle(::Type{<:AbstractStrideVector{<:Any,<:Any,1}}) = IndexLinear()
@generated function Base.IndexStyle(::Type{<:AbstractStrideArray{S,T,N,X}}) where {S,T,N,X}
    x = 1
    for n ∈ 1:N
        Xₙ = (X.parameters[n])::Int
        x == Xₙ || return IndexCartesian()
        Sₙ = (S.parameters[n])::Int
        Sₙ == -1 && return IndexCartesian()
        x *= Sₙ
    end
    IndexLinear()
end


# FIXME: Need to clean up this mess
@inline val_length(::AbstractFixedSizeArray{S,T,N,X,V,L}) where {S,T,N,X,V,L} = Val{L}()

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
@generated is_sized(::AbstractStrideArray{S,T,N,X}) where {S,T,N,X} = !(anyneg1(S.parameters) || anyneg1(X.parameters))
@generated is_sized(::Type{<:AbstractStrideArray{S,T,N,X}}) where {S,T,N,X} = !(anyneg1(S.parameters) || anyneg1(X.parameters))
function simple_vec_prod(sv::Core.SimpleVector)
    p = 1
    for n ∈ 1:length(sv)
        p *= (sv[n])::Int
    end
    p
end
anyneg1(sv::Core.SimpleVector) = any(s -> s == -1, tointvec(sv))
function tointvec(sv::Core.SimpleVector)
    v = Vector{Int}(undef, length(sv))
    for i ∈ eachindex(v)
        v[i] = sv[i]
    end
    v
end

@generated type_length(::AbstractFixedSizeArray{S}) where {S} = simple_vec_prod(S.parameters)
@generated type_length(::Type{<:AbstractFixedSizeArray{S}}) where {S} = simple_vec_prod(S.parameters)
@generated param_type_length(::AbstractFixedSizeArray{S}) where {S} = simple_vec_prod(S.parameters)
@generated param_type_length(::Type{<:AbstractFixedSizeArray{S}}) where {S} = simple_vec_prod(S.parameters)
@inline is_sized(::Any) = false


function inddimprod(X, i, xi, A::Symbol = :A)
    Xᵢ = (X[i])::Int
    if Xᵢ == 1
        ind = :(i[$i] - 1)
    elseif Xᵢ == -1
        ind = :((i[$i] - 1) * $A.size[$xi] )
        xi += 1
    else
        ind = :((i[$i] - 1) * $Xᵢ )
    end
    ind, xi
end
"""
Converts Cartesian one-based index into linear one-based index.
Just subtract 1 for a zero based index.
"""
@noinline function sub2ind_expr(X::Core.SimpleVector, A::Symbol = :A)#, N::Int = length(X))
    N = length(X)
    ind1, x1 = inddimprod(X, 1, 1, A)
    if N == 1
        inds = ind1
    else
        inds = Expr(:call, :+, ind1)
        for n ∈ 2:N
            indi, x1 = inddimprod(X, n, x1, A)
            push!(inds.args, indi)
        end
    end
    Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,@__FILE__), inds)
end
# @generated function sub2ind(
#     s::NTuple{N},
#     i::Union{<: NTuple{N}, CartesianIndex{N}},
#     P = size(s,1)
# ) where {N}
#     N == 1 && return :(@inbounds i[1])
#     ex = :(i[$N] - 1)
#     for j ∈ (N-1):-1:2
#         ex = :(i[$j] - 1 + s[$j] * $ex)
#     end
#     :(@inbounds i[1] + P * $ex)
# end
@generated function sub2ind(A::AbstractStrideArray{S,T,N,X}, i) where {S,T,N,X}
    sub2ind_expr(X.parameters, :A)
end

function calc_padding(nrow::Int, T)
    W = VectorizationBase.pick_vector_width(T)
    W > nrow ? VectorizationBase.nextpow2(nrow) : VectorizationBase.align(nrow, T)
end

@inline rev(t::NTuple{0}) = t
@inline rev(t::NTuple{1}) = t
@inline rev(t::NTuple) = reverse(t)
reverse_tuple_type(t) = reverse_tuple_type(t.parameters)
function reverse_tuple_type(t::Core.SimpleVector)
    tnew = Expr(:curly, :Tuple)
    N = length(t)
    for n ∈ 0:N-1
        push!(tnew.args, t[N - n])
    end
    tnew
end
@generated function Base.adjoint(A::AbstractFixedSizeArray{S,T,N,X,V,L}) where {S,T,N,X,V,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), Expr(:call, :(PtrArray{$S,$T,$N,$X,0,0,$V,$L}), :(pointer(A)), tuple(), tuple()))
end
@generated function Base.transpose(A::AbstractFixedSizeArray{S,T,N,X,V,L}) where {S,T,N,X,V,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), Expr(:call, :(PtrArray{$S,$T,$N,$X,0,0,$V,$L}), :(pointer(A)), tuple(), tuple()))
end
@generated function Base.adjoint(A::AbstractStrideArray{S,T,N,X,SN,XN,V,L}) where {S,T,N,X,SN,XN,V,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(PtrArray{$S,$T,$N,$X,$SN,$XN,$V,$L}(pointer(A), rev(A.size), rev(A.stride))))
end
@generated function Base.transpose(A::AbstractStrideArray{S,T,N,X,SN,XN,V,L}) where {S,T,N,X,SN,XN,V,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(PtrArray{$S,$T,$N,$X,$SN,$XN,$V,$L}(pointer(A), rev(A.size), rev(A.stride))))
end
@generated function Base.adjoint(A::ConstantArray{S,T,N,X,L}) where {S,T,N,X,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(ConstantArray{$S,$T,$N,$X,$L}(A.data, rev(A.size), rev(A.stride))))
end
@generated function Base.transpose(A::ConstantArray{S,T,N,X,L}) where {S,T,N,X,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(ConstantArray{$S,$T,$N,$X,$L}(A.data, rev(A.size), rev(A.stride))))
end
@generated function Base.adjoint(A::StrideArray{S,T,N,X,0,0,L}) where {S,T,N,X,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(StrideArray{$S,$T,$N,$X,0,0,$L}(A.data, rev(A.size), rev(A.stride))))
end
@generated function Base.transpose(A::StrideArray{S,T,N,X,0,0,L}) where {S,T,N,X,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(StrideArray{$S,$T,$N,$X,0,0,$L}(A.data, rev(A.size), rev(A.stride))))
end
@generated function Base.adjoint(A::StrideArray{S,T,N,X,SN,XN,L}) where {S,T,N,X,SN,XN,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(StrideArray{$S,$T,$N,$X,$SN,$XN,$L}(A.data, rev(A.size), rev(A.stride))))
end
@generated function Base.transpose(A::StrideArray{S,T,N,X,SN,XN,L}) where {S,T,N,X,SN,XN,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(StrideArray{$S,$T,$N,$X,$SN,$XN,$L}(A.data, rev(A.size), rev(A.stride))))
end

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

@enum StridedAxisType::Int8 dense=0 padded=1 view=2

"""
Paramters are 
S - Sizes. `NTuple{N,Int}`. Unknowns are `-1`
T - eltype
N - number of axis
X - Strides. `NTuple{N,Int}`. Unknowns are `-1`
C - contiguous axis. None is represented with -1.
B - batch size for contiguous axis. If C != dim with stride 1, it means sets of `B` contiguous elements are grouped together.
O - Order. `NTuple{N,Int}`. Gives relative order of strides, from smallest (`1`) to largest (`N`).
D - Dense. `NTuple{N,StridedAxisType}`. Indicates whether an axis is dense.
SN - Number of unknown sizes.
XN - Number of unknown strides.
"""
abstract type AbstractStrideArray{S,T,N,X,C,B,O,D,SN,XN} <: DenseArray{T,N} end
abstract type AbstractMutableStrideArray{S,T,N,X,C,B,O,D,SN,XN} <: AbstractStrideArray{S,T,N,X,C,B,O,D,SN,XN} end
abstract type AbstractPtrStrideArray{S,T,N,X,C,B,O,D,SN,XN} <: AbstractMutableStrideArray{S,T,N,X,C,B,O,D,SN,XN} end

const AbstractFixedSizeArray{S,T,N,X,C,B,O,D} = AbstractStrideArray{S,T,N,X,C,B,O,D,0,0}
const AbstractStrideVector{S,T,X,C,B,O,D,SN,XN} = AbstractStrideArray{S,T,1,X,C,B,O,D,SN,XN}
const AbstractStrideMatrix{S,T,X,C,B,O,D,SN,XN} = AbstractStrideArray{S,T,2,X,C,B,O,D,SN,XN}
const AbstractMutableStrideMatrix{S,T,X,C,B,O,D,SN,XN} = AbstractMutableStrideArray{S,T,2,X,C,B,O,D,SN,XN}

const AbstractFixedSizeVector{S,T,X,C,B,O,D} = AbstractStrideArray{S,T,1,X,C,B,O,D,0,0}
const AbstractFixedSizeMatrix{S,T,X,C,B,O,D} = AbstractStrideArray{S,T,2,X,C,B,O,D,0,0}
const AbstractMutableFixedSizeMatrix{S,T,X,C,B,O,D} = AbstractMutableStrideArray{S,T,2,X,C,B,O,D,0,0}

default_order_expr(N::Int) = (t = Expr(:tuple); foreach(n -> push!(t.args, n), Base.OneTo(N)); t)
@generated default_order(::Val{N}) where {N} = default_order_expr(N)
all_dense_expr(N::Int) = (t = Expr(:tuple); foreach(_ -> push!(t.args, :dense), Base.OneTo(N)); t)
@generated all_dense(::Val{N}) where {N} = all_dense_expr(N)
function default_org(N::Int)
    O = default_order_expr(N)
    D = all_dense_expr(N)
    1, 1, O, D
end


# const ALIGN_ALL_FS_ARRAYS = true
check_N(::Val{N}, ::NTuple{N,Int}, ::NTuple{N,Int}, ::NTuple{N,Int}, ::NTuple{N,StridedAxisType}) where {N} = nothing
check_N(::Any, ::Any, ::Any, ::Any, ::Any) = throw("Arg mismatch")
struct PtrArray{S,T,N,X,C,B,O,D,SN,XN} <: AbstractPtrStrideArray{S,T,N,X,C,B,O,D,SN,XN}
    ptr::Ptr{T}
    size::NTuple{SN,Int}
    stride::NTuple{XN,Int}
    @inline function PtrArray{S,T,N,X,C,B,O,D}(ptr::Ptr{T}, size::NTuple{SN,Int}, stride::NTuple{XN,Int}) where {S,T,N,X,C,B,O,D,SN,XN}
        check_N(Val{N}(), S, X, O, D)
        new{S,T,N,X,C,B,O,D,SN,XN}(ptr, size, stride)
    end
end
function similar(A::AbstractPtrStrideArray{S,T,N,X,C,B,O,D}, ptr::Ptr{T}) where {S,T,N,X,C,B,O,D}
    PtrArray{S,T,N,X,C,B,O,D}(ptr, size_tuple(A), stride_tuple(A))
end
# @inline function PtrArray{S,T,N,X,C,B,O,D}(ptr::Ptr{T}, size::NTuple{SN,Int}, stride::NTuple{XN,Int}) where {S,T,N,X,C,B,O,D,SN,XN}
#     PtrArray(ptr, size, stride)
# end
struct StrideArray{S,T,N,X,C,B,O,D,SN,XN} <: AbstractMutableStrideArray{S,T,N,X,C,B,O,D,SN,XN}
    ptr::PtrArray{S,T,N,X,C,B,O,D,SN,XN}
    data::Vector{T}
end
@inline function StrideArray{S,T,N,X,C,B,O,D,SN,XN}(ptr, sz, sx, data) where {S,T,N,X,C,B,O,D,SN,XN}
    StrideArray{S,T,N,X,C,B,O,D,SN,XN}(PtrArray{S,T,N,X,C,B,O,D,SN,XN}(ptr, sz, sx), data)
end
struct FixedSizeArray{S,T,N,X,C,B,O,D,SN,XN,L} <: AbstractMutableStrideArray{S,T,N,X,C,B,O,D,SN,XN}
    ptr::PtrArray{S,T,N,X,C,B,O,D,SN,XN}
    data::Base.RefValue{NTuple{L,T}}
end
function default_org(::Val{N}) where {N}
    C = 1; B = 1
    O = default_order(Val{N}())
    D = all_dense(Val{N}())
    C, B, O, D
end
@inline function FixedSizeArray{S,T,N,X,L}(::UndefInitializer) where {S,T,N,X,L}
    r = Ref{NTuple{L,T}}()
    ptr = VectorizationBase.align(Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(r)))
    C, B, O, D = default_org(Val{N}())
    # FixedSizeArray{S,T,N,X,1,1,O,D,0,0,L}(PtrArray{S,T,N,X,1,1,O,D,0,0}(ptr, tuple(), tuple()), r)
    FixedSizeArray(PtrArray{S,T,N,X,C,B,O,D,0,0}(ptr, tuple(), tuple()), r)
end
@inline function FixedSizeArray{S,T,N,X,C,B,O,D,0,0,L}(::UndefInitializer) where {S,T,N,X,C,B,O,D,L}
    r = Ref{NTuple{L,T}}()
    ptr = VectorizationBase.align(Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(r)))
    FixedSizeArray(PtrArray{S,T,N,X,C,B,O,D,0,0}(ptr, tuple(), tuple()), r)
end

@inline function FixedSizeArray{S,T,N,X,C,B,O,D}(ptr::Ptr{T}, sz::NTuple{SN,Int}, st::NTuple{XN,Int}, data::Base.RefValue{NTuple{L,T}}) where {S,T,N,X,L,SN,XN,C,B,O,D}
    FixedSizeArray(PtrArray{S,T,N,X,C,B,O,D,SN,XN}(ptr, sz, st), data)
end
# struct ConstantArray{S,T,N,X,L} <: AbstractStrideArray{S,T,N,X,0,0,false}
#     data::NTuple{L,Core.VecElement{T}}
# end
# const LazyArray{F,S,T,N,X,SN,XN,V,L} = VectorizationBase.LazyMap{F,A,A<:AbstractStrideArray{S,T,N,X,SN,XN,V,L}}
struct LazyMap{F,S,T,N,X,C,B,O,D,SN,XN} <: AbstractStrideArray{S,T,N,X,C,B,O,D,SN,XN}
    f::F
    ptr::PtrArray{S,T,N,X,C,B,O,D,SN,XN}
end

@inline size_tuple(A::PtrArray) = A.size
@inline stride_tuple(A::PtrArray) = A.stride
@inline size_tuple(A::AbstractStrideArray) = A.ptr.size
@inline stride_tuple(A::AbstractStrideArray) = A.ptr.stride


# const AbstractStrideVector{M,T,X1,SN,XN,V} = AbstractStrideArray{Tuple{M},T,1,Tuple{X1},SN,XN,V}
# const AbstractStrideMatrix{M,N,T,X1,X2,SN,XN,V} = AbstractStrideArray{Tuple{M,N},T,2,Tuple{X1,X2},SN,XN,V}
# const StrideVector{M,T,X1,SN,XN,V} = StrideArray{Tuple{M},T,1,Tuple{X1},SN,XN,V}
# const StrideMatrix{M,N,T,X1,X2,SN,XN,V} = StrideArray{Tuple{M,N},T,2,Tuple{X1,X2},SN,XN,V}
# const FixedSizeVector{M,T,X1,L} = FixedSizeArray{Tuple{M},T,1,Tuple{X1},L}
# const FixedSizeMatrix{M,N,T,X1,X2,L} = FixedSizeArray{Tuple{M,N},T,2,Tuple{X1,X2},L}
# const ConstantVector{M,T,X1,L} = ConstantArray{Tuple{M},T,1,Tuple{X1},L}
# const ConstantMatrix{M,N,T,X1,X2,L} = ConstantArray{Tuple{M,N},T,2,Tuple{X1,X2},L}
# const PtrVector{M,T,X1,SN,XN,V} = PtrArray{Tuple{M},T,1,Tuple{X1},SN,XN,V}
# const PtrMatrix{M,N,T,X1,X2,SN,XN,V} = PtrArray{Tuple{M,N},T,2,Tuple{X1,X2},SN,XN,V}
# const AbstractFixedSizeArray{S,T,N,X,V} = AbstractStrideArray{S,T,N,X,0,0,V}
# const AbstractFixedSizeVector{S,T,X,V} = AbstractStrideArray{Tuple{S},T,1,Tuple{X},0,0,V}
# const AbstractFixedSizeMatrix{M,N,T,X1,X2,V} = AbstractStrideArray{Tuple{M,N},T,2,Tuple{X1,X2},0,0,V}
# const AbstractMutableFixedSizeArray{S,T,N,X,V} = AbstractMutableStrideArray{S,T,N,X,0,0,V}
# const AbstractMutableFixedSizeVector{S,T,X,V} = AbstractMutableStrideArray{Tuple{S},T,1,Tuple{X},0,0,V}
# const AbstractMutableFixedSizeMatrix{M,N,T,X1,X2,V} = AbstractMutableStrideArray{Tuple{M,N},T,2,Tuple{X1,X2},0,0,V}


@inline Base.pointer(A::PtrArray) = A.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::PtrArray{S,T}) where {S,T} = A.ptr
@inline Base.pointer(A::StrideArray) = A.ptr.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::StrideArray{S,T}) where {S,T} = A.ptr.ptr
@inline Base.pointer(A::FixedSizeArray) = A.ptr.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::FixedSizeArray{S,T}) where {S,T} = A.ptr.ptr
# @inline Base.pointer(A::FixedSizeArray{S,T,N,X,L,Ptr{T}}) where {S,T,N,X,L} = A.ptr
# @inline Base.unsafe_convert(::Type{Ptr{T}}, A::FixedSizeArray{S,T,N,X,L,Ptr{T}}) where {S,T,N,X,L} = A.ptr
# @inline Base.pointer(A::FixedSizeArray{S,T,N,X,L,Nothing}) where {S,T,N,X,L} = Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(A))
# @inline Base.unsafe_convert(::Type{Ptr{T}}, A::FixedSizeArray{S,T,N,X,L,Nothing}) where {S,T,N,X,L} = Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(A))
@inline Base.pointer(A::LazyMap) = A.ptr.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::LazyMap{S,T}) where {S,T} = pointer(A.ptr)
@inline Base.elsize(::AbstractStrideArray{<:Any,T}) where {T} = sizeof(T)



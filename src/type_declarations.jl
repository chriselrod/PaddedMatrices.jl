abstract type AbstractStrideArray{S,T,N,X,SN,XN,V} <: DenseArray{T,N} end
abstract type AbstractMutableStrideArray{S,T,N,X,SN,XN,V} <: AbstractStrideArray{S,T,N,X,SN,XN,V} end
abstract type AbstractPtrStrideArray{S,T,N,X,SN,XN,V} <: AbstractMutableStrideArray{S,T,N,X,SN,XN,V} end

# const ALIGN_ALL_FS_ARRAYS = true

struct StrideArray{S,T,N,X,SN,XN} <: AbstractMutableStrideArray{S,T,N,X,SN,XN,false}
    data::SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}
    size::NTuple{SN,Int}
    stride::NTuple{XN,Int}
end
mutable struct FixedSizeArray{S,T,N,X,L} <: AbstractMutableStrideArray{S,T,N,X,0,0,false}
    data::NTuple{L,T}
    ptr::Ptr{T}
    @inline function FixedSizeArray{S,T,N,X,L}(::UndefInitializer) where {S,T,N,X,L,P}
        A = new()
        A.ptr = VectorizationBase.align(Base.pointer_from_objref(A))
        A
    end
end
struct ConstantArray{S,T,N,X,L} <: AbstractStrideArray{S,T,N,X,0,0,false}
    data::NTuple{L,Core.VecElement{T}}
end
struct PtrArray{S,T,N,X,SN,XN,V} <: AbstractPtrStrideArray{S,T,N,X,SN,XN,V}
    ptr::Ptr{T}
    size::NTuple{SN,Int}
    stride::NTuple{XN,Int}
end
# const LazyArray{F,S,T,N,X,SN,XN,V,L} = VectorizationBase.LazyMap{F,A,A<:AbstractStrideArray{S,T,N,X,SN,XN,V,L}}
struct LazyMap{F,S,T,N,X,SN,XN,V} <: AbstractStrideArray{S,T,N,X,SN,XN,V}
    f::F
    ptr::Ptr{T}
    size::NTuple{SN,Int}
    stride::NTuple{XN,Int}
end

const AbstractStrideVector{M,T,X1,SN,XN,V} = AbstractStrideArray{Tuple{M},T,1,Tuple{X1},SN,XN,V}
const AbstractStrideMatrix{M,N,T,X1,X2,SN,XN,V} = AbstractStrideArray{Tuple{M,N},T,2,Tuple{X1,X2},SN,XN,V}
const StrideVector{M,T,X1,SN,XN} = StrideArray{Tuple{M},T,1,Tuple{X1},SN,XN}
const StrideMatrix{M,N,T,X1,X2,SN,XN} = StrideArray{Tuple{M,N},T,2,Tuple{X1,X2},SN,XN}
const FixedSizeVector{M,T,X1} = FixedSizeArray{Tuple{M},T,1,Tuple{X1}}
const FixedSizeMatrix{M,N,T,X1,X2} = FixedSizeArray{Tuple{M,N},T,2,Tuple{X1,X2}}
const ConstantVector{M,T,X1} = ConstantArray{Tuple{M},T,1,Tuple{X1}}
const ConstantMatrix{M,N,T,X1,X2} = ConstantArray{Tuple{M,N},T,2,Tuple{X1,X2}}
const PtrVector{M,T,X1,SN,XN,V} = PtrArray{Tuple{M},T,1,Tuple{X1},SN,XN,V}
const PtrMatrix{M,N,T,X1,X2,SN,XN,V} = PtrArray{Tuple{M,N},T,2,Tuple{X1,X2},SN,XN,V}
const AbstractFixedSizeArray{S,T,N,X,V} = AbstractStrideArray{S,T,N,X,0,0,V}
const AbstractFixedSizeVector{S,T,X,V} = AbstractStrideArray{Tuple{S},T,1,Tuple{X},0,0,V}
const AbstractFixedSizeMatrix{M,N,T,X1,X2,V} = AbstractStrideArray{Tuple{M,N},T,2,Tuple{X1,X2},0,0,V}
const AbstractMutableFixedSizeArray{S,T,N,X,V} = AbstractMutableStrideArray{S,T,N,X,0,0,V}
const AbstractMutableFixedSizeVector{S,T,X,V} = AbstractMutableStrideArray{Tuple{S},T,1,Tuple{X},0,0,V}
const AbstractMutableFixedSizeMatrix{M,N,T,X1,X2,V} = AbstractMutableStrideArray{Tuple{M,N},T,2,Tuple{X1,X2},0,0,V}


@inline Base.pointer(A::StrideArray) = pointer(A.data)
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::StrideArray{S,T}) where {S,T} = pointer(A.data)
@inline Base.pointer(A::FixedSizeArray) = A.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::FixedSizeArray{S,T}) where {S,T} = A.ptr
# @inline Base.pointer(A::FixedSizeArray{S,T,N,X,L,Ptr{T}}) where {S,T,N,X,L} = A.ptr
# @inline Base.unsafe_convert(::Type{Ptr{T}}, A::FixedSizeArray{S,T,N,X,L,Ptr{T}}) where {S,T,N,X,L} = A.ptr
# @inline Base.pointer(A::FixedSizeArray{S,T,N,X,L,Nothing}) where {S,T,N,X,L} = Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(A))
# @inline Base.unsafe_convert(::Type{Ptr{T}}, A::FixedSizeArray{S,T,N,X,L,Nothing}) where {S,T,N,X,L} = Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(A))
@inline Base.pointer(A::PtrArray) = A.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::PtrArray{S,T}) where {S,T} = A.ptr
@inline Base.pointer(A::LazyMap) = A.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::LazyMap{S,T}) where {S,T} = A.ptr


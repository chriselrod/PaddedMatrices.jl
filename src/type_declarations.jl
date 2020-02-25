abstract type AbstractStrideArray{S,T,N,X,SN,XN,V,L} <: DenseArray{T,N} end
abstract type AbstractMutableStrideArray{S,T,N,X,SN,XN,V,L} <: AbstractStrideArray{S,T,N,X,SN,XN,V,L} end

struct StrideArray{S,T,N,X,SN,XN,L} <: AbstractMutableStrideArray{S,T,N,X,SN,XN,false,L}
    data::SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}
    size::NTuple{SN,UInt32}
    stride::NTuple{XN,UInt32}
end
mutable struct FixedSizeArray{S,T,N,X,L} <: AbstractMutableStrideArray{S,T,N,X,0,0,false,L}
    data::NTuple{L,Core.VecElement{T}}
    @inline FixedSizeArray{S,T,N,X,L}(::UndefInitializer) where {S,T,N,X,L} = new()
end
struct ConstantArray{S,T,N,X,L} <: AbstractStrideArray{S,T,N,X,0,0,false,L}
    data::NTuple{L,Core.VecElement{T}}
end
struct PtrArray{S,T,N,X,SN,XN,V,L} <: AbstractMutableStrideArray{S,T,N,X,SN,XN,V,L}
    ptr::Ptr{T}
    size::NTuple{SN,UInt32}
    stride::NTuple{XN,UInt32}
end
# const LazyArray{F,S,T,N,X,SN,XN,V,L} = VectorizationBase.LazyMap{F,A,A<:AbstractStrideArray{S,T,N,X,SN,XN,V,L}}
struct LazyMap{F,S,T,N,X,SN,XN,V,L} <: AbstractStrideArray{S,T,N,X,SN,XN,V,L}
    f::F
    ptr::Ptr{T}
    size::NTuple{SN,UInt32}
    stride::NTuple{XN,UInt32}
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
const AbstractFixedSizeVector{S,T,X,V,L} = AbstractStrideArray{Tuple{S},T,1,Tuple{X},0,0,V,L}
const AbstractFixedSizeMatrix{M,N,T,X1,X2,V,L} = AbstractStrideArray{Tuple{M,N},T,2,Tuple{X1,X2},0,0,V,L}
const AbstractMutableFixedSizeArray{S,T,N,X,V,L} = AbstractMutableStrideArray{S,T,N,X,0,0,V,L}
const AbstractMutableFixedSizeVector{S,T,X,V,L} = AbstractMutableStrideArray{Tuple{S},T,1,Tuple{X},0,0,V,L}
const AbstractMutableFixedSizeMatrix{M,N,T,X1,X2,V,L} = AbstractMutableStrideArray{Tuple{M,N},T,2,Tuple{X1,X2},0,0,V,L}


@inline Base.pointer(A::StrideArray) = pointer(A.data)
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::StrideArray{S,T}) where {S,T} = pointer(A.data)
@inline Base.pointer(A::FixedSizeArray{S,T}) where {S,T} = Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(A))
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::FixedSizeArray{S,T}) where {S,T} = Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(A))
@inline Base.pointer(A::PtrArray) = A.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::PtrArray{S,T}) where {S,T} = A.ptr
@inline Base.pointer(A::LazyMap) = A.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::LazyMap{S,T}) where {S,T} = A.ptr


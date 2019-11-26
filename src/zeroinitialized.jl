

struct ZeroInitializedArray{S,T,N,X,L} <: AbstractMutableFixedSizeArray{S,T,N,X,L}
    ptr::Ptr{T}
end
const ZeroInitializedVector{P,T,L} = ZeroInitializedArray{Tuple{P},T,1,Tuple{1},L}
const ZeroInitializedMatrix{M,N,T,P,L} = ZeroInitializedArray{Tuple{M,N},T,2,Tuple{1,P},L}
@inline Base.pointer(A::ZeroInitializedArray) = A.ptr
@inline VectorizationBase.vectorizable(A::ZeroInitializedArray) = VectorizationBase.ZeroInitializedPointer(A.ptr)
@inline ZeroInitializedArray(A::AbstractMutableFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L} = ZeroInitializedArray{S,T,N,X,L}(pointer(A))
@inline VectorizationBase.zeroinitialized(A::AbstractMutableFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L} = ZeroInitializedArray{S,T,N,X,L}(pointer(A))
@inline Base.getindex(A::ZeroInitializedArray{S,T}, i...) where {S,T} = zero(T)


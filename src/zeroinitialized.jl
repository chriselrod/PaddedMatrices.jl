
struct ZeroInitializedArray{S,T,N,X,SN,XN,V,A <: AbstractStrideArray{S,T,N,X,SN,XN,V}} <: AbstractStrideArray{S,T,N,X,SN,XN,V}
    data::A
end

@inline Base.parent(A::ZeroInitializedArray) = A.data
@inline Base.pointer(A::ZeroInitializedArray) = pointer(A.data)

@inline Base.getindex(A::ZeroInitializedArray{S,T}, i...) where {S,T} = zero(T)
@inline Base.setindex!(A::ZeroInitializedArray, v, i...) = setindex!(A.data, v, i...)

@inline initialized(A::ZeroInitializedArray) = A.data
@inline initialized(A) = A

@inline VectorizationBase.stridedpointer(A::ZeroInitializedArray) = ZeroInitializedStridedPointer(stridedpointer(parent(A)))


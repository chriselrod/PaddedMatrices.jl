

@inline function alloc_adjoint(A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    PtrArray{S,T,N,X,L,false}(SIMDPirates.alloca(Val(L), T))
end
# if given a pointer, alloc_adjoint defaults to a view-PtrArray
@inline alloc_adjoint(ptr::Ptr{T}, A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L} = PtrArray{S,T,N,X,L,true}(ptr)
@inline stack_pointer_call(::typeof(alloc_adjoint), sptr::StackPointer, A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L} = PtrArray{S,T,N,X,L,false}(sptr)

struct UninitializedArray{S,T,N,X,V,L} <: AbstractMutableFixedSizeArray{S,T,N,X,V,L}
    ptr::Ptr{T}
end
const UninitializedVector{P,T,V,L} = UninitializedArray{Tuple{P},T,1,Tuple{1},V,L}
const UninitializedMatrix{M,N,T,P,V,L} = UninitializedArray{Tuple{M,N},T,2,Tuple{1,P},V,L}
@inline Base.pointer(A::UninitializedArray) = A.ptr
@inline function uninitialized(A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    # lifetime_start(A)
    UninitializedArray{S,T,N,X,L}(pointer(A))
end
@inline function initialized(A::UninitializedArray{S,T,N,X,L}) where {S,T,N,X,L}
    PtrArray{S,T,N,X,L,true}(pointer(A))
end
@inline uninitialized(A::LinearAlgebra.Adjoint{T,<:AbstractArray{T}}) where {T} = uninitialized(A.parent)'
@inline initialized(A::LinearAlgebra.Adjoint{T,<:AbstractArray{T}}) where {T} = initialized(A.parent)'
isinitialized(::Type{<:UninitializedArray}) = false




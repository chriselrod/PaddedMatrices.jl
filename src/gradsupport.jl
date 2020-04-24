

# @inline function alloc_adjoint(A::AbstractFixedSizeArray{S,T,N,X}) where {S,T,N,X}
    # PtrArray{S,T,N,X,false}(SIMDPirates.alloca(Val(L), T))
# end
# if given a pointer, alloc_adjoint defaults to a view-PtrArray
@inline alloc_adjoint(ptr::Ptr{T}, A::AbstractFixedSizeArray{S,T,N,X}) where {S,T,N,X} = PtrArray{S,T,N,X,true}(ptr)
@inline stack_pointer_call(::typeof(alloc_adjoint), sptr::StackPointer, A::AbstractFixedSizeArray{S,T,N,X}) where {S,T,N,X} = PtrArray{S,T,N,X,false}(sptr)

struct UninitializedArray{S,T,N,X,V} <: AbstractMutableFixedSizeArray{S,T,N,X,V}
    ptr::Ptr{T}
end
const UninitializedVector{P,T,V} = UninitializedArray{Tuple{P},T,1,Tuple{1},V}
const UninitializedMatrix{M,N,T,P,V} = UninitializedArray{Tuple{M,N},T,2,Tuple{1,P},V}
@inline Base.pointer(A::UninitializedArray) = A.ptr
@inline function uninitialized(A::AbstractFixedSizeArray{S,T,N,X}) where {S,T,N,X}
    # lifetime_start(A)
    UninitializedArray{S,T,N,X}(pointer(A))
end
@inline function initialized(A::UninitializedArray{S,T,N,X}) where {S,T,N,X}
    PtrArray{S,T,N,X,0,0,true}(pointer(A))
end
@inline uninitialized(A::LinearAlgebra.Adjoint{T,<:AbstractArray{T}}) where {T} = uninitialized(A.parent)'
@inline initialized(A::LinearAlgebra.Adjoint{T,<:AbstractArray{T}}) where {T} = initialized(A.parent)'
isinitialized(::Type{<:UninitializedArray}) = false




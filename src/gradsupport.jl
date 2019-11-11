

@inline function alloc_adjoint(A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    PtrArray{S,T,N,X,L,false}(SIMDPirates.alloca(Val(L), T))
end
# @inline function radj(A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    # PtrArray{S,T,N,X,L,false}(SIMDPirates.alloca(Val(L),T))
# end
# if given a pointer, alloc_adjoint defaults to a view-PtrArray
@inline alloc_adjoint(ptr::Ptr{T}, A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L} = PtrArray{S,T,N,X,L,true}(ptr)
@inline alloc_adjoint(sptr::StackPointer, A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L} = PtrArray{S,T,N,X,L,false}(sptr)
@inline alloc_adjoint(sp::StackPointer, ::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,X,L}}) where {S,T,N,X,L} = PtrArray{S,T,N,X,L,false}(sp)

# @inline function radj(sptr::StackPointer, A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    # PtrArray{S,T,N,X,L,false}(sptr)
# end
struct UninitializedArray{S,T,N,X,L} <: AbstractMutableFixedSizeArray{S,T,N,X,L}
    ptr::Ptr{T}
end
const UninitializedVector{P,T,L} = UninitializedArray{Tuple{P},T,1,Tuple{1},L}
const UninitializedMatrix{M,N,T,P,L} = UninitializedArray{Tuple{M,N},T,2,Tuple{1,P},L}
@inline Base.pointer(A::UninitializedArray) = A.ptr
@inline function uninitialized(A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    UninitializedArray{S,T,N,X,L}(pointer(A))
end
@inline function initialized(A::UninitializedArray{S,T,N,X,L}) where {S,T,N,X,L}
    PtrArray{S,T,N,X,L,true}(pointer(A))
end
@inline uninitialized(A::LinearAlgebra.Adjoint{T,<:AbstractArray{T}}) where {T} = uninitialized(A.parent)'
@inline initialized(A::LinearAlgebra.Adjoint{T,<:AbstractArray{T}}) where {T} = initialized(A.parent)'
isinitialized(::Type{<:UninitializedArray}) = false


# Implicitly identity Jacobian.
@generated function RESERVED_INCREMENT_SEED_RESERVED!(
    C::AbstractMutableFixedSizeArray{S,T,N,X,L},
    A::AbstractMutableFixedSizeArray{S,T,N,X,L}
) where {S,T,N,X,L}
    quote
        $(Expr(:meta,:inline))
        @vvectorize $T 4 for l ∈ 1:$L
            C[l] += A[l]
        end
        nothing
    end
end
@generated function RESERVED_INCREMENT_SEED_RESERVED!(
    C::UninitializedArray{S,T,N,X,L},
    A::AbstractMutableFixedSizeArray{S,T,N,X,L}
) where {S,T,N,X,L}
    quote
        $(Expr(:meta,:inline))
        @vvectorize $T 4 for l ∈ 1:$L
            C[l] = A[l]
        end
        nothing
    end
end
# A is implicitly a [block] diagonal Jacobian
@generated function RESERVED_INCREMENT_SEED_RESERVED!(
    C::AbstractMutableFixedSizeArray{S,T,N,X,L},
    A::AbstractMutableFixedSizeArray{S,T,N,X,L},
    B::AbstractMutableFixedSizeArray{S,T,N,X,L}
) where {S,T,N,X,L}
    quote
        $(Expr(:meta,:inline))
        @vvectorize $T 4 for l ∈ 1:$L
            C[l] += A[l] * B[l]
        end
        nothing
    end
end
@generated function RESERVED_INCREMENT_SEED_RESERVED!(
    C::UninitializedArray{S,T,N,X,L},
    A::AbstractMutableFixedSizeArray{S,T,N,X,L},
    B::AbstractMutableFixedSizeArray{S,T,N,X,L}
) where {S,T,N,X,L}
    quote
        $(Expr(:meta,:inline))
        @vvectorize $T 4 for l ∈ 1:$L
            C[l] = A[l] * B[l]
        end
        nothing
    end
end

# Dense Jacobian
@inline function RESERVED_INCREMENT_SEED_RESERVED!(
    C::AbstractMutableFixedSizeVector,
    A::AbstractFixedSizeMatrix,
    B::AbstractFixedSizeVector
)
    muladd!(X, A, B)
    nothing
end
@inline function RESERVED_INCREMENT_SEED_RESERVED!(
    C::UninitializedVector,
    A::AbstractFixedSizeMatrix,
    B::AbstractFixedSizeVector
)
    mul!(X, A, B)
    nothing
end




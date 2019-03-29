

# function determine_vectorloads(num_rows, ::Type{T} = Float64) where T
#     W = VectorizationBase.pick_vector_width(num_rows, T)
#     num_vectors = cld(num_rows, W)
#     num_vectors, num_vectors * W
# end

@generated function Base.sum(A::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L}
    quote
        $(Expr(:meta, :inline))
        out = zero($T)
        @vectorize $T for i ∈ 1:$L
            out += A[i]
        end
        out
    end
end
@inline Base.sum(A::LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedArray{S,T}}) where {S,T} = sum(A.parent)
@generated function Base.prod(A::AbstractFixedSizePaddedVector{L,T}) where {L,T}
    quote
        $(Expr(:meta, :inline))
        out = one(T)
        @vectorize $T for i ∈ 1:$L
            out *= A[i]
        end
        out
    end
end

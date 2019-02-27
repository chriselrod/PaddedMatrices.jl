

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

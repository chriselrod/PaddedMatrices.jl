

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

function Base.cumsum!(A::AbstractMutableFixedSizePaddedVector{M}) where {M}
    @inbounds for m ∈ 2:M
        A[m] += A[m-1]
    end
    A
end
Base.cumsum(A::AbstractMutableFixedSizePaddedVector) = cumsum!(copy(A))
Base.cumsum(A::AbstractConstantFixedSizePaddedVector) = ConstantFixedSizePaddedVector(cumsum!(MutableFixedSizePaddedVector(A)))

@generated function Base.maximum(A::AbstractFixedSizePaddedArray{S,T,P,L}) where {S,T,P,L}
    W, Wshift = VectorizationBase.pick_vector_width_shift(L, T)
    quote
        m = SIMDPirates.vbroadcast(Vec{$W, $T}, -Inf)
        @vectorize $T for l ∈ 1:$L
            a = A[l]
            m = SIMDPirates.vmax(a, m)
        end
        SIMDPirates.maximum(m)
    end
end

@generated function Base.maximum(::typeof(abs), A::AbstractFixedSizePaddedArray{S,T,P,L}) where {S,T,P,L}
    W, Wshift = VectorizationBase.pick_vector_width_shift(L, T)
    quote
        m = SIMDPirates.vbroadcast(Vec{$W, $T}, -Inf)
        @vectorize $T for l ∈ 1:$L
            a = SIMDPirates.vabs(A[l])
            m = SIMDPirates.vmax(a, m)
        end
        SIMDPirates.maximum(m)
    end
end

@inline Base.pointer(x::Symmetric{T,MutableFixedSizePaddedMatrix{P,P,T,R,L}}) where {P,T,R,L} = pointer(x.data)


struct AbstractSymmetricMatrix{P,T,L} <: AbstractDiagTriangularMatrix{P,T,L} end

"""
The data layout is:
diagonal elements
sub diagonal lower triangular part

The matrix is also padded.
Given avx2, a 7x7 matrix of Float64 would be stored as:
7 diagonal elements + 1 element padding

6x6 lower triangle...
column 1: 6 elements + 2 padding
column 2: 5 elements + 3 padding
column 3: 4 elements + 0 padding
column 4: 3 elements + 1 padding
column 4: 2 elements + 2 padding
column 4: 1 elements + 3 padding
"""
struct SymmetricMatrixL{P,T,L} <: AbstractSymmetricMatrix{P,T,L}
    data::NTuple{L,T}
end

@inline function Base.getindex(S::SymmetricMatrix{P,T,L}, i, j) where {P,T,L}
    j, i = minmax(j, i)
    @boundscheck i > P && throw(BoundsError())
    @inbounds S.data[lower_triangle_sub2ind(Val{P}(), T, i, j)]
end

struct SymmetricMatrixU{P,T,L} <: AbstractSymmetricMatrix{P,T,L}
    data::NTuple{L,T}
end
@inline function Base.getindex(S::SymmetricMatrixU{P,T,L}, i, j) where {P,T,L}
    i, j = minmax(i, j)
    @boundscheck j > P && throw(BoundsError())
    @inbounds S.data[upper_triangle_sub2ind(Val{P}(), T, i, j)]
end


@generated function lower_cholesky(Σ::SymmetricMatrixL{P,T,L}) where {P,T,L}

end

### Be more clever than this.
@generated function quadform(x::AbstractFixedSizePaddedVector{P,T,R}, Σ::SymmetricMatrixL{P,T,L}) where {P,T,R,L}
    W = pick_vector_width(P, T)
    quote
        # vout = vbroadcast(Vec{$P,$T}, zero($T))
        out = zero($T)
        @vectorize for i ∈ 1:$P
            out += x[i] * Σ[i]
        end
        for 
    end
end

# @generated function Base.:*(A::AbstractFixedSizePaddedMatrix{M,P,T}, Σ::SymmetricMatrix{P,T,L}) where {M,P,T,L}
#     W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
#
#     quote
#
#     end
# end


struct AbstractSymmetricMatrix{P,T,L} <: AbstractMatrix{T} end

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
Base.size(::AbstractSymmetricMatrixL{P}) = (P,P)
@inline function Base.getindex(S::AbstractSymmetricMatrix{P,T,L}, i) where {P,T,L}
    @boundscheck i > L && throw(BoundsError())
    @inbounds S.data[i]
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
    j, i = minmax(j, i)
    @boundscheck i > P && throw(BoundsError())
    @inbounds S.data[upper_triangle_sub2ind(Val{P}(), T, i, j)]
end


@generated function cholesky()

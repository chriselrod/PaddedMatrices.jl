
abstract type AbstractTriangularMatrix{P,T,L} <: AbstractDiagTriangularMatrix{P,T,L} end

struct LowerTriangularMatrix{P,T,L} <: AbstractTriangularMatrix{P,T,L}
    data::NTuple{L,T}
end
struct UpperTriangularMatrix{P,T,L} <: AbstractTriangularMatrix{P,T,L}
    data::NTuple{L,T}
end

@generated function LinearAlgebra.det(A::AbstractTriangularMatrix{P,T,L}) where {P,T,L}
    quote
        out = one(T)
        @vectorize for i ∈ 1:$P
            out *= A[i]
        end
        out
    end
end
"""
logdet(A) will be slower than log(det(A)),
but should be more numerically accurate.
det(A) is at high risk of over/underflow
for large matrices.
"""
@generated function LinearAlgebra.logdet(A::AbstractTriangularMatrix{P,T,L}) where {P,T,L}
    quote
        out = zero(T)
        @vectorize for i ∈ 1:$P
            out += log(A[i])
        end
        out
    end
end

@generated function lower_cholesky(Σ::SymmetricMatrixL{P,T,L}) where {P,T,L}

end

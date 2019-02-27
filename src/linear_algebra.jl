
"""
Fall back definition for scalars.
"""
@inline invchol(x) = SIMDPirates.rsqrt(x)

@generated function invchol(A::Diagonal{T,<:ConstantFixedSizePaddedVector{N,T,P}}) where {N,T,P}
    quote
        $(Expr(:meta,:inline)) # do we really want to force inline this?
        mv = MutableFixedSizePaddedVector{N,T}(undef)
        Adiag = A.diag
        @vectorize $T for i ∈ 1:$P
            mv[i] = rsqrt(Adiag[i])
        end
        Diagonal(ConstantFixedSizePaddedArray(mv))
    end
end

abstract type AbstractDiagTriangularMatrix{P,T,L} <: AbstractMatrix{T} end

Base.size(::AbstractDiagTriangularMatrix{P}) where {P} = (P,P)
@inline function Base.getindex(S::AbstractDiagTriangularMatrix{P,T,L}, i) where {P,T,L}
    @boundscheck i > L && throw(BoundsError())
    @inbounds S.data[i]
end
@inline VectorizationBase.vectorizable(A::AbstractDiagTriangularMatrix) = vStaticPaddedArray(A,0)



@inline binomial2(n) = (n*(n-1)) >> 1

function padded_diagonal_length(P, T)
    W = VectorizationBase.pick_vector_width(P, T)
    rem = P & (W - 1)
    rem == 0 ? P : P - rem + W
end
function num_complete_blocks(P, T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
    (P - 1) >> Wshift
end
function calculate_L(P, T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
    W² = W*W
    Wm1 = W - 1
    rem = P & (W - 1)
    padded_diag_length = rem == 0 ? P : P - rem + W
    L = padded_diag_length

    Pm1 = P - 1
    num_complete_blocks = Pm1 >> Wshift
    L += W² * binomial(1+num_complete_blocks, 2)
    rem_block = Pm1 & Wm1
    L += (1+num_complete_blocks)*W*rem_block
    L
end

using VectorizationBase
function lower_triangle_sub2ind_quote(P, T)
    # assume inputs are i, j
    # for row, column
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
    W² = W * W
    Wm1 = W - 1
    rem = P & Wm1
    if rem == 0
        pad = 0
        diag_length = P
        num_rows = P
        rem_row = Wm1
    else
        pad = W - rem
        diag_length = P + pad
        if rem == 1
            num_rows = P - rem
            rem_row = W
        else
            num_rows = diag_length
            rem_row = rem - 1
        end
        # num_rows = rem == 1 ? P - rem : diag_length
        # rem_row = rem - 1
    end
    # @show diag_length, num_rows, rem_row
    quote
        # We want i >= j, for a lower triangular matrix.
        j, i = minmax(j, i)
        # @boundscheck i > $P && ThrowBoundsError(str)
        i == j && return i
        ind = $(P - num_rows + W - rem_row - 1) + j*$num_rows + i
        if j > $rem_row
            j2 = j - $rem_row
            jrem = j2 & $(W-1)
            number_blocks = 1+(j2 >> $Wshift)
            ind -= jrem * $W * number_blocks + binomial(number_blocks,2) << $(2Wshift)
        end
        ind
    end
end
@generated function lower_triangle_sub2ind(::Val{P}, ::Type{T}, i, j) where {T,P}
    quote
        # $(Expr(:meta, :inline))
        $(lower_triangle_sub2ind_quote(P, T))
    end
end

function upper_triangle_sub2ind_quote(P, T)
    # assume inputs are i, j
    # for row, column
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
    rem = P & (W - 1)
    diag_length = rem == 0 ? P : P - rem + W
    quote
        # We want i <= j, for a upper triangular matrix.
        i, j = minmax(i, j)
        i == j && return i
        num_blocks = (j-2) >> $Wshift
        ind = $diag_length + ((num_blocks*(num_blocks+1)) << $(2Wshift - 1))
        jrem = (j-2) & $(W-1)
        ind += ((num_blocks+1) * jrem) << $Wshift
        ind + i
    end
end
@generated function upper_triangle_sub2ind(::Val{P}, ::Type{T}, i, j) where {T,P}
    quote
        # $(Expr(:meta, :inline))
        $(upper_triangle_sub2ind_quote(P, T))
    end
end


@generated function Base.inv()

end
@generated function invchol()

end


# K = 12
# [upper_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [upper_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 13
# [upper_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [upper_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 14
# [upper_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [upper_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 15
# [upper_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [upper_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 16
# [upper_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [upper_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 17
# [upper_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [upper_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 18
# [upper_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [upper_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
#
#
#
# K = 12
# [lower_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [lower_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 13
# [lower_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [lower_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 14
# [lower_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [lower_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 15
# [lower_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [lower_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 16
# [lower_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [lower_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 17
# [lower_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [lower_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 18
# [lower_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [lower_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]

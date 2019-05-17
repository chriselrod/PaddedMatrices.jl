
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


# @generated function invchol(A::AbstractConstantFixedSizePaddedMatrix{P,P,T,L}) where {P,T,L}
#     quote
#         $(Expr(:meta,:inline)) # do we really want to force inline this?
#         mv = MutableFixedSizePaddedVector{N,T}(undef)
#         Adiag = A.diag
#         @vectorize $T for i ∈ 1:$P
#             mv[i] = rsqrt(Adiag[i])
#         end
#         Diagonal(ConstantFixedSizePaddedArray(mv))
#     end
# end

sym(A, i, j) = Symbol(A, :_, i, :_, j)
symi(A, i, j) = Symbol(A, :i_, i, :_, j)
function invchol_L_core_quote!(qa, P, output = :L, input = :S, ::Type{T} = Float64) where T
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(input, r, c)) -= $(sym(input, r, cr)) * $(sym(input, c, cr))  ) )
        end
        push!(qa, :( $(sym(input, c, c)) = sqrt( $(sym(input, c, c)) ) ) )
        push!(qa, :( $(sym(output, c, c)) = $(one(T)) / $(sym(input, c, c)) ))
        for r ∈ c+1:P
            push!(qa, :( $(sym(input, r, c)) *= $(sym(output, c, c))  ) )
        end
    end
    for c ∈ 1:P
        for r ∈ c+1:P
            push!(qa, :( $(sym(output, r, c)) = $(sym(input, r, c)) * $(sym(output, c, c)) ))
            for cr ∈ c+1:r-1
                push!(qa, :( $(sym(output, r, c)) += $(sym(input, r, cr)) * $(sym(output, cr, c)) ))
            end
            push!(qa, :( $(sym(output, r, c)) *=  -$(sym(output, r, r)) ) )
        end
    end
end
function invcholdet_L_core_quote!(qa, P, output = :L, input = :S, ::Type{T} = Float64) where T
    push!(qa, :(det = one($T)))
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(input, r, c)) -= $(sym(input, r, cr)) * $(sym(input, c, cr))  ) )
        end
        push!(qa, :( $(sym(input, c, c)) = sqrt( $(sym(input, c, c)) ) ) )
        push!(qa, :( $(sym(output, c, c)) = $(one(T)) / $(sym(input, c, c)) ))
        push!(qa, :( det *=  $(sym(output, c, c)) ))
        for r ∈ c+1:P
            push!(qa, :( $(sym(input, r, c)) *= $(sym(output, c, c))  ) )
        end
    end
    for c ∈ 1:P
        for r ∈ c+1:P
            push!(qa, :( $(sym(output, r, c)) = $(sym(input, r, c)) * $(sym(output, c, c)) ))
            for cr ∈ c+1:r-1
                push!(qa, :( $(sym(output, r, c)) += $(sym(input, r, cr)) * $(sym(output, cr, c)) ))
            end
            push!(qa, :( $(sym(output, r, c)) *=  -$(sym(output, r, r)) ) )
        end
    end
end
function chol_L_core_quote!(qa, P, input = :S, ::Type{T} = Float64, symout = sym, symin = sym) where T
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(input, r, c)) -= $(sym(input, r, cr)) * $(sym(input, c, cr))  ) )
        end
        push!(qa, :( $(sym(input, c, c)) = sqrt( $(sym(input, c, c)) ) ) )
        for r ∈ c+1:P
            push!(qa, :( $(sym(input, r, c)) /= $(sym(input, c, c))  ) )
        end
    end
end
function inv_L_core_quote!(qa, P, output = :L, input = :S, ::Type{T} = Float64) where T
    for c ∈ 1:P
        push!(qa, :( $(sym(output, c, c)) = $(one(T)) / $(sym(input, c, c)) ))
    end
    for c ∈ 1:P
        # push!(qa, :( $(sym(output, c, c)) = $(one(T)) / $(sym(input, c, c)) ))
        for r ∈ c+1:P
            push!(qa, :( $(sym(output, r, c)) = $(sym(input, r, c)) * $(sym(output, c, c)) ))
            for cr ∈ c+1:r-1
                push!(qa, :( $(sym(output, r, c)) += $(sym(input, r, cr)) * $(sym(output, cr, c)) ))
            end
            push!(qa, :( $(sym(output, r, c)) *=  -$(sym(output, r, r)) ) )
        end
    end
end
function load_L_quote!(qa, P, R, symbol_name = :L, extract_from = :L)
    for c ∈ 1:P, r ∈ c:P
        push!(qa, :($(sym(symbol_name, r, c)) = $extract_from[ $(r + (c-1)*R) ]) )
    end
end
function store_L_quote!(qa, P, R, output = :Li, ::Type{T} = Float64) where {T} # N x N block, with stride S.
    outtup = Union{Symbol,T}[]
    for c ∈ 1:P
        for r ∈ 1:c-1
            push!(outtup, zero(T))
        end
        for r ∈ c:P
            push!(outtup, sym(output, r, c))
        end
        for r ∈ P+1:R
            push!(outtup, zero(T))
        end
    end
    push!(qa, :(ConstantFixedSizePaddedMatrix{$P,$P,$T,$R,$(P*R)}($(
        Expr(:tuple, outtup...)
    ))))
end
function store_U_quote!(qa, P, R, output = :Li, ::Type{T} = Float64) where {T} # N x N block, with stride S.
    # W = VectorizationBase.pick_vector_width(P, T)
    # rem = P & (W-1)
    # L = rem == 0 ? P : P - rem + W
    outtup = Union{Symbol,T}[]
    for c ∈ 1:P
        for r ∈ 1:c
            push!(outtup, sym(output, c, r))
        end
        for r ∈ c+1:R
            push!(outtup, zero(T))
        end
    end
    push!(qa, :(ConstantFixedSizePaddedMatrix{$P,$P,$T,$R,$(P*R)}($(
        Expr(:tuple, outtup...)
    ))))
end
"""
Assumes the input matrix is positive definite (no checking is done; will return NaNs if not PD).
Uses the lower triangle of the input matrix S, and returns the upper triangle of the input matrix.
"""
@generated function invchol(S::AbstractConstantFixedSizePaddedMatrix{P,P,T,R}) where {P,T,R}
    # q = quote @fastmath @inbounds begin end end
    # qa = q.args[2].args[3].args[3].args
    q = quote end
    qa = q.args
    load_L_quote!(qa, P, R, :S, :S)
    invchol_L_core_quote!(qa, P, :L, :S, T)
    store_U_quote!(qa, P, R, :L, T)
    quote
        $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            $q
        end
    end
    # q
end
###
### invcholdet, LLi
### for Lower of S, Lower triangle out, and det is of the inverse
###
@generated function invcholdetLLi!(L::AbstractMutableFixedSizePaddedMatrix{P,P,T,R}, S::AbstractFixedSizePaddedMatrix{P,P,T,R}) where {P,T,R}
    # q = quote @fastmath @inbounds begin end end
    # qa = q.args[2].args[3].args[3].args
    q = quote end
    qa = q.args
    load_L_quote!(qa, P, R, :S, :S)
    invcholdet_L_core_quote!(qa, P, :L, :S, T)
    # store_L_quote!(qa, P, R, :L, T, false)
    storeq = quote end
    ind = 0
    for c ∈ 1:P
        for r ∈ 1:c-1
            ind += 1
            push!(storeq.args, :( L[$ind] = zero(T)) )
        end
        for r ∈ c:P
            ind += 1
            push!(storeq.args, :( L[$ind] = $(sym(:L, r, c))))
        end
        for r ∈ P+1:R
            ind += 1
            push!(storeq.args, :( L[$ind] = zero(T)) )
        end
    end
    quote
        $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            $q
            $storeq
            det
        end
    end
end
"""
Assumes the input matrix is positive definite (no checking is done; will return NaNs if not PD).
Uses the lower triangle of the input matrix S, and returns the upper triangle of the input matrix.
"""
@generated function chol(S::AbstractConstantFixedSizePaddedMatrix{P,P,T,R}) where {P,R,T}
    # q = quote @fastmath @inbounds begin end end
    # qa = q.args[2].args[3].args[3].args
    q = quote end
    qa = q.args
    load_L_quote!(qa, P, R, :S, :S)
    chol_L_core_quote!(qa, P, :S, T)
    store_L_quote!(qa, P, R, :S, T)
    quote
        $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            $q
        end
    end
    # q
end

function LAPACK_chol!(A::AbstractMutableFixedSizePaddedMatrix{N,N,Float64,LDA}, UPLO = 'U') where {N,LDA}
    INFO = 0
    ccall((LinearAlgebra.BLAS.@blasfunc(dpotrf_), LinearAlgebra.BLAS.libblas), Cvoid,
        (Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt}),
        UPLO, N, A, LDA, INFO)
end
function LAPACK_tri_inv!(A::AbstractMutableFixedSizePaddedMatrix{N,N,Float64,LDA}, UPLO = 'U', DIAG = 'N') where {N,LDA}
    INFO = 0
    ccall((LinearAlgebra.BLAS.@blasfunc(dtrtri_), LinearAlgebra.BLAS.libblas), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt}),
        UPLO, DIAG, N, A, LDA, INFO)
end
function BLAS_dtrmv!(A::AbstractMutableFixedSizePaddedMatrix{N,N,Float64,LDA}, x::AbstractMutableFixedSizePaddedVector{N,Float64,LDA}, UPLO = 'U', TRANS = 'N', DIAG = 'N') where {N,LDA}
    INCX = 1

    ccall((LinearAlgebra.BLAS.@blasfunc(dtrmv_), LinearAlgebra.BLAS.libblas), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}),
        UPLO, TRANS, DIAG, N, A, LDA, x, INCX)

end
function BLAS_dsymv!(A::AbstractMutableFixedSizePaddedMatrix{N,N,Float64,LDA}, x::AbstractMutableFixedSizePaddedVector{N,Float64,LDA}, y::AbstractMutableFixedSizePaddedVector{N,Float64,LDA}, α = 1.0, β = 0.0, UPLO = 'U') where {N,LDA}
    INCX = 1
    INCY = 1

    ccall((LinearAlgebra.BLAS.@blasfunc(dsymv_), LinearAlgebra.BLAS.libblas), Cvoid,
        (Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}),
        UPLO, N, α, A, LDA, x, INCX, β, y, INCY)

end
# function LAPACK_dsyrk!(A::AbstractMutableFixedSizePaddedMatrix{N,N,Float64,LDA}, UPLO = 'U', DIAG = 'N') where {N,LDA}
#     INFO = 0
#     ccall((LinearAlgebra.BLAS.@blasfunc(dsyrk_), LinearAlgebra.BLAS.libblas), Cvoid,
#         (Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt}),
#         UPLO, DIAG, N, A, LDA, INFO)
#
# end


"""
Assumes the input matrix is lower triangular.
"""
@generated function ltinv(L::AbstractConstantFixedSizePaddedMatrix{P,P,T,R}) where {P,T,R}
    # q = quote @fastmath @inbounds begin end end
    # qa = q.args[2].args[3].args[3].args
    q = quote end
    qa = q.args
    load_L_quote!(qa, P, R, :L, :L)
    inv_L_core_quote!(qa, P, :U, :L, T)
    store_U_quote!(qa, P, R, :U, T)
    quote
        $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            $q
        end
    end
    # q
end


"""
Fall back definition for scalars.
"""
@inline invchol(x) = SIMDPirates.rsqrt(x)

@generated function invchol(A::Diagonal{T,<:ConstantFixedSizeVector{N,T,P}}) where {N,T,P}
    quote
        $(Expr(:meta,:inline)) # do we really want to force inline this?
        mv = FixedSizeVector{N,T}(undef)
        Adiag = A.diag
        @vectorize $T for i ∈ 1:$P
            mv[i] = rsqrt(Adiag[i])
        end
        Diagonal(ConstantFixedSizeArray(mv))
    end
end


# @generated function invchol(A::AbstractConstantFixedSizeMatrix{P,P,T,L}) where {P,T,L}
#     quote
#         $(Expr(:meta,:inline)) # do we really want to force inline this?
#         mv = FixedSizeVector{N,T}(undef)
#         Adiag = A.diag
#         @vectorize $T for i ∈ 1:$P
#             mv[i] = rsqrt(Adiag[i])
#         end
#         Diagonal(ConstantFixedSizeArray(mv))
#     end
# end

sym(A, i, j) = Symbol(A, :_, i, :_, j)
symi(A, i, j) = Symbol(A, :i_, i, :_, j)
@noinline function invchol_L_core_quote!(qa::Vector{Any}, P::Int, output::Symbol = :L, input::Symbol = :S)
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(input, r, c)) -= $(sym(input, r, cr)) * $(sym(input, c, cr))  ) )
        end
        push!(qa, :( $(sym(input, c, c)) = sqrt( $(sym(input, c, c)) ) ) )
        push!(qa, :( $(sym(output, c, c)) = inv( $(sym(input, c, c)) )))
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
@noinline function invcholdeti_L_core_quote!(qa::Vector{Any}, P::Int, output::Symbol = :L, input::Symbol = :S)
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(input, r, c)) -= $(sym(input, r, cr)) * $(sym(input, c, cr))  ) )
        end
        outcc = sym(output, c, c)
        push!(qa, :( $(sym(input, c, c)) = sqrt( $(sym(input, c, c)) ) ) )
        push!(qa, :( $outcc = inv($(sym(input, c, c)) )))
        push!(qa, Expr(c == 1 ? :(=) : :(*=), :det, outcc))
        for r ∈ c+1:P
            push!(qa, :( $(sym(input, r, c)) *= $outcc  ) )
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
@noinline function invcholdetc_L_core_quote!(qa::Vector{Any}, P::Int, output::Symbol = :L, input::Symbol = :S)
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(input, r, c)) -= $(sym(input, r, cr)) * $(sym(input, c, cr))  ) )
        end
        incc = sym(input, c, c)
        push!(qa, :( $(sym(input, c, c)) = sqrt( $incc ) ) )
        push!(qa, :( $(sym(output, c, c)) = inv($incc )))
        push!(qa, Expr(c == 1 ? :(=) : :(*=), :det, incc))
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
@noinline function chol_L_core_quote!(qa::Vector{Any}, P::Int, input::Symbol = :S, symout::Symbol = sym, symin::Symbol = sym)
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
@noinline function inv_L_core_quote!(qa::Vector{Any}, P::Int, output::Symbol = :L, input::Symbol = :S)
    for c ∈ 1:P
        push!(qa, :( $(sym(output, c, c)) = inv($(sym(input, c, c)) )))
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
@noinline function load_L_quote!(qa::Vector{Any}, P::Int, R::Int, symbol_name::Symbol = :L, extract_from::Symbol = :L)
    for c ∈ 1:P, r ∈ c:P
        push!(qa, :($(sym(symbol_name, r, c)) = $extract_from[ $(r + (c-1)*R) ]) )
    end
end
@noinline function store_L_quote!(qa::Vector{Any}, P::Int, R::Int, output::Symbol = :Li, T = Float64)
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
    push!(qa, :(ConstantFixedSizeMatrix{$P,$P,$T,$R,$(P*R)}($(
        Expr(:tuple, outtup...)
    ))))
end
@noinline function store_U_quote!(qa::Vector{Any}, P::Int, R::Int, output::Symbol = :Li, T = Float64)
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
    push!(qa, :(ConstantFixedSizeMatrix{$P,$P,$T,$R,$(P*R)}($(
        Expr(:tuple, outtup...)
    ))))
end
"""
Assumes the input matrix is positive definite (no checking is done; will return NaNs if not PD).
Uses the lower triangle of the input matrix S, and returns the upper triangle of the input matrix.
"""
@generated function invchol(S::AbstractConstantFixedSizeMatrix{P,P,T,R}) where {P,T,R}
    # q = quote @fastmath @inbounds begin end end
    # qa = q.args[2].args[3].args[3].args
    q = quote end
    qa = q.args
    load_L_quote!(qa, P, R, :S, :S)
    invchol_L_core_quote!(qa, P, :L, :S)
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
@generated function invcholdetLLi!(L::AbstractMutableFixedSizeMatrix{P,P,T,R}, S::AbstractFixedSizeMatrix{P,P,T,R}) where {P,T,R}
    # q = quote @fastmath @inbounds begin end end
    # qa = q.args[2].args[3].args[3].args
    q = quote end
    qa = q.args
    load_L_quote!(qa, P, R, :S, :S)
    invcholdeti_L_core_quote!(qa, P, :L, :S)
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
@generated function invcholdetLLc!(L::AbstractMutableFixedSizeMatrix{P2,P1,T,R1}, S::AbstractFixedSizeMatrix{P3,P3,T,R2}) where {P1,P2,P3,T,R1,R2}
    # q = quote @fastmath @inbounds begin end end
    # qa = q.args[2].args[3].args[3].args
    q = quote end
    qa = q.args
    P = min(P1,P3)
    load_L_quote!(qa, P, R1, :S, :S)
    invcholdetc_L_core_quote!(qa, P, :L, :S)
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
        for r ∈ P+1:R1
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
@generated function chol(S::AbstractFixedSizeMatrix{P,P,T,R}) where {P,R,T}
    # q = quote @fastmath @inbounds begin end end
    # qa = q.args[2].args[3].args[3].args
    q = quote end
    qa = q.args
    load_L_quote!(qa, P, R, :S, :S)
    chol_L_core_quote!(qa, P, :S)
    store_L_quote!(qa, P, R, :S, T)
    quote
        $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            $q
        end
    end
    # q
end

function LAPACK_chol!(A::AbstractMutableFixedSizeMatrix{N,N,Float64,LDA}, UPLO::Char = 'U') where {N,LDA}
    INFO = 0
    ccall((LinearAlgebra.BLAS.@blasfunc(dpotrf_), LinearAlgebra.BLAS.libblas), Cvoid,
        (Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt}),
        UPLO, N, A, LDA, INFO)
end
function LAPACK_tri_inv!(A::AbstractMutableFixedSizeMatrix{N,N,Float64,LDA}, UPLO::Char = 'U', DIAG::Char = 'N') where {N,LDA}
    INFO = 0
    ccall((LinearAlgebra.BLAS.@blasfunc(dtrtri_), LinearAlgebra.BLAS.libblas), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt}),
        UPLO, DIAG, N, A, LDA, INFO)
end
function BLAS_dtrmv!(A::AbstractMutableFixedSizeMatrix{N,N,Float64,LDA}, x::AbstractMutableFixedSizeVector{N,Float64,LDA}, UPLO::Char = 'U', TRANS::Char = 'N', DIAG::Char = 'N') where {N,LDA}
    INCX = 1

    ccall((LinearAlgebra.BLAS.@blasfunc(dtrmv_), LinearAlgebra.BLAS.libblas), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}),
        UPLO, TRANS, DIAG, N, A, LDA, x, INCX)

end
function BLAS_dsymv!(A::AbstractMutableFixedSizeMatrix{N,N,Float64,LDA}, x::AbstractMutableFixedSizeVector{N,Float64,LDA}, y::AbstractMutableFixedSizeVector{N,Float64,LDA}, α::Float64 = 1.0, β::Float64 = 0.0, UPLO::Char = 'U') where {N,LDA}
    INCX = 1
    INCY = 1

    ccall((LinearAlgebra.BLAS.@blasfunc(dsymv_), LinearAlgebra.BLAS.libblas), Cvoid,
        (Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}),
        UPLO, N, α, A, LDA, x, INCX, β, y, INCY)

end
# function LAPACK_dsyrk!(A::AbstractMutableFixedSizeMatrix{N,N,Float64,LDA}, UPLO = 'U', DIAG = 'N') where {N,LDA}
#     INFO = 0
#     ccall((LinearAlgebra.BLAS.@blasfunc(dsyrk_), LinearAlgebra.BLAS.libblas), Cvoid,
#         (Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt}),
#         UPLO, DIAG, N, A, LDA, INFO)
#
# end


"""
Assumes the input matrix is lower triangular.
"""
@generated function ltinv(L::AbstractConstantFixedSizeMatrix{P,P,T,R}) where {P,T,R}
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


@noinline function safecholdet_L_core_quote!(qa::Vector{Any}, P::Int, output::Symbol = :L)
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(output, r, c)) -= $(sym(output, r, cr)) * $(sym(output, c, cr))  ) )
        end
        symcc = sym(output, c, c)
        push!(qa, :( $symcc > 0 || return (oftype($sym(output, c, c), -Inf), false) ))
        push!(qa, :( $symcc = sqrt( $symcc ) ) )
        push!(qa, Expr(c == 1 ? :(=) : :(*=), :det, symcc))
        push!(qa, :( $(symi(output, c, c)) = inv($symcc )))
        for r ∈ c+1:P
            push!(qa, :( $(sym(output, r, c)) *= $(symi(output, c, c))  ) )
        end
    end
end
@generated function safecholdet!(L::AbstractMutableFixedSizeMatrix{P,P,T,R}) where {P,T,R}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, R, :L, :L)
    safecholdet_L_core_quote!(qa, P, :L)
    for c ∈ 1:P
        for r ∈ 1:c-1
            push!(qa, zero(T))
        end
        for r ∈ c:P
            push!(qa, sym(:L, r, c))
        end
    end
    # store_L_quote!(qa, P, R, :L)
    push!(qa, :(det, true))
    q
end

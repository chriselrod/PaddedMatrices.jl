
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
function inv_L_core_quote!(qa, P, output = :L, input = :S, ::Type{T} = Float64) where T
    for c ∈ 1:P
        push!(qa, :( $(sym(output, c, c)) = $(one(T)) / $(sym(input, c, c)) ))
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
"""
Assumes the input matrix is lower triangular.
"""
@generated function ltinv(L::AbstractConstantFixedSizePaddedMatrix{P,P,T,R}) where {P,T,R}
    # q = quote @fastmath @inbounds begin end end
    # qa = q.args[2].args[3].args[3].args
    q = quote end
    qa = q.args
    load_L_quote!(qa, P, R, :S, :S)
    inv_L_core_quote!(qa, P, :L, :S, T)
    store_U_quote!(qa, P, R, :L, T)
    quote
        $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            $q
        end
    end
    # q
end

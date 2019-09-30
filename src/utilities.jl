

# function determine_vectorloads(num_rows, ::Type{T} = Float64) where T
#     W = VectorizationBase.pick_vector_width(num_rows, T)
#     num_vectors = cld(num_rows, W)
#     num_vectors, num_vectors * W
# end

@generated function Base.sum(A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    quote
        $(Expr(:meta, :inline))
        out = zero($T)
        @vectorize $T for i ∈ 1:$L
            out += A[i]
        end
        out
    end
end
@inline Base.sum(A::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T}}) where {S,T} = sum(A.parent)
@generated function Base.prod(A::AbstractFixedSizeVector{L,T}) where {L,T}
    quote
        $(Expr(:meta, :inline))
        out = one(T)
        @vectorize $T for i ∈ 1:$L
            out *= A[i]
        end
        out
    end
end

function Base.cumsum!(A::AbstractMutableFixedSizeVector{M}) where {M}
    @inbounds for m ∈ 2:M
        A[m] += A[m-1]
    end
    A
end
Base.cumsum(A::AbstractMutableFixedSizeVector) = cumsum!(copy(A))
Base.cumsum(A::AbstractConstantFixedSizeVector) = ConstantFixedSizeVector(cumsum!(MutableFixedSizeVector(A)))

# @generated function Base.maximum(A::AbstractFixedSizeArray{S,T,P,L}) where {S,T,P,L}
#     W, Wshift = VectorizationBase.pick_vector_width_shift(L, T)
#     V = Vec{W,T}
#     q = quote
#         $(Expr(:meta,:inline))
#         m = SIMDPirates.vbroadcast($V, -Inf)
#         pA = VectorizationBase.vectorizable(A)
#     end
#     fulliters = L >> Wshift
#     if fulliters > 0
#         push!(q.args,
#         quote
#             for l ∈ 0:$(fulliters-1)
#                 a = SIMDPirates.vabs(SIMDPirates.vload($V, pA + l*$W))
#                 m = SIMDPirates.vmax(a, m)
#             end
#         end)
#     end
#     r = L & (W - 1)
#     if r > 0
#         U = VectorizationBase.mask_type(W)
#         push!(q.args, quote
#             mask = $( U(2^r-1) )
#             a = SIMDPirates.vabs(SIMDPirates.vload($V, pA + $(fulliters*W), mask ))
#             m = vmax(m, a)
#         end)
#     end
#     push!(q.args, :(SIMDPirates.vmaximum(m)))
#     q
# end

@generated function Base.maximum(::typeof(abs), A::AbstractFixedSizeArray{S,T,X,L}) where {S,T,X,L}
    SV = S.parameters
    R1 = SV[1]
    D = 1
    for i ∈ 2:length(SV)
        D *= SV[i]
    end
    W, Wshift = VectorizationBase.pick_vector_width_shift(R1, T)
    V = Vec{W,T}
    q = quote
        $(Expr(:meta,:inline))
        m = SIMDPirates.vbroadcast($V, -Inf)
        pA = VectorizationBase.vectorizable(A)
    end
    fulliters = R1 >> Wshift
    innerloop = quote end
    if fulliters > 0
        push!(innerloop.args,
        quote
            for l ∈ 0:$(fulliters-1)
                a = SIMDPirates.vabs(SIMDPirates.vload($V, pA + l*$W + j*$(P*W) ))
                m = SIMDPirates.vmax(a, m)
            end
        end)
    end
    r = L & (W - 1)
    if r > 0
        U = VectorizationBase.mask_type(W)
        push!(innerloop.args, quote
            mask = $( U(2^r-1) )
            a = SIMDPirates.vabs(SIMDPirates.vload($V, pA + $(fulliters*W), mask ))
            m = SIMDPirates.vmax(m, a)
        end)
    end
    push!(q.args, quote
        for j ∈ 0:$(D-1)
            $innerloop
        end
        SIMDPirates.vmaximum(m)
    end)
    q
end

@generated function Base.maximum(::typeof(abs), A::AbstractFixedSizeVector{S,T,L}) where {S,T,L}
    W, Wshift = VectorizationBase.pick_vector_width_shift(S, T)
    V = Vec{W,T}
    q = quote
        $(Expr(:meta,:inline))
        m = SIMDPirates.vbroadcast($V, -Inf)
        pA = VectorizationBase.vectorizable(A)
    end
    fulliters = S >> Wshift
    if fulliters > 0
        push!(q.args,
        quote
            for l ∈ 0:$(fulliters-1)
                a = SIMDPirates.vabs(SIMDPirates.vload($V, pA + l*$W))
                m = SIMDPirates.vmax(a, m)
            end
        end)
    end
    r = S & (W - 1)
    if r > 0
        U = VectorizationBase.mask_type(W)
        push!(q.args, quote
            mask = $( U(2^r-1) )
            a = SIMDPirates.vabs(SIMDPirates.vload($V, pA + $(fulliters*W), mask ))
            m = SIMDPirates.vmax(m, a)
        end)
    end
    push!(q.args, :(SIMDPirates.vmaximum(m)))
    q
end

@inline Base.pointer(x::Symmetric{T,MutableFixedSizeMatrix{P,P,T,R,L}}) where {P,T,R,L} = pointer(x.data)


@generated function vexp!(
    B::AbstractMutableFixedSizeArray{S,T,N,X,L},
) where {S,T,N,X,L}
    quote
        $(Expr(:meta,:inline))
        LoopVectorization.@vvectorize $T for l ∈ 1:$L
            B[l] = SLEEFPirates.exp(B[l])
        end
        B
    end
end
@generated function vexp!(
    B::AbstractMutableFixedSizeArray{S,T,N,X,L},
    A::AbstractFixedSizeArray{S,T,N,X,L}
) where {S,T,N,X,L}
    quote
        $(Expr(:meta,:inline))
        LoopVectorization.@vvectorize $T for l ∈ 1:$L
            B[l] = SLEEFPirates.exp(A[l])
        end
        B
    end
end
@inline function vexp(
    A::AbstractFixedSizeArray{S,T,N,X,L}
) where {S,T,N,X,L}
    vexp!(MutableFixedSizeArray{S,T,N,X,L}(undef), A)
end
@inline function vexp(
    A::AbstractConstantFixedSizeArray{S,T,N,X,L}
) where {S,T,N,X,L}
    MutableFixedSizeArray{S,T,N,X,L}(undef) |>
        vexp! |>
        ConstantFixedSizeArray
end
function vexp(
    sp::StackPointer,
    A::AbstractFixedSizeArray{S,T,N,X,L}
) where {S,T,N,X,L}
    B = PtrArray{S,T,N,X,L}(pointer(sp,T))
    sp + VectorizationBase.align(sizeof(T)*L), vexp!(B, A)
end

@noinline function pointer_vector_expr(
    outsym::Symbol, @nospecialize(M::Union{Int,Symbol}), T, sp::Bool = true, ptrsym::Symbol = :sptr#; align_dynamic::Bool=true
)
    vector_expr = if M isa Int
        if sp
            :(PtrVector{$M,$T}($ptrsym))
        else
            :(MutableFixedSizeVector{$M,$T}(undef))
        end
    else
        if sp
            :(DynamicPointerArray{$T}($ptrsym, ($M,), VectorizationBase.align($M, $T)))# $(align_dynamic ? :(VectorizationBase.align($M, $T)) : $M) ))
        else
            :(Vector{$T}(undef, $M))
        end
    end
    Expr(:(=), sp ? :($ptrsym,$outsym) : outsym, vector_expr)
end

@noinline function pointer_matrix_expr(
    outsym::Symbol, @nospecialize(M::Union{Int,Symbol}), @nospecialize(N::Union{Int,Symbol}), T, sp::Bool = true, ptrsym::Symbol = :sptr#; align_dynamic::Bool=true
)
    matrix_expr = if M isa Int && N isa Int
        if sp
            :(PtrMatrix{$M,$N,$T}($ptrsym))
        else
            :(MutableFixedSizeMatrix{$M,$N,$T}(undef))
        end
    else
        if sp
            :(DynamicPointerArray{$T}($ptrsym, ($M,$N), $(M isa Int ? VectorizationBase.align(M,T) : :(VectorizationBase.align($M, $T)))))# $(align_dynamic ? :(VectorizationBase.align($M, $T)) : $M) ))
        else
            :(Matrix{$T}(undef, $M, $N))
        end
    end
    Expr(:(=), sp ? :($ptrsym,$outsym) : outsym, matrix_expr)
end


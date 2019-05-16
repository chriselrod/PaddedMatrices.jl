

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

# @generated function Base.maximum(A::AbstractFixedSizePaddedArray{S,T,P,L}) where {S,T,P,L}
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

# @generated
function Base.maximum(::typeof(abs), A::AbstractFixedSizePaddedArray{S,T,P,L}) where {S,T,P,L}
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

@generated function Base.maximum(::typeof(abs), A::AbstractFixedSizePaddedVector{S,T,P,L}) where {S,T,P,L}
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

@inline Base.pointer(x::Symmetric{T,MutableFixedSizePaddedMatrix{P,P,T,R,L}}) where {P,T,R,L} = pointer(x.data)

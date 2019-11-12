n

# function determine_vectorloads(num_rows, ::Type{T} = Float64) where T
#     W = VectorizationBase.pick_vector_width(num_rows, T)
#     num_vectors = cld(num_rows, W)
#     num_vectors, num_vectors * W
# end

@noinline function reduction_expr(S,T,N,X,L,op,f,reduction,init)
    FL = simple_vec_prod(S.parameters)
    if N == 1 || L == FL#first(S.parameters)::Int == (X.parameters[2])::Int
        FL = N == 1 ? first(S.parameters)::Int : FL
        # W, Wshift = VectorizationBase.pick_vector_width_shift(FL, T)
        return quote
            $(Expr(:meta, :inline))
            out = $init
            # @vvectorize $T 4 for i ∈ 1:$FL
            @inbounds @simd for i ∈ 1:$FL
                $(Expr(op, :out, :(A[i])))
            end
            out
        end
    end
    W = VectorizationBase.pick_vector_width(T)
    Xv = X.parameters
    Sv = S.parameters
    @assert first(Xv) == 1 "Sum for non-unit stride not yet implemented."
    q = quote
        # @vvectorize for i_1 ∈ 1:$(first(Sv))
        @inbounds @simd for i_1 ∈ 1:$(first(Sv))
            out = $f(A[i_1 + incr_1], out)
        end
    end
    for n ∈ 2:N
        i_n = Symbol(:i_,n)
        q = quote
            for $i_n ∈ 0:$(Sv[n] - 1)
                $(Symbol(:incr_,n-1)) = $(n == N ? :($i_n * $(Xv[N])) : :($(Symbol(:incr_,n)) + $i_n * $(Xv[n])))
                $q
            end
        end
    end
    quote
        out = vbroadcast(Vec{$W,$T}, $init)
        $q
        $reduction(out)
    end
end

@generated function Base.sum(A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,X,N,L}
    reduction_expr(S,T,N,X,L,:(+=),:vadd,:vsum,zero(T))
end
@inline Base.sum(A::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T}}) where {S,T} = sum(A.parent)
@generated function Base.prod(A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    reduction_expr(S,T,N,X,L,:(*=),:vmul,:vprod,one(T))
end

function Base.cumsum!(A::AbstractMutableFixedSizeVector{M}) where {M}
    @inbounds for m ∈ 2:M
        A[m] += A[m-1]
    end
    A
end
Base.cumsum(A::AbstractMutableFixedSizeVector) = cumsum!(copy(A))
Base.cumsum(A::AbstractConstantFixedSizeVector) = ConstantFixedSizeVector(cumsum!(FixedSizeVector(A)))

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

@generated function Base.maximum(
    ::typeof(abs),
    A::AbstractFixedSizeArray{S,T,X,L}
) where {S,T,X,L}
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

@generated function Base.maximum(
    ::typeof(abs),
    A::AbstractFixedSizeVector{S,T,L}
) where {S,T,L}
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

@inline Base.pointer(x::Symmetric{T,<:AbstractMutableFixedSizeMatrix{P,P,T,R,L}}) where {P,T,R,L} = pointer(x.data)


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
    vexp!(FixedSizeArray{S,T,N,X,L}(undef), A)
end
@inline function vexp(
    A::AbstractConstantFixedSizeArray{S,T,N,X,L}
) where {S,T,N,X,L}
    FixedSizeArray{S,T,N,X,L}(undef) |>
        vexp! |>
        ConstantFixedSizeArray
end
function vexp(
    sp::StackPointer,
    A::AbstractFixedSizeArray{S,T,N,X,L}
) where {S,T,N,X,L}
    sp, B = PtrArray{S,T,N,X,L}(sp)
    sp, vexp!(B, A)
end

@noinline function pointer_vector_expr(
    outsym::Symbol, @nospecialize(M::Union{Int,Symbol}), T, sp::Bool = true, ptrsym::Symbol = :sptr#; align_dynamic::Bool=true
)
    vector_expr = if M isa Int
        if sp
            :(PtrVector{$M,$T}($ptrsym))
        else
            :(FixedSizeVector{$M,$T}(undef))
        end
    else
        if sp
            :(DynamicPointerArray{$T}($ptrsym, ($M,), PaddedMatrices.calc_padding($M, $T)))# $(align_dynamic ? :(VectorizationBase.align($M, $T)) : $M) ))
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
            :(FixedSizeMatrix{$M,$N,$T}(undef))
        end
    else
        if sp
            :(DynamicPointerArray{$T}($ptrsym, ($M,$N), $(M isa Int ? calc_padding(M,T) : :(PaddedMatrices.calc_padding($M, $T)))))# $(align_dynamic ? :(VectorizationBase.align($M, $T)) : $M) ))
        else
            :(Matrix{$T}(undef, $M, $N))
        end
    end
    Expr(:(=), sp ? :($ptrsym,$outsym) : outsym, matrix_expr)
end

unique_copyto!(ptr::Ptr{T}, x::T) = VectorizationBase.store!(ptr, x)
function unique_copyto!(ptr::Ptr{T}, x::Union{AbstractPaddedVector{T},AbstractArray{T}}) where {T}
    @inbounds for i ∈ eachindex(x)
        VectorizationBase.store!(ptr, x[i])
        ptr += sizeof(T)
    end
end
@generated function unique_copyto!(ptr::Ptr{T}, x::AbstractPaddedArray{T,N}) where {T,N}
    quote
        ptrx = pointer(x)
        @nloops $N n x begin
            VectorizationBase.store!(ptr, (Base.Cartesian.@nref $N x n))
            ptr += $(sizeof(T))
        end
    end
end
@generated function unique_copyto!(ptr::Ptr{T}, A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    nrows = first(S.parameters)::Int
    if N > 1 && first(S.parameters)::Int != (X.parameters[2])::Int
        P = (X.parameters[2])::Int
        return quote
            ind = 0
            Base.Cartesian.@nloops $(N-1) i j -> 1:size(A,j+1) begin
                @vvectorize $T 4 for i_0 ∈ 1:$nrows
                    ptr[i_0] = A[ind + i_0]
                end
                ind += $P
                ptr += $(sizeof(T)*$nrows)
            end
        end
    else #if N == 1
        return quote
            @vvectorize $T 4 for i ∈ 1:$nrows
                ptr[i] = A[i]
            end
        end
    end
end



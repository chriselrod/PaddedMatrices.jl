
function Base.sum(A::AbstractFixedSizeArray)
    s = zero(eltype(A))
    @avx for i ∈ eachindex(A)
        s += A[i]
    end
    s
end
function Base.prod(A::AbstractFixedSizeArray)
    s = zero(eltype(A))
    @avx for i ∈ eachindex(A)
        s *= A[i]
    end
    s
end
                  
@inline Base.sum(A::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T}}) where {S,T} = sum(A.parent)
@inline Base.prod(A::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T}}) where {S,T} = prod(A.parent)

function Base.cumsum!(A::AbstractMutableFixedSizeVector{M}) where {M}
    @inbounds for m ∈ 2:M
        A[m] += A[m-1]
    end
    A
end
Base.cumsum(A::AbstractMutableFixedSizeVector) = cumsum!(copy(A))
Base.cumsum(A::AbstractConstantFixedSizeVector) = ConstantFixedSizeVector(cumsum!(FixedSizeVector(A)))

@generated function Base.all(::typeof(isfinite), a::AbstractFixedSizeVector{M,T}) where {M,T}
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    U = VectorizationBase.mask_type(W)
    reps = M >> Wshift
    rem = M & (W - 1)
    quote
        $(Expr(:meta,:inline))
        mask = $(typemax(U))
        va = VectorizationBase.vectorizable(a)
        i = 0
        for _ ∈ 1:$reps
            v = vload(Vec{$W,$T}, va, i)
            mask &= SIMDPirates.visfinite_mask(v)
            i += $W
        end
        v = vload(Vec{$W,$T}, va, i, $(VectorizationBase.mask(T, rem)))
        mask &= SIMDPirates.visfinite_mask(v)
        mask >= $(Base.unsafe_trunc(U, (1 << W) - 1))
    end
end

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

unique_copyto!(ptr::Ptr{T}, x::T) where {T} = VectorizationBase.store!(ptr, x)
function unique_copyto!(ptr::Ptr{T}, x::Union{AbstractPaddedVector{T},AbstractArray{T}}) where {T}
    @inbounds for i ∈ eachindex(x)
        VectorizationBase.store!(ptr, x[i])
        ptr = gep(ptr, 1)
    end
end
@generated function unique_copyto!(ptr::Ptr{T}, x::AbstractPaddedArray{T,N}) where {T,N}
    quote
        ptrx = pointer(x)
        Base.Cartesian.@nloops $N n x begin
            VectorizationBase.store!(ptr, (Base.Cartesian.@nref $N x n))
            ptr = gep(ptr, 1)
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
                ptrA = gep(pointer(A), ind)
                @avx for i_0 ∈ 1:$nrows
                    ptr[i_0] = ptrA[i_0]
                end
                ind += $P
                ptr = gep(ptr, $nrows)
            end
        end
    else #if N == 1
        return quote
            @avx for i ∈ 1:$nrows
                ptr[i] = A[i]
            end
        end
    end
end




struct ConstantFixedSizePaddedArray{S,T,N,P,L} <: AbstractConstantFixedSizePaddedArray{S,T,N,P,L}
    data::NTuple{L,T}
end
const ConstantFixedSizePaddedVector{S,T,P,L} = ConstantFixedSizePaddedArray{Tuple{S},T,1,P,L}
const ConstantFixedSizePaddedMatrix{M,N,T,P,L} = ConstantFixedSizePaddedArray{Tuple{M,N},T,2,P,L}

@noinline function calc_NP(S, L)
    SV = S.parameters
    N = length(SV)
    P = L
    for n ∈ 2:N
        P ÷= (SV[n])::Int
    end
    N, P
end
@noinline function pick_L(N, T = Float64)
    VectorizationBase.align(N, T)
end

@inline ConstantFixedSizePaddedVector{N}(data::NTuple{L,T}) where {N,L,T} = ConstantFixedSizePaddedVector{N,T,L,L}(data)
@generated function ConstantFixedSizePaddedArray{S}(data::NTuple{L,T}) where {S,T,L}
    N, P = calc_NP(S, L)
    quote
        $(Expr(:meta,:inline))
        ConstantFixedSizePaddedArray{$S,$T,$N,$P,$L}(data)
    end
end
@generated function ConstantFixedSizePaddedArray{S,T}(data::NTuple{L,T}) where {S,T,L}
    N, P = calc_NP(S, L)
    quote
        $(Expr(:meta,:inline))
        ConstantFixedSizePaddedArray{$S,$T,$N,$P,$L}(data)
    end
end
@generated function ConstantFixedSizePaddedArray{S,T,N}(data::NTuple{L,T}) where {S,T,L,N}
    Nv2, P = calc_NP(S, L)
    @assert N == Nv2
    quote
        $(Expr(:meta,:inline))
        ConstantFixedSizePaddedArray{$S,$T,$N,$P,$L}(data)
    end
end
@inline ConstantFixedSizePaddedVector{N,T,P}(A::NTuple{P,T}) where {N,T,P} = ConstantFixedSizePaddedVector{N,T,P,P}(A)
@inline ConstantFixedSizePaddedVector(A::AbstractFixedSizePaddedArray{S,T,1,P,L}) where {S,T,P,L} = ConstantFixedSizePaddedArray{S,T,1,P,L}(A.data)
@inline ConstantFixedSizePaddedMatrix(A::AbstractFixedSizePaddedArray{S,T,2,P,L}) where {S,T,P,L} = ConstantFixedSizePaddedArray{S,T,2,P,L}(A.data)
@inline ConstantFixedSizePaddedArray(A::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L} = ConstantFixedSizePaddedArray{S,T,N,P,L}(A.data)

@generated function ConstantFixedSizePaddedMatrix{M,N}(data::Matrix{T}) where {M,N,T}
    P = pick_L(M, T)
    outtup = Expr(:tuple)
    ind = 0
    for n ∈ 1:N
        for m ∈ 1:M
            ind += 1
            push!(outtup.args, :(data[$ind]))
        end
        for p ∈ M+1:P
            push!(outtup.args, zero(T))
        end
    end
    :(@inbounds ConstantFixedSizePaddedMatrix{$M,$N,$T}($outtup))
end

@generated function Base.fill(v::T, ::Type{<:ConstantFixedSizePaddedArray{S}}) where {S,T}
    N, P, L = calc_NPL(S, T)
    quote
        $(Expr(:meta,:inline))
        ConstantFixedSizePaddedArray{$S,$T,$N,$P,$L}(ntuple(l -> v, Val{$L}()))
    end
end
@inline filltuple(v, ::Val{L}) where {L} = ntuple(l -> v, Val{L}())
@generated function Base.fill(v::T, ::Static{S}) where {S,T}
    if isa(S, Integer)
        ST = Tuple{S}
    else
        ST = S
    end
    N, P, L = calc_NPL(ST, T)
    quote
        $(Expr(:meta,:inline))
        ConstantFixedSizePaddedArray{$ST,$T,$N,$P,$L}(filltuple(v, Val{$L}()))
    end
end
# @inline function ConstantFixedSizePaddedMatrix(A::AbstractMutableFixedSizePaddedArray{S,T,2,P,L}) where {S,T,P,L}
#     ConstantFixedSizePaddedArray{S,T,2,P,L}(unsafe_load(Base.unsafe_convert(Ptr{NTuple{L,T}}, pointer(A))))
#     # ConstantFixedSizePaddedArray{S,T,2,P,L}(A.data)
# end

# Type unstable convenience method
function ConstantFixedSizePaddedVector(x::Vector{T}) where {T}
    @assert isbitstype(T)
    N = length(x)
    Wm1 = VectorizationBase.pick_vector_width(N, T) - 1
    L = (N + Wm1) & ~Wm1
    ConstantFixedSizePaddedVector{N,T,L,L}(ntuple(i -> i > N ? zero(T) : x[i], L))
end

@generated function Base.size(::AbstractConstantFixedSizePaddedArray{S,T,N}) where {S,T,N}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:tuple, [S.parameters[n] for n ∈ 1:N]...))
    end
end
@generated function Base.length(A::AbstractConstantFixedSizePaddedArray{S,T,N}) where {S,T,N}
    SV = S.parameters
    L = SV[1]
    for n ∈ 2:N
        L *= SV[n]
    end
    quote
        $(Expr(:meta, :inline))
        $L
    end
end

@inline function Base.getindex(A::AbstractConstantFixedSizePaddedArray{S,T,1,L,L}, i::Int) where {S,T,L}
    @boundscheck i <= L || ThrowBoundsError("Index $i < full length $(full_length(A)).")
    @inbounds A.data[i]
end
@inline function Base.getindex(A::AbstractConstantFixedSizePaddedArray{S,T,1,L,L}, i::Int, j::Int) where {S,T,L}
    @boundscheck begin
        j == 1 || ThrowBoundsError("Column index j = $j != 1.")
        i <= L || ThrowBoundsError("Index $i < full length $(full_length(A)).")
    end
    @inbounds A.data[i]
end
@inline function Base.getindex(A::AbstractConstantFixedSizePaddedArray, i::Int)
    @boundscheck i <= full_length(A) || ThrowBoundsError("Index $i < full length $(full_length(A)).")
    @inbounds A.data[i]
end
@inline function Base.getindex(A::LinearAlgebra.Adjoint{Union{},<: AbstractConstantFixedSizePaddedMatrix{M,N,Vec{W,T}}}, i::Int) where {M,N,W,T}
    @boundscheck i <= full_length(A) || ThrowBoundsError("Index $i < full length $(full_length(A)).")
    @inbounds A.parent.data[i]
end
@generated function Base.getindex(A::AbstractConstantFixedSizePaddedArray{S,T,N,P}, i::Vararg{<:Integer,N}) where {S,T,N,P}
    SV = S.parameters
    ex = sub2ind_expr(SV, P)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $P : i[d] > $SV[d]) d->ThrowBoundsError() d -> nothing
        end
        @inbounds A.data[$ex]
    end
end
@generated function Base.getindex(A::AbstractConstantFixedSizePaddedArray{S,T,N,P}, i::CartesianIndex{N}) where {S,T,N,P}
    SV = S.parameters
    ex = sub2ind_expr(SV, P)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $P : i[d] > $SV[d]) d->ThrowBoundsError() d -> nothing
        end
        @inbounds A.data[$ex]
    end
end
@generated function Base.getindex(A::LinearAlgebra.Adjoint{Union{},<:AbstractConstantFixedSizePaddedMatrix{M,N,Vec{W,T},R}}, i::Int, j::Int) where {M,N,W,T,R}
    quote
        $(Expr(:meta, :inline))
        @boundscheck if (i > N) || (j > M)
            ThrowBoundsError("(i > N) = ($i > $N) || (j > M) = ($j > $M).")
        end
        @inbounds A.parent.data[ (i-1)*R + j ]
    end
end

@generated function Base.strides(A::AbstractConstantFixedSizePaddedArray{S,T,N,P}) where {S,T,N,P}
    SV = S.parameters
    N = length(SV)
    N == 1 && return (1,)
    last = P
    q = Expr(:tuple, 1, last)
    for n ∈ 3:N
        last *= SV[n-1]
        push!(q.args, last)
    end
    q
end

@generated function Base.zero(::Type{<:ConstantFixedSizePaddedVector{L,T}}) where {L,T}
    Wm1 = VectorizationBase.pick_vector_width(L,T)
    N = (L + Wm1) & ~Wm1
    :(ConstantFixedSizePaddedVector{$L,$T,$N,$N}($(Expr(:tuple,zeros(N)...))))
end
@inline return_mutable(A::AbstractMutableFixedSizePaddedArray) = A
@inline return_mutable(A::ConstantFixedSizePaddedArray) = MutableFixedSizePaddedArray(A)

struct vStaticPaddedArray{SPA}
    spa::SPA
    offset::Int
end
@inline VectorizationBase.vectorizable(A::vStaticPaddedArray) = A
@inline VectorizationBase.vectorizable(A::AbstractConstantFixedSizePaddedArray) = vStaticPaddedArray(A,0)
@inline VectorizationBase.vectorizable(A::LinearAlgebra.Diagonal{T,<:AbstractConstantFixedSizePaddedArray}) where {T} = vStaticPaddedArray(A.diag,0)
@inline VectorizationBase.vectorizable(A::LinearAlgebra.Adjoint{T,<:AbstractConstantFixedSizePaddedArray}) where {T} = vStaticPaddedArray(A.parent,0)
@inline Base.:+(A::vStaticPaddedArray, i) = vStaticPaddedArray(A.spa, A.offset + i)
@inline Base.:+(i, A::vStaticPaddedArray) = vStaticPaddedArray(A.spa, A.offset + i)
@inline Base.:-(A::vStaticPaddedArray, i) = vStaticPaddedArray(A.spa, A.offset - i)

@generated function SIMDPirates.vload(::Type{Vec{N,T}}, A::vStaticPaddedArray, i::Int) where {N,T}
    quote
        $(Expr(:meta, :inline))
        @inbounds $(Expr(:tuple, [:(Core.VecElement{$T}(A.spa.data[i + $n + A.offset])) for n ∈ 0:N-1]...))
    end
end
@generated function SIMDPirates.vload(::Type{SVec{N,T}}, A::vStaticPaddedArray, i::Int) where {N,T}
    quote
        $(Expr(:meta, :inline))
        @inbounds SVec($(Expr(:tuple, [:(Core.VecElement{$T}(A.spa.data[i + $n + A.offset])) for n ∈ 0:N-1]...)))
    end
end
@generated function SIMDPirates.vload(::Type{Vec{N,T}}, A::vStaticPaddedArray) where {N,T}
    quote
        $(Expr(:meta, :inline))
        @inbounds $(Expr(:tuple, [:(Core.VecElement{$T}(A.spa.data[$n + A.offset])) for n ∈ 1:N]...))
    end
end
@generated function SIMDPirates.vload(::Type{SVec{N,T}}, A::vStaticPaddedArray) where {N,T}
    quote
        $(Expr(:meta, :inline))
        @inbounds SVec($(Expr(:tuple, [:(Core.VecElement{$T}(A.spa.data[$n + A.offset])) for n ∈ 1:N]...)))
    end
end
### It is assumed that there is sufficient padding for the load to be unmasked.
### The mask is then applied via "vifelse". Because of the possibility of contaminants cropping up in the padding,
### like NaNs when taking the inverse, applying the mask is necessary for general correctness.
### Perhaps an opt-out method should be provided.
### Or perhaps checks to see if the compiler can figure out it can drop the masking?
### Either way, in a few tests the compiler was smart enough to turn this into a masked load.
@inline function SIMDPirates.vload(::Type{V}, A::vStaticPaddedArray, mask::Union{Vec{N,Bool},<:Unsigned}) where {T,N,V <: Union{Vec{N,T},SVec{N,T}}}
    v = vload(V, A)
    vifelse(mask, v, vbroadcast(V, zero(T)))
end

@inline Base.unsafe_load(A::vStaticPaddedArray) = @inbounds A.spa.data[A.offset + 1]
@inline Base.unsafe_load(A::vStaticPaddedArray, i::Int) = @inbounds A.spa.data[A.offset + i]
@inline VectorizationBase.load(A::vStaticPaddedArray) = @inbounds A.spa.data[A.offset + 1]
@inline Base.getindex(A::vStaticPaddedArray, i::Int) = @inbounds A.spa.data[A.offset + i]



@noinline function vload_constant_matrix_quote(T, SV::Core.SimpleVector, @nospecialize(A))
    # SV = S.parameters
    N = length(SV)
    nrow = (SV[1])::Int
    W, Wshift = VectorizationBase.pick_vector_width_shift(nrow, T)

    num_unmasked_loads_per_row = nrow >>> Wshift
    remainder = nrow & (W - 1)

    padded_rows = remainder == 0 ? nrow : nrow + W - remainder

    remaining_dims = 1
    for n ∈ 2:N
        remaining_dims *= (SV[n])::Int
    end
    L = padded_rows * remaining_dims

    W_full, Wshift_full = VectorizationBase.pick_vector_width_shift(L, T)
    num_unmasked_loads = L >>> Wshift_full
    remainder_full = L & (W_full - 1)

    q = quote end
    output = Expr(:tuple)

    V = Vec{W_full,T}
    if remainder == 0 # W == W_full

        for n ∈ 0:num_unmasked_loads-1
            v = gensym(:v)
            push!(q.args, :($v = vload($V, ptr + $(W_full*n))))
            for w ∈ 1:W_full
                push!(output.args, :($v[$w].value))
            end
        end
        if remainder_full > 0
            v = gensym(:v)
            push!(q.args, :($v = vload($V, ptr + $(W_full*num_unmasked_loads), $(unsafe_trunc(VectorizationBase.mask_type(W_full), (2^remainder_full-1)) ))))
            for w ∈ 1:remainder_full
                push!(output.args, :($v[$w].value))
            end
        end

    elseif W != W_full # remainder != 0
        # We need to concatenate multiple masks. A single load covers more than one column.
        mask = concatenate_masks(VectorizationBase.mask_type(W_full), W, W_full, remainder)
        for n ∈ 0:num_unmasked_loads-1
            v = gensym(:v)
            push!(q.args, :($v = vload($V, ptr + $(W_full*n), $mask)))
            for w ∈ 1:W_full
                push!(output.args, :($v[$w].value))
            end
        end
        if remainder_full > 0
            v = gensym(:v)
            push!(q.args, :($v = vload($V, ptr + $(W_full*n), $( mask & unsafe_trunc(VectorizationBase.mask_type(W_full), (2^remainder_full-1)) ))))
            for w ∈ 1:remainder_full
                push!(output.args, :($v[$w].value))
            end
        end

    else # W == W_full and remainder != 0; remainder_full == remainder because W == W_full
        # A single load covers at most 1 column. The last load has rows.
        for c ∈ 0:remaining_dims-1
            for n ∈ 0:num_unmasked_loads_per_row-1
                v = gensym(:v)
                push!(q.args, :($v = vload($V, ptr + $(W_full*n + c*nrow))))
                for w ∈ 1:W_full
                    push!(output.args, :($v[$w].value))
                end
            end
            v = gensym(:v)
            push!(q.args, :($v = vload($V, ptr + $(W_full*num_unmasked_loads_per_row + c*nrow), $(unsafe_trunc(VectorizationBase.mask_type(W_full), (2^remainder-1))))))
            for w ∈ 1:W_full
                push!(output.args, :($v[$w].value))
            end
        end

    end
    quote
        $(Expr(:meta,:inline))
        $q
        @inbounds $A($output)
    end
end
@noinline function concatenate_masks(T, Wsmall, Wfull, remainder)
    single_mask = unsafe_trunc(T, 1 << remainder - 1)
    while Wsmall < Wfull
        single_mask_old = single_mask
        single_mask <<= Wsmall
        single_mask += single_mask_old
        Wsmall <<= 1
    end
    single_mask
end

# vStaticPaddedArray
@generated function SIMDPirates.vload(::Type{A}, ptr::VectorizationBase.Pointer{T}) where {T, S, A <: AbstractConstantFixedSizePaddedArray{S,T}}
    vload_constant_matrix_quote(T, S.parameters, A)
end
@generated function SIMDPirates.vload(::Type{A}, ptr::vStaticPaddedArray{SPA}) where {T, S,S2, A <: AbstractConstantFixedSizePaddedArray{S,T}, SPA <: AbstractConstantFixedSizePaddedArray{S2,T}}
    vload_constant_matrix_quote(T, S.parameters, A)
end


@generated function broadcast_array_quote(S, P, W, T, ::Val{N}) where {N}
    quote
        q = quote end
        SV = S.parameters
        V = Vec{W,T}
        outtup = Expr(:tuple,)
        indbase = 0
        Base.Cartesian.@nloops $(N-1) i j -> 1:(SV[j+1])::Int begin
            ind = indbase
            @inbounds for i_0 ∈ 1:SV[1]
                ind += 1
                push!(q.args, :($(Symbol(:A_, ind)) = vbroadcast(Vec{$W,$T}, @inbounds A[$ind])))
                push!(outtup.args, Symbol(:A_, ind))
            end
            indbase += P
        end
        push!(q.args, Expr(:call,
            Expr(:curly, :ConstantFixedSizePaddedArray, S, V, N, (SV[1])::Int, simple_vec_prod(SV)),
            outtup)
        )
        q
    end
end
@generated function SIMDPirates.vbroadcast(::Type{Vec{W,T}}, A::ConstantFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L,W}
    broadcast_array_quote(S, P, W, T, Val(N))
end

@generated function Base.vcat(a::AbstractConstantFixedSizePaddedVector{M,T}, b::AbstractConstantFixedSizePaddedVector{N,T}) where {M,N,T}
    MpN = M + N
    L = pick_L(MpN, T)
    outtup = Expr(:tuple,)
    for m ∈ 1:M
        push!(outtup.args, :(a[$m]))
    end
    for n ∈ 1:N
        push!(outtup.args, :(b[$n]))
    end
    for z ∈ MpN+1:L
        push!(outtup.args, zero(T))
    end
    :(@inbounds ConstantFixedSizePaddedVector{$MpN,$T}($outtup))
end
Base.vcat(a::AbstractConstantFixedSizePaddedVector, b::AbstractConstantFixedSizePaddedVector, c::AbstractFixedSizePaddedVector...) = vcat(vcat(a,b), c...)

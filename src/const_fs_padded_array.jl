
struct ConstantFixedSizeArray{S,T,N,X,L} <: AbstractConstantFixedSizeArray{S,T,N,X,L}
    data::NTuple{L,Core.VecElement{T}}
end
const ConstantFixedSizeVector{S,T,L} = ConstantFixedSizeArray{Tuple{S},T,1,Tuple{1},L}
const ConstantFixedSizeMatrix{M,N,T,P,L} = ConstantFixedSizeArray{Tuple{M,N},T,2,Tuple{1,P},L}

@noinline function calc_NP(SV::Core.SimpleVector, L)
    N = length(SV)
    P = Vector{Int}(undef, N)
    P[N] = L ÷ (SV[N])::Int
    for n ∈ N-1:-1:2
        P[n-1] = P[n] ÷ (SV[n])::Int
    end
    N, Tuple{P...}
end

@inline ConstantFixedSizeVector{N}(data::NTuple{L,T}) where {N,L,T} = ConstantFixedSizeVector{N,T,L}(data)
@generated function ConstantFixedSizeArray{S}(data::NTuple{L,T}) where {S,T,L}
    N, P = calc_NP(S.parameters, L)
    quote
        $(Expr(:meta,:inline))
        ConstantFixedSizeArray{$S,$T,$N,$P,$L}(data)
    end
end
@generated function ConstantFixedSizeArray{S,T}(data::NTuple{L,T}) where {S,T,L}
    N, P = calc_NP(S.parameters, L)
    quote
        $(Expr(:meta,:inline))
        ConstantFixedSizeArray{$S,$T,$N,$P,$L}(data)
    end
end
@generated function ConstantFixedSizeArray{S,T,N}(data::NTuple{L,T}) where {S,T,L,N}
    Nv2, P = calc_NP(S.parameters, L)
    @assert N == Nv2
    quote
        $(Expr(:meta,:inline))
        ConstantFixedSizeArray{$S,$T,$N,$P,$L}(data)
    end
end
# @inline ConstantFixedSizeVector{N,T,P}(A::NTuple{P,T}) where {N,T,P} = ConstantFixedSizeVector{N,T,P}(A)
# @inline ConstantFixedSizeVector(A::AbstractFixedSizeVector{S,T,L}) where {S,T,L} = ConstantFixedSizeArray{S,T,1,Tuple{1},L}(A.data)
# @inline ConstantFixedSizeMatrix(A::AbstractFixedSizeMatrix{M,N,T,P,L}) where {M,N,T,P,L} = ConstantFixedSizeArray{Tuple{M,N},T,2,P,L}(A.data)
@inline ConstantFixedSizeArray(A::AbstractFixedSizeArray{S,T,N,P,L}) where {S,T,N,P,L} = ConstantFixedSizeArray{S,T,N,P,L}(A.data)
@inline ConstantFixedSizeMatrix(A::AbstractFixedSizeArray{S,T,2,P,L}) where {S,T,P,L} = ConstantFixedSizeArray{S,T,2,P,L}(A.data)

@generated function ConstantFixedSizeMatrix{M,N}(data::Matrix{T}) where {M,N,T}
    X = calc_padding(M, T)
    P = (X.parameters[2])::Int
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
    :(@inbounds ConstantFixedSizeMatrix{$M,$N,$T,$P}($outtup))
end

@generated function Base.fill(v::T, ::Type{<:ConstantFixedSizeArray{S}}) where {S,T}
    N, P, L = calc_NPL(S.parameters, T)
    quote
        $(Expr(:meta,:inline))
        ConstantFixedSizeArray{$S,$T,$N,$P,$L}(ntuple(_ -> v, Val{$L}()))
    end
end
@generated function Base.fill(v::T, ::Static{S}) where {S,T}
    if isa(S, Integer)
        ST = Tuple{S}
    else
        ST = S
    end
    N, P, L = calc_NPL(ST.parameters, T)
    quote
        $(Expr(:meta,:inline))
        ConstantFixedSizeArray{$ST,$T,$N,$P,$L}(ntuple(_ -> v, Val{$L}()))
    end
end

# Type unstable convenience method
function ConstantFixedSizeVector(x::Vector{T}) where {T}
    @assert isbitstype(T)
    N = length(x)
    L = calc_padding(N)
    @inbounds ConstantFixedSizeVector{N,T,L}(ntuple(i -> i > N ? zero(T) : x[i], L))
end

@inline function Base.getindex(A::AbstractConstantFixedSizeArray{S,T,1,Tuple{1},L}, i::Int) where {S,T,L}
    @boundscheck i <= L || ThrowBoundsError("Index $i < full length $(full_length(A)).")
    @inbounds A.data[i]
end
@inline function Base.getindex(A::AbstractConstantFixedSizeArray{S,T,1,Tuple{1},L}, i::Int, j::Int) where {S,T,L}
    @boundscheck begin
        j == 1 || ThrowBoundsError("Column index j = $j != 1.")
        i <= L || ThrowBoundsError("Index $i < full length $(full_length(A)).")
    end
    @inbounds A.data[i]
end
@inline function Base.getindex(A::AbstractConstantFixedSizeArray, i::Int)
    @boundscheck i <= full_length(A) || ThrowBoundsError("Index $i < full length $(full_length(A)).")
    @inbounds A.data[i]
end
@inline function Base.getindex(A::LinearAlgebra.Adjoint{Union{},<: AbstractConstantFixedSizeMatrix{M,N,Vec{W,T}}}, i::Int) where {M,N,W,T}
    @boundscheck i <= full_length(A) || ThrowBoundsError("Index $i < full length $(full_length(A)).")
    @inbounds A.parent.data[i]
end
@generated function Base.getindex(A::AbstractConstantFixedSizeArray{S,T,N,X}, i::Vararg{<:Integer,N}) where {S,T,N,X}
    ex = sub2ind_expr(X.parameters)
    P = N == 1 ? (S.parameters[1])::Int : (X.parameters[2])::Int
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $P : i[d] > size(A,d)) d->ThrowBoundsError() d -> nothing
        end
        @inbounds A.data[1 + $ex]
    end
end
@generated function Base.getindex(A::AbstractConstantFixedSizeArray{S,T,N,X}, i::CartesianIndex{N}) where {S,T,N,X}
    ex = sub2ind_expr(X.parameters)
    P = N == 1 ? (S.parameters[1])::Int : (X.parameters[2])::Int
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $P : i[d] > $SV[d]) d->ThrowBoundsError() d -> nothing
        end
        @inbounds A.data[1 + $ex ]
    end
end
@generated function Base.getindex(A::LinearAlgebra.Adjoint{Union{},<:AbstractConstantFixedSizeMatrix{M,N,Vec{W,T},X}}, i::Int, j::Int) where {M,N,W,T,X}
    R = (X.parameters[2])::Int
    quote
        $(Expr(:meta, :inline))
        @boundscheck if (i > N) || (j > M)
            ThrowBoundsError("(i > N) = ($i > $N) || (j > M) = ($j > $M).")
        end
        @inbounds A.parent.data[ (i-1)*R + j ]
    end
end

@generated function Base.zero(::Type{<:ConstantFixedSizeVector{L,T}}) where {L,T}
    N = calc_padding(L, T)
    :(ConstantFixedSizeVector{$L,$T,$N,$N}($(Expr(:tuple,zeros(N)...))))
end
@inline return_mutable(A::AbstractMutableFixedSizeArray) = A
@inline return_mutable(A::ConstantFixedSizeArray) = MutableFixedSizeArray(A)

struct vStaticPaddedArray{SPA}
    spa::SPA
end
@inline VectorizationBase.vectorizable(A::vStaticPaddedArray) = A
@inline VectorizationBase.vectorizable(A::AbstractConstantFixedSizeArray) = vStaticPaddedArray(A)
@inline VectorizationBase.vectorizable(A::LinearAlgebra.Diagonal{T,<:AbstractConstantFixedSizeArray}) where {T} = vStaticPaddedArray(A.diag)
@inline VectorizationBase.vectorizable(A::LinearAlgebra.Adjoint{T,<:AbstractConstantFixedSizeArray}) where {T} = vStaticPaddedArray(A.parent)
# @inline Base.:+(A::vStaticPaddedArray, i) = vStaticPaddedArray(A.spa, A.offset + i)
# @inline Base.:+(i, A::vStaticPaddedArray) = vStaticPaddedArray(A.spa, A.offset + i)
# @inline Base.:-(A::vStaticPaddedArray, i) = vStaticPaddedArray(A.spa, A.offset - i)

@generated function SIMDPirates.vload(::Type{Vec{W,T}}, A::vStaticPaddedArray, i::Int) where {W,T}
    quote
        $(Expr(:meta, :inline))
        @inbounds $(Expr(:tuple, [:(Core.VecElement{$T}(A.spa.data[i + $w])) for w ∈ 1:W]...))
    end
end
@generated function SIMDPirates.vload(::Type{SVec{W,T}}, A::vStaticPaddedArray, i::Int) where {W,T}
    quote
        $(Expr(:meta, :inline))
        @inbounds SVec($(Expr(:tuple, [:(Core.VecElement{$T}(A.spa.data[i + $w])) for w ∈ 1:W]...)))
    end
end
@generated function SIMDPirates.vload(::Type{Vec{W,T}}, A::vStaticPaddedArray) where {W,T}
    quote
        $(Expr(:meta, :inline))
        @inbounds $(Expr(:tuple, [:(Core.VecElement{$T}(A.spa.data[$w])) for w ∈ 1:W]...))
    end
end
@generated function SIMDPirates.vload(::Type{SVec{W,T}}, A::vStaticPaddedArray) where {W,T}
    quote
        $(Expr(:meta, :inline))
        @inbounds SVec($(Expr(:tuple, [:(Core.VecElement{$T}(A.spa.data[$w])) for w ∈ 1:W]...)))
    end
end
### It is assumed that there is sufficient padding for the load to be unmasked.
### The mask is then applied via "vifelse". Because of the possibility of contaminants cropping up in the padding,
### like NaNs when taking the inverse, applying the mask is necessary for general correctness.
### Perhaps an opt-out method should be provided.
### Or perhaps checks to see if the compiler can figure out it can drop the masking?
### Either way, in a few tests the compiler was smart enough to turn this into a masked load.
@inline function SIMDPirates.vload(::Type{V}, A::vStaticPaddedArray, mask::Union{Vec{N,Bool},<:Unsigned}) where {T,N,V <: Union{Vec{N,T},SVec{N,T}}}
    vifelse(mask, vload(V, A), vbroadcast(V, zero(T)))
end
@inline function SIMDPirates.vload(::Type{V}, A::vStaticPaddedArray, i::Int, mask::Union{Vec{N,Bool},<:Unsigned}) where {T,N,V <: Union{Vec{N,T},SVec{N,T}}}
    vifelse(mask, vload(V, A, i), vbroadcast(V, zero(T)))
end

@inline Base.unsafe_load(A::vStaticPaddedArray) = first(A.spa.data)
@inline Base.unsafe_load(A::vStaticPaddedArray, i::Int) = @inbounds A.spa.data[i]
@inline VectorizationBase.load(A::vStaticPaddedArray) = first(A.spa.data)
@inline Base.getindex(A::vStaticPaddedArray, i::Int) = @inbounds A.spa.data[i]



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
            push!(q.args, :($v = vload($V, ptr + $(W_full*num_unmasked_loads), $(unsafe_trunc(VectorizationBase.mask_type(W_full), (1<<remainder_full-1)) ))))
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
            push!(q.args, :($v = vload($V, ptr + $(W_full*n), $( mask & unsafe_trunc(VectorizationBase.mask_type(W_full), (1<<remainder_full-1)) ))))
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
            push!(q.args, :($v = vload($V, ptr + $(W_full*num_unmasked_loads_per_row + c*nrow), $(unsafe_trunc(VectorizationBase.mask_type(W_full), (1<<remainder-1))))))
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
@generated function SIMDPirates.vload(::Type{A}, ptr::VectorizationBase.Pointer{T}) where {T, S, A <: AbstractConstantFixedSizeArray{S,T}}
    vload_constant_matrix_quote(T, S.parameters, A)
end
@generated function SIMDPirates.vload(::Type{A}, ptr::vStaticPaddedArray{SPA}) where {T, S,S2, A <: AbstractConstantFixedSizeArray{S,T}, SPA <: AbstractConstantFixedSizeArray{S2,T}}
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
            Expr(:curly, :ConstantFixedSizeArray, S, V, N, (SV[1])::Int, simple_vec_prod(SV)),
            outtup)
        )
        q
    end
end
@generated function SIMDPirates.vbroadcast(::Type{Vec{W,T}}, A::ConstantFixedSizeArray{S,T,N,P,L}) where {S,T,N,P,L,W}
    broadcast_array_quote(S, P, W, T, Val(N))
end

@generated function Base.vcat(a::AbstractConstantFixedSizeVector{M,T}, b::AbstractConstantFixedSizeVector{N,T}) where {M,N,T}
    MpN = M + N
    L = calc_padding(MpN, T)
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
    :(@inbounds ConstantFixedSizeVector{$MpN,$T}($outtup))
end
Base.vcat(a::AbstractConstantFixedSizeVector, b::AbstractConstantFixedSizeVector, c::AbstractFixedSizeVector...) = vcat(vcat(a,b), c...)

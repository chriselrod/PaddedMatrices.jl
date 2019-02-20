function init_mutable_fs_padded_array_quote(S, T)
    SV = S.parameters
    N = length(SV)

    nrow = SV[1]
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    TwoN = 2nrow
    if W < TwoN
        rem = (nrow & (W-1))
        padded_rows = rem == 0 ? nrow : nrow + W - rem
        L = padded_rows
        for n ∈ 2:N
            L *= SV[n]
        end
        return quote
            out = MStaticPaddedArray{$S,$T,$N,$padded_rows,$L}(undef)
            @nloops $(N-1) i j -> 1:S[j+1] begin
                @inbounds for i_0 ∈ $(padded_rows+1-W):$padded_rows
                    ( @nref $N out n -> i_{n-1} ) = zero($T)
                end
            end
            out
        end
    end
    while W >= TwoN
        W >>= 1
    end
    rem = (nrow & (W-1))
    padded_rows = rem == 0 ? nrow : nrow + W - rem
    L = padded_rows
    for n ∈ 2:N
        L *= SV[n]
    end
    quote
        out = MStaticPaddedArray{$S,$T,$N,$padded_rows,$L}(undef)
        for l ∈ 1:$L
            out[l] = zero($T)
        end
        out
    end
end


"""
Parameters are
S, the size tuple
T, the element type
N, the number of axis
R, the actual number of rows, where the excess R > S[1] are zero.
L, number of elements (including buffer zeros).
"""
mutable struct MutableFixedSizePaddedArray{S,T,N,P,L} <: AbstractMutableFixedSizePaddedArray{S,T,N,P,L}
    data::NTuple{L,T}
    function MutableFixedSizePaddedArray{S,T,N,R,L}(::UndefInitializer) where {S,T,N,R,L}
        new()
    end
    @generated function MutableFixedSizePaddedArray{S,T}(::UndefInitializer) where {S,T}
        init_mpadded_array_quote(S, T)
    end
    @generated function MutableFixedSizePaddedArray{S,T,N}(::UndefInitializer) where {S,T,N}
        init_mpadded_array_quote(S, T)
    end
end
const MutableFixedSizePaddedVector{S,T,P,L} = MutableFixedSizePaddedArray{Tuple{S},T,1,P,L}
const MutableFixedSizePaddedMatrix{M,N,T,P,L} = MutableFixedSizePaddedArray{Tuple{M,N},T,2,P,L}

@generated function MutableFixedSizePaddedArray(::UndefInitializer, ::Val{S}, ::Type{T1}=Float64) where {S,T1}
    SD = Tuple{S...}
    init_mutable_fs_padded_array_quote(SD, T)
end

"""
This is only meant to make recursive algorithms easiesr to implement.
Wraps a pointer, while passing info on the size of the block and stride.
"""
struct PtrArray{S,T,N,R,L} <: AbstractMutableFixedSizePaddedArray{S,T,N,R,L}
    ptr::Ptr{T}
    @generated function PtrArray{S,T,N,R}(ptr::Ptr{T}) where {S,T,N,R}
        L = R
        for i ∈ 2:N
            L *= S.parameters[i]
        end
        :(PtrArray{$S,$T,$N,$R,$L}(ptr))
    end
    @generated function PtrArray{S,T}(ptr::Ptr{T}) where {S,T}
        SV = S.parameters
        N = length(SV)
        R, L = calculate_L_from_size(SV)
        :(PtrArray{$S,$T,$N,$R,$L}(ptr))
    end
end
const PtrVector{N,T,R,L} = PtrArray{Tuple{N},T,1,R,L} # R and L will always be the same...
const PtrMatrix{M,N,T,R,L} = PtrArray{Tuple{M,N},T,2,R,L}


@inline Base.pointer(ptr::PtrMatrix) = ptr.ptr
@inline Base.pointer(A::AbstractMutableFixedSizePaddedArray{S,T}) where {S,T} = pointer_from_objref(Base.unsafe_convert(Ptr{T}, A))

@inline VectorizationBase.vectorizable(A::AbstractMutableFixedSizePaddedArray) = pointer(A)


@generated function Base.setindex!(A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}, v, i::CartesianIndex{N}) where {S,T,N,R,L}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->( d == 1 ? i[d] > $R : i[d] > $dims[d]) d -> ThrowBoundsError("Dimension $d out of bounds") d -> nothing
        end
        # T = eltype(A)
        unsafe_store!(pointer(A), convert($T,v), $ex)
        v
    end
end
@generated function Base.setindex!(A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}, v, i::Vararg{Integer,M}) where {S,T,N,R,L,M}
    dims = ntuple(j -> S.parameters[j], Val(N))
    if M == 1
        ex = :(@inbounds i[1])
    else
        ex = sub2ind_expr(dims, R)
    end
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->( d == 1 ? i[d] > $R : i[d] > $dims[d]) d-> ThrowBoundsError("Dimension $d out of bounds $(i[d]) > $R") d -> nothing
        end
        unsafe_store!(pointer(A), convert($T,v), $ex)
        v
    end
end
# AbstractMutableFixedSizePaddedArray


@inline Base.@propagate_inbounds Base.getindex(A::AbstractMutableFixedSizePaddedArray, i...) = A.data[i...]

@inline function Base.getindex(A::AbstractMutableFixedSizePaddedArray{S,T,1,L,L}, i::Int) where {S,T,L}
    @boundscheck i <= L || ThrowBoundsError("Index $i > full length $L.")
    unsafe_load(pointer(A), i)
end
@inline function Base.getindex(A::AbstractMutableFixedSizePaddedArray, i::Int)
    @boundscheck i <= full_length(A) || ThrowBoundsError("Index $i > full length $(full_length(A)).")
    T = eltype(A)
    unsafe_load(pointer(A), i)
end

"""
Returns zero based index. Don't forget to add one when using with arrays instead of pointers.
"""
function sub2ind_expr(S, R)
    N = length(S)
    N == 1 && return :(i[1])
    ex = :(i[$N] - 1)
    for i ∈ (N - 1):-1:2
        ex = :(i[$i] - 1 + $(S[i]) * $ex)
    end
    :(i[1] + $R * $ex)
end

@generated function Base.getindex(A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}, i::Vararg{Int,N}) where {S,T,N,R,L}
    # dims = ntuple(j -> S.parameters[j], Val(N))
    sv = S.parameters
    ex = sub2ind_expr(sv, R)
    quote
        $(Expr(:meta, :inline))
        # @show i, S
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > $sv[d]) d->ThrowBoundsError() d -> nothing
        end
        # T = eltype(A)
        # unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A) + $(sizeof(T))*($ex) ))
        # A.data[$ex+1]
        unsafe_load(pointer(A), $ex)
    end
end
@generated function Base.getindex(A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}, i::CartesianIndex{N}) where {S,T,N,R,L}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        # @show i, S
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > $dims[d]) d->ThrowBoundsError() d -> nothing
        end
        unsafe_load(pointer(A), $ex)
    end
end

@inline Base.unsafe_convert(::Type{Ptr{T}}, A::SizedSIMDArray) where {T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@generated function strides(A::AbstractFixedSizePaddedArray{S,T,N,R,L}) where {S,T,N,R,L}
    SV = S.parameters
    N = length(SV)
    N == 1 && return (1,)
    last = R
    q = Expr(:tuple, 1, last)
    for n ∈ 3:N
        last *= SV[n-1]
        push!(q.args, last)
    end
    q
end

to_tuple(S) = tuple(S.parameters...)
@generated Base.size(::AbstractFixedSizePaddedArray{S}) where {S} = to_tuple(S)


# Do we want this, or L?
@generated Base.length(::AbstractFixedSizePaddedArray{S}) where {S} = prod(to_tuple(S))

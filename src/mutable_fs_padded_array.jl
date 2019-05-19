function calc_padding(nrow, T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    TwoN = 2nrow

    while W >= TwoN
        W >>= 1
    end
    Wm1 = W - 1
    # rem = nrow & Wm1
    (nrow + Wm1) & ~Wm1
end
calc_NPL(S::DataType, T) = calc_NPL(S.parameters, T)

function calc_NPL(SV, T)
    # N, padded_rows, L = calc_NPL(S, T)
    # SV = S.parameters
    N = length(SV)

    nrow = SV[1]
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    TwoN = 2nrow

    while W >= TwoN
        W >>= 1
    end
    Wm1 = W - 1
    # rem = nrow & Wm1
    padded_rows = (nrow + Wm1) & ~Wm1
    L = padded_rows
    for n ∈ 2:N
        L *= SV[n]
    end
    N, padded_rows, L
end

function init_mutable_fs_padded_array_quote(S, T)
    SV = S.parameters
    N = length(SV)

    nrow = SV[1]
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    TwoN = 2nrow
    if W < TwoN
        Wm1 = W - 1
        rem = nrow & Wm1
        padded_rows = (nrow + Wm1) & ~Wm1
        L = padded_rows
        for n ∈ 2:N
            L *= SV[n]
        end
        q = quote
            $(Expr(:meta,:inline))
            out = MutableFixedSizePaddedArray{$S,$T,$N,$padded_rows,$L}(undef)
        end
        if rem > 0
            push!(q.args, quote
                @nloops $(N-1) i j -> 1:$SV[j+1] begin
                    @inbounds for i_0 ∈ $(padded_rows+1-W):$padded_rows
                        ( @nref $N out n -> i_{n-1} ) = zero($T)
                    end
                end
                out
            end)
        end
        return q
    end
    while W >= TwoN
        W >>= 1
    end
    Wm1 = W - 1
    # rem = nrow & Wm1
    padded_rows = (nrow + Wm1) & ~Wm1
    L = padded_rows
    for n ∈ 2:N
        L *= SV[n]
    end
    quote
        $(Expr(:meta,:inline))
        out = MutableFixedSizePaddedArray{$S,$T,$N,$padded_rows,$L}(undef)
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
    @inline function MutableFixedSizePaddedArray{S,T,N,R,L}(::UndefInitializer) where {S,T,N,R,L}
        new()
    end
    @inline function MutableFixedSizePaddedArray{S,T,N,R,L}(data::NTuple{L,T}) where {S,T,N,R,L}
        new(data)
    end
end
@generated function MutableFixedSizePaddedArray{S,T}(::UndefInitializer) where {S,T}
    init_mutable_fs_padded_array_quote(S, T)
end
@generated function MutableFixedSizePaddedArray{S,T,N}(::UndefInitializer) where {S,T,N}
    init_mutable_fs_padded_array_quote(S, T)
end
const MutableFixedSizePaddedVector{M,T,P,L} = MutableFixedSizePaddedArray{Tuple{M},T,1,P,L}
const MutableFixedSizePaddedMatrix{M,N,T,P,L} = MutableFixedSizePaddedArray{Tuple{M,N},T,2,P,L}

@inline MutableFixedSizePaddedVector(A::AbstractFixedSizePaddedArray{S,T,1,P,L}) where {S,T,P,L} = MutableFixedSizePaddedArray{S,T,1,P,L}(A.data)
@inline MutableFixedSizePaddedMatrix(A::AbstractFixedSizePaddedArray{S,T,2,P,L}) where {S,T,P,L} = MutableFixedSizePaddedArray{S,T,2,P,L}(A.data)
@inline MutableFixedSizePaddedArray(A::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L}= MutableFixedSizePaddedArray{S,T,N,P,L}(A.data)

@generated function MutableFixedSizePaddedArray(::UndefInitializer, ::Val{S}, ::Type{T1}=Float64) where {S,T1}
    SD = Tuple{S...}
    init_mutable_fs_padded_array_quote(SD, T1)
end
@generated function MutableFixedSizePaddedArray{S,T,N,P}(::UndefInitializer) where {S,T,N,P}
    L = P
    SV = S.parameters
    for n in 2:N
        L *= SV[n]
    end
    :(MutableFixedSizePaddedArray{$S,$T,$N,$P,$L}(undef))
end

@generated function Base.zero(::Type{<:MutableFixedSizePaddedArray{S,T}}) where {S,T}
    N,P,L = calc_NPL(S, T)
    quote
        $(Expr(:meta,:inline))
        ma = MutableFixedSizePaddedArray{$S,$T,$N,$P,$L}(undef)
        @inbounds @simd ivdep for l ∈ 1:$L
            ma[l] = zero($T)
        end
        ma
    end
end
@generated function Base.zero(::Type{<:MutableFixedSizePaddedArray{S}}) where {S}
    T = Float64
    N,P,L = calc_NPL(S, T)
    quote
        $(Expr(:meta,:inline))
        ma = MutableFixedSizePaddedArray{$S,$T,$N,$P,$L}(undef)
        @inbounds @simd ivdep for l ∈ 1:$L
            ma[l] = zero($T)
        end
        ma
    end
end
@generated function Base.zero(::Type{<:MutableFixedSizePaddedArray{S,Vec{W,T}}}) where {S,W,T}
    SV = S.parameters
    P = SV[1]
    L = prod(SV)
    N = length(SV)
    quote
        $(Expr(:meta,:inline))
        ma = MutableFixedSizePaddedArray{$S,Vec{$W,$T},$N,$P,$L}(undef)
        @inbounds for l ∈ 1:$L
            ma[l] = SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T))
        end
        ma
    end
end
function zero!(A::AbstractMutableFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L}
    @inbounds for i ∈ 1:L
        A[i] = zero(T)
    end
end
function zero!(A::AbstractMutableFixedSizePaddedArray{S,Vec{W,T},N,P,L}) where {S,W,T,N,P,L}
    @inbounds for i ∈ 1:L
        A[i] = SIMDPirates.vbroadcast(Vec{W,T}, zero(T))
    end
end

function Base.copy(A::AbstractMutableFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L}
    B = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @inbounds for l ∈ 1:L
        B[l] = A[l]
    end
    B
end
function Base.copyto!(B::AbstractMutableFixedSizePaddedArray{S,T,N,P,L}, A::AbstractMutableFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L}
    @inbounds for l ∈ 1:L
        B[l] = A[l]
    end
    B
end
Base.similar(A::AbstractMutableFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L} = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
Base.similar(A::AbstractMutableFixedSizePaddedArray{S,T1,N,P,L},::Type{T2}) where {S,T1,T2,N,P,L} = MutableFixedSizePaddedArray{S,T2,N}(undef)

function MutableFixedSizePaddedArray(A::AbstractArray{T,N}) where {T,N}
    mA = MutableFixedSizePaddedArray{Tuple{size(A)...},Float64}(undef)
    mA .= A
    mA
end
function MutableFixedSizePaddedVector(A::AbstractVector{T}) where {T}
    mA = MutableFixedSizePaddedVector{length(A),Float64}(undef)
    mA .= A
    mA
end
function MutableFixedSizePaddedMatrix(A::AbstractMatrix{T}) where {T}
    M,N = size(A)
    mA = MutableFixedSizePaddedMatrix{M,N,Float64}(undef)
    mA .= A
    mA
end

mutable_similar(A::AbstractArray) = similar(A)
mutable_similar(A::AbstractFixedSizePaddedArray{S,T,N,R,L}) where {S,T,N,R,L} = MutableFixedSizePaddedArray{S,T,N,R,L}(undef)
mutable_similar(A::AbstractArray, T) = similar(A, T)
mutable_similar(A::AbstractFixedSizePaddedArray{S,T1,N,R,L}, ::Type{T2}) where {S,T1,T2,N,R,L} = MutableFixedSizePaddedArray{S,T2,R,L}(undef)
function Base.fill!(A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}, v::T) where {S,T,N,R,L}
    @inbounds for l ∈ 1:L
        A[l] = v
    end
    A
end

"""
This is only meant to make recursive algorithms easiesr to implement.
Wraps a pointer, while passing info on the size of the block and stride.
"""
struct PtrArray{S,T,N,R,L} <: AbstractMutableFixedSizePaddedArray{S,T,N,R,L}
    ptr::Ptr{T}
end
@generated function PtrArray{S,T,N,R}(ptr::Ptr{T}) where {S,T,N,R}
    L = R
    for i ∈ 2:N
        L *= S.parameters[i]
    end
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$R,$L}(ptr)
    end
end
@generated function PtrArray{S,T}(ptr::Ptr{T}) where {S,T}
    N, P, L = calc_NPL(S, T)
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$P,$L}(ptr)
    end
end
@generated function PtrArray{S}(ptr::Ptr{T}) where {S,T}
    N, P, L = calc_NPL(S, T)
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$P,$L}(ptr)
    end
end
const PtrVector{N,T,R,L} = PtrArray{Tuple{N},T,1,R,L} # R and L will always be the same...
const PtrMatrix{M,N,T,R,L} = PtrArray{Tuple{M,N},T,2,R,L}


@inline Base.pointer(A::PtrArray) = A.ptr
@inline Base.pointer(A::AbstractMutableFixedSizePaddedArray{S,T}) where {S,T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@inline Base.pointer(A::AbstractMutableFixedSizePaddedArray{S,NTuple{W,Core.VecElement{T}}}) where {S,T,W} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@inline Base.pointer(A::PtrArray{S,NTuple{W,Core.VecElement{T}}}) where {S,T,W} = Base.unsafe_convert(Ptr{T}, A.ptr)


@inline VectorizationBase.vectorizable(A::AbstractMutableFixedSizePaddedArray) = VectorizationBase.vpointer(pointer(A))
@inline VectorizationBase.vectorizable(A::LinearAlgebra.Diagonal{T,<:AbstractMutableFixedSizePaddedArray}) where {T} = VectorizationBase.vpointer(pointer(A.diag))
@inline VectorizationBase.vectorizable(A::LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizePaddedArray}) where {T} = VectorizationBase.vpointer(pointer(A.parent))



@generated function Base.setindex!(A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}, v, i::CartesianIndex{N}) where {S,T,N,R,L}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->( d == 1 ? i[d] > $R : i[d] > $dims[d]) d -> ThrowBoundsError("Dimension $d out of bounds") d -> nothing
        end
        # T = eltype(A)
        VectorizationBase.store!(pointer(A) + sizeof(T) * ($ex - 1), convert($T,v))
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
        VectorizationBase.store!(pointer(A) + sizeof(T) * ($ex - 1), convert($T,v))
        v
    end
end
@generated function Base.setindex!(A::AbstractMutableFixedSizePaddedArray{S,NTuple{W,Core.VecElement{T}},N,R,L},
                                    v::NTuple{W,Core.VecElement{T}}, i::CartesianIndex{N}) where {S,T,N,R,L,W}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->( d == 1 ? i[d] > $R : i[d] > $dims[d]) d -> ThrowBoundsError("Dimension $d out of bounds") d -> nothing
        end
        # T = eltype(A)
        SIMDPirates.vstore!(pointer(A) + sizeof(NTuple{W,Core.VecElement{T}}) * ($ex - 1), v)
        v
    end
end
@generated function Base.setindex!(A::AbstractMutableFixedSizePaddedArray{S,NTuple{W,Core.VecElement{T}},N,R,L},
                                    v::NTuple{W,Core.VecElement{T}}, i::Vararg{Integer,M}) where {S,T,N,R,L,M,W}
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
        SIMDPirates.vstore!(pointer(A) + sizeof(NTuple{W,Core.VecElement{T}}) * ($ex - 1), v)
        v
    end
end
# AbstractMutableFixedSizePaddedArray


# @inline Base.@propagate_inbounds Base.getindex(A::AbstractMutableFixedSizePaddedArray, i...) = A.data[i...]

@inline function Base.getindex(A::AbstractMutableFixedSizePaddedArray{S,T,1,L,L}, i::Int) where {S,T,L}
    @boundscheck i <= L || ThrowBoundsError("Index $i > full length $L.")
    VectorizationBase.load(pointer(A) + sizeof(T) * (i - 1))
end
@inline function Base.getindex(A::AbstractMutableFixedSizePaddedArray, i::Int)
    @boundscheck i <= full_length(A) || ThrowBoundsError("Index $i > full length $(full_length(A)).")
    T = eltype(A)
    VectorizationBase.load(pointer(A) + sizeof(T) * (i - 1))
end
@inline function Base.getindex(A::AbstractMutableFixedSizePaddedArray{S,Vec{W,T},1}, i::Int) where {S,T,L,W}
    @boundscheck i <= full_length(A) || ThrowBoundsError("Index $i > full length $L.")
    SIMDPirates.vload(Vec{W,T}, pointer(A) + sizeof(Vec{W,T}) * (i - 1))
end
@inline function Base.getindex(A::AbstractMutableFixedSizePaddedArray{S,Vec{W,T},N,R,L}, i::Int) where {S,T,L,W,N,R}
    @boundscheck i <= L || ThrowBoundsError("Index $i > full length $L.")
    SIMDPirates.vload(Vec{W,T}, pointer(A) + sizeof(Vec{W,T}) * (i - 1))
end
@inline function Base.getindex(A::LinearAlgebra.Adjoint{Union{},<: AbstractMutableFixedSizePaddedArray{S,Vec{W,T},N,R,L}}, i::Int) where {S,T,L,W,N,R}
    @boundscheck i <= L || ThrowBoundsError("Index $i > full length $L.")
    SIMDPirates.vload(Vec{W,T}, pointer(A.parent) + sizeof(Vec{W,T}) * (i - 1))
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
        VectorizationBase.load(pointer(A) + $(sizeof(T)) * ($ex - 1))
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
        VectorizationBase.load(pointer(A) + $(sizeof(T)) * ($ex - 1))
    end
end
@generated function Base.getindex(A::AbstractMutableFixedSizePaddedArray{S,Vec{W,T},N,R,L}, i::Vararg{Int,N}) where {S,T,N,R,L,W}
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
        SIMDPirates.vload(Vec{W,T}, pointer(A) + $(sizeof(Vec{W,T})) * ($ex - 1))
    end
end
@generated function Base.getindex(A::AbstractMutableFixedSizePaddedArray{S,Vec{W,T},N,R,L}, i::CartesianIndex{N}) where {S,T,N,R,L,W}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        # @show i, S
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > $dims[d]) d->ThrowBoundsError() d -> nothing
        end
        SIMDPirates.vload(Vec{W,T}, pointer(A) + $(sizeof(Vec{W,T})) * ($ex - 1))
    end
end
@generated function Base.getindex(A::LinearAlgebra.Adjoint{Union{},<: AbstractMutableFixedSizePaddedMatrix{M,N,Vec{W,T},R,L}}, i::Int, j::Int) where {M,N,W,T,R,L}
    # dims = ntuple(j -> S.parameters[j], Val(N))
    quote
        $(Expr(:meta, :inline))
        # @show i, S
        @boundscheck begin
            if (j > M) || (i > N)
                ThrowBoundsError("At least one of: $j > $M or $i > $N.")
            end
        end
        # T = eltype(A)
        # unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A) + $(sizeof(T))*($ex) ))
        # A.data[$ex+1]
        SIMDPirates.vload(Vec{W,T}, pointer(A.parent) + $(sizeof(Vec{W,T})) * ( (i-1)*$R + j-1 ))
    end
end

@inline Base.unsafe_convert(::Type{Ptr{T}}, A::AbstractMutableFixedSizePaddedArray) where {T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@generated function Base.strides(A::AbstractFixedSizePaddedArray{S,T,N,R,L}) where {S,T,N,R,L}
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
@inline function Base.stride(A::AbstractFixedSizePaddedArray{S,T,N,R}, n::Integer) where {S,T,N,R}
    n == 1 && return 1
    n == 2 && return R
    S.parameters[n-1]
end

to_tuple(S) = tuple(S.parameters...)
@generated Base.size(::AbstractFixedSizePaddedArray{S}) where {S} = to_tuple(S)


# Do we want this, or L?
@generated Base.length(::AbstractFixedSizePaddedArray{S}) where {S} = prod(to_tuple(S))


staticrangelength(::Type{Static{R}}) where R = 1 + last(R) - first(R)
"""
Note that the stride will currently only be correct when N <= 2.
Perhaps R should be made into a stride-tuple?
"""
@generated function sview(A::AbstractMutableFixedSizePaddedArray{S,T,N,P,L}, inds...) where {S,T,N,P,L}
    @assert length(inds) == N
    SV = S.parameters
    s2 = Vector{Int}(undef, N)
    local offset
    for n ∈ 1:N
        # @show inds[n], n
        if inds[n] == Colon
            s2[n] = SV[n]
            if n == 1
                offset = 0
            end
        else
            @assert inds[n] <: Static
            s2[n] = staticrangelength(inds[n])
            if n == 1
                offset = sizeof(T) * (first(static_type(inds[n]))-1)
            end
        end
    end
    S2 = Tuple{s2...}
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S2,$T,$N,$P,$(prod(s2))}(pointer(A) + $offset)
    end
end


macro sview(expr)
    @assert expr.head == :ref
    q = Expr(:call, :(PaddedMatrices.sview), expr.args[1])
    for n ∈ 2:length(expr.args)
        original_ind = expr.args[n]
        if original_ind isa Expr && original_ind.args[1] == :(:)
            new_ind = :(PaddedMatrices.Static{$original_ind}())
        else
            new_ind = original_ind
        end
        push!(q.args, new_ind)
    end
    esc(q)
end


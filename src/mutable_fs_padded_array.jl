
@noinline function calc_NPL(SV::Core.SimpleVector, T, align_stride::Bool = true, pad_to_align_length::Bool = false)#true)
    nrow = (SV[1])::Int
    N = length(SV)
    if align_stride
        padded_rows = calc_padding(nrow, T)
    else
        W, Wshift = VectorizationBase.pick_vector_width_shift(T)
        TwoN = 2nrow

        while W >= TwoN
            W >>>= 1
        end
        Wm1 = W - 1
        # rem = nrow & Wm1
        padded_rows = (nrow + Wm1) & ~Wm1
    end
    L = padded_rows
    for n ∈ 2:N
        L *= (SV[n])::Int
    end
    LA = VectorizationBase.align(L,T)
    if pad_to_align_length
        LA += VectorizationBase.pick_vector_width(T)
    end
    N, padded_rows, LA
end

@noinline function init_mutable_fs_padded_array_quote(SV::Core.SimpleVector, T)
    N = length(SV)
    nrow = (SV[1])::Int
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    TwoN = 2nrow
    if W < TwoN
        Wm1 = W - 1
        rem = nrow & Wm1
        padded_rows = (nrow + Wm1) & ~Wm1
        L = padded_rows
        for n ∈ 2:N
            L *= (SV[n])::Int
        end
        q = quote
            $(Expr(:meta,:inline))
            out = MutableFixedSizePaddedArray{Tuple{$(SV...)},$T,$N,$padded_rows,$L}(undef)
        end
        return q
    end
    while W >= TwoN
        W >>>= 1
    end
    Wm1 = W - 1
    # rem = nrow & Wm1
    padded_rows = (nrow + Wm1) & ~Wm1
    L = padded_rows
    for n ∈ 2:N
        L *= (SV[n])::Int
    end
    quote
        $(Expr(:meta,:inline))
        MutableFixedSizePaddedArray{Tuple{$(SV...)},$T,$N,$padded_rows,$L}(undef)
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
    init_mutable_fs_padded_array_quote(S.parameters, T)
end
@generated function MutableFixedSizePaddedArray{S,T,N}(::UndefInitializer) where {S,T,N}
    init_mutable_fs_padded_array_quote(S.parameters, T)
end
const MutableFixedSizePaddedVector{M,T,L} = MutableFixedSizePaddedArray{Tuple{M},T,1,L,L}
const MutableFixedSizePaddedMatrix{M,N,T,P,L} = MutableFixedSizePaddedArray{Tuple{M,N},T,2,P,L}
@generated function MutableFixedSizePaddedVector{S,T}(::UndefInitializer) where {S,T}
    L = calc_padding(S,T)
    :(MutableFixedSizePaddedVector{$S,$T,$L}(undef))
end

@inline MutableFixedSizePaddedVector(A::AbstractFixedSizePaddedArray{S,T,1,P,L}) where {S,T,P,L} = MutableFixedSizePaddedArray{S,T,1,P,L}(A.data)
@inline MutableFixedSizePaddedMatrix(A::AbstractFixedSizePaddedArray{S,T,2,P,L}) where {S,T,P,L} = MutableFixedSizePaddedArray{S,T,2,P,L}(A.data)
@inline MutableFixedSizePaddedArray(A::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L}= MutableFixedSizePaddedArray{S,T,N,P,L}(A.data)



@generated function MutableFixedSizePaddedArray(::UndefInitializer, ::Val{S}, ::Type{T1}=Float64) where {S,T1}
    SD = Tuple{S...}
    init_mutable_fs_padded_array_quote(SD.parameters, T1)
end
@generated function MutableFixedSizePaddedArray{S,T,N,P}(::UndefInitializer) where {S,T,N,P}
    L = P
    SV = S.parameters
    for n in 2:N
        L *= (SV[n])::Int
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
    P = (SV[1])::Int
    N = length(SV)
    L = P
    for n in 2:N
        L *= (SV[n])::Int
    end
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

@generated function Base.copy(A::AbstractMutableFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L}
    quote
        B = MutableFixedSizePaddedArray{$S,$T,$N,$P,$L}(undef)
        @vvectorize $T for l ∈ 1:$L
            B[l] = A[l]
        end
        B
    end
end
@generated function Base.copyto!(B::AbstractMutableFixedSizePaddedArray{S,T,N,P,L}, A::AbstractMutableFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L}
    quote
        @vvectorize $T for l ∈ 1:$L
            B[l] = A[l]
        end
        B
    end
end
Base.similar(A::AbstractMutableFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L} = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
Base.similar(A::AbstractMutableFixedSizePaddedArray{S,T1,N,P,L},::Type{T2}) where {S,T1,T2,N,P,L} = MutableFixedSizePaddedArray{S,T2,N}(undef)

function MutableFixedSizePaddedArray(A::AbstractArray{T,N}) where {T,N}
    mA = MutableFixedSizePaddedArray{Tuple{size(A)...},T}(undef)
    mA .= A
    mA
end
function MutableFixedSizePaddedVector(A::AbstractVector{T}) where {T}
    mA = MutableFixedSizePaddedVector{length(A),T}(undef)
    mA .= A
    mA
end
function MutableFixedSizePaddedMatrix(A::AbstractMatrix{T}) where {T}
    M,N = size(A)
    mA = MutableFixedSizePaddedMatrix{M,N,T}(undef)
    mA .= A
    mA
end
function MutableFixedSizePaddedArray{S}(A::AbstractArray{T,N}) where {S,T,N}
    mA = MutableFixedSizePaddedArray{S,T}(undef)
    mA .= A
    mA
end
function MutableFixedSizePaddedVector{L}(A::AbstractVector{T}) where {L,T}
    mA = MutableFixedSizePaddedVector{L,T}(undef)
    mA .= A
    mA
end
function MutableFixedSizePaddedMatrix{M,N}(A::AbstractMatrix{T}) where {M,N,T}
    mA = MutableFixedSizePaddedMatrix{M,N,T}(undef)
    mA .= A
    mA
end
function MutableFixedSizePaddedArray{S,T1}(A::AbstractArray{T2,N}) where {S,T1,T2,N}
    mA = MutableFixedSizePaddedArray{S,T1}(undef)
    mA .= A
    mA
end
function MutableFixedSizePaddedVector{L,T1}(A::AbstractVector{T2}) where {L,T1,T2}
    mA = MutableFixedSizePaddedVector{L,T1}(undef)
    mA .= A
    mA
end
function MutableFixedSizePaddedMatrix{M,N,T1}(A::AbstractMatrix{T2}) where {M,N,T1,T2}
    mA = MutableFixedSizePaddedMatrix{M,N,T1}(undef)
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
struct PtrArray{S,T,N,R,L,P} <: AbstractMutableFixedSizePaddedArray{S,T,N,R,L}
    ptr::Ptr{T}
end
@generated function PtrArray{S,T,N,R,L}(ptr::Ptr{T},::Val{P}=Val{true}()) where {S,T,N,R,P,L}
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
    end
end
@generated function PtrArray{S,T,N,R}(ptr::Ptr{T},::Val{P}=Val{true}()) where {S,T,N,R,P}
    L = R
    SV = S.parameters
    for n ∈ 2:N
        L *= (SV[n])::Int
    end
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
    end
end
@generated function PtrArray{S,T,N}(ptr::Ptr{T},::Val{P}=Val{true}()) where {S,T,N,P}
    R = calc_padding((S.parameters[1])::Int, T)
    L = R
    SV = S.parameters
    for n ∈ 2:N
        L *= (SV[n])::Int
    end
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
    end
end
@generated function PtrArray{S,T}(ptr::Ptr{T}, ::Val{P} = Val{true}()) where {S,T,P}
    N, R, L = calc_NPL(S, T,true,false)
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
    end
end
@generated function PtrArray{S}(ptr::Ptr{T}) where {S,T}
    N, P, L = calc_NPL(S, T,true,false)
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$P,$L,true}(ptr)
    end
end
@generated function PtrArray{S}(ptr::Ptr{T}, ::Val{P}) where {S,T,P}
    N,R,L = calc_NPL(S,T,true,false)
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
    end
end
@generated function PtrArray{S}(sp::StackPointer, ::Val{P} = Val{true}()) where {S,P}
    N,R,L = calc_NPL(S,Float64,true,false)
    quote
        $(Expr(:meta,:inline))
        sp + $(VectorizationBase.align(8L)), PtrArray{$S,Float64,$N,$R,$L,$P}(pointer(sp, Float64))
    end
end
@generated function PtrArray{S,T}(sp::StackPointer, ::Val{P} = Val{true}()) where {S,T,P}
    N,R,L = calc_NPL(S,T,true,false)
    quote
        $(Expr(:meta,:inline))
        sp + $(VectorizationBase.align(sizeof(T)*L)), PtrArray{$S,$T,$N,$R,$L,$P}(pointer(sp, $T))
    end
end
@generated function PtrArray{S,T,N}(sp::StackPointer, ::Val{P} = Val{true}()) where {S,T,N,P}
    R = calc_padding((S.parameters[1])::Int, T)
    L = R
    SV = S.parameters
    for n ∈ 2:N
        L *= (SV[n])::Int
    end
    quote
        $(Expr(:meta,:inline))
        sp + $(VectorizationBase.align(sizeof(T)*L)), PtrArray{$S,$T,$N,$R,$L,$P}(pointer(sp, $T))
    end
end
@generated function PtrArray{S,T,N,R}(sp::StackPointer, ::Val{P} = Val{true}()) where {S,T,N,R,P}
    L = R
    SV = S.parameters
    for n ∈ 2:N
        L *= (SV[n])::Int
    end
    quote
        $(Expr(:meta,:inline))
        ptr = Base.unsafe_convert(Ptr{$T}, sp.ptr)
        sp + $(VectorizationBase.align(sizeof(T)*L)), PtrArray{$S,$T,$N,$R,$L,$P}(pointer(sp, $T))
#        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
    end
end
@generated function PtrArray{S,T,N,R,L}(sp::StackPointer, ::Val{P} = Val{true}()) where {S,T,N,R,P,L}
    quote
        $(Expr(:meta,:inline))
        ptr = Base.unsafe_convert(Ptr{$T}, sp.ptr)
        sp + $(VectorizationBase.align(sizeof(T)*L)), PtrArray{$S,$T,$N,$R,$L,$P}(pointer(sp, $T))
#        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
    end
end
@generated function PtrArray(A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}) where {S,T,N,R,L}
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$R,$L,true}(pointer(A))
    end
end

const PtrVector{N,T,L,P} = PtrArray{Tuple{N},T,1,L,L,P} # R and L will always be the same...
const PtrMatrix{M,N,T,R,L,P} = PtrArray{Tuple{M,N},T,2,R,L,P}

@generated function PtrVector{P,T}(a) where {P,T}
    L = calc_padding(P, T)
    quote
        $(Expr(:meta,:inline))
        PtrArray{Tuple{$P},$T,1,$L,$L}(a)
    end
end
# Now defining these, because any use of PtrArray is going to already be using some pointer.
# These will therefore be fully manual.
#@support_stack_pointer PaddedMatrices PtrVector;
#@support_stack_pointer PaddedMatrices PtrMatrix;
#@support_stack_pointer PaddedMatrices PtrArray;



@inline Base.pointer(A::PtrArray) = A.ptr
@inline Base.pointer(A::AbstractMutableFixedSizePaddedArray{S,T}) where {S,T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@inline Base.pointer(A::AbstractMutableFixedSizePaddedArray{S,NTuple{W,Core.VecElement{T}}}) where {S,T,W} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@inline Base.pointer(A::PtrArray{S,NTuple{W,Core.VecElement{T}}}) where {S,T,W} = Base.unsafe_convert(Ptr{T}, A.ptr)


@inline VectorizationBase.vectorizable(A::AbstractMutableFixedSizePaddedArray) = VectorizationBase.Pointer(pointer(A))
@inline VectorizationBase.vectorizable(A::LinearAlgebra.Diagonal{T,<:AbstractMutableFixedSizePaddedArray}) where {T} = VectorizationBase.Pointer(pointer(A.diag))
@inline VectorizationBase.vectorizable(A::LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizePaddedArray}) where {T} = VectorizationBase.Pointer(pointer(A.parent))


@inline Base.similar(sp::StackPointer, A::AbstractMutableFixedSizePaddedArray{S,T,N,R}) where {S,T,N,R} = PtrArray{S,T,N,R}(sp, Val{true}())
@inline function Base.copy(sp::StackPointer, A::AbstractMutableFixedSizePaddedArray{S,T,N,R}) where {S,T,N,R}
    sp, B = PtrArray{S,T,N,R}(sp, Val{true}())
    sp, copyto!(B, A)
end


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
        bound_check = :(i[1] > $L && ThrowBoundsError("index == $(i[1]) > $L == length of array"))
        ex = :(@inbounds i[1])
    else
        ex = sub2ind_expr(dims, R)
        bound_check = :(Base.Cartesian.@nif $(N+1) d->( d == 1 ? i[d] > $R : i[d] > $dims[d]) d-> ThrowBoundsError("Dimension $d out of bounds $(i[d]) > $R") d -> nothing)
    end
    quote
        
        $(Expr(:meta, :inline))
        @boundscheck begin
            $bound_check
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
@inline function Base.getindex(A::AbstractMutableFixedSizePaddedArray{S,Vec{W,T},1,L,L}, i::Int) where {S,T,L,W}
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

@inline Base.unsafe_convert(::Type{Ptr{T}}, A::AbstractMutableFixedSizePaddedArray) where {T} = Base.unsafe_convert(Ptr{T}, pointer(A))
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

@noinline function simple_vec_prod(sv::Core.SimpleVector)
    p = 1
    for n in 1:length(sv)
        p *= (sv[n])::Int
    end
    p
end

# @noinline to_tuple(S) = tuple(S.parameters...)
@generated Base.size(::AbstractFixedSizePaddedArray{S}) where {S} = tuple(S.parameters...)#to_tuple(S)


# Do we want this, or L?
@generated Base.length(::AbstractFixedSizePaddedArray{S}) where {S} = simple_vec_prod(S.parameters)


staticrangelength(::Type{Static{R}}) where R = 1 + last(R) - first(R)
"""
Note that the stride will currently only be correct when N <= 2.
Perhaps R should be made into a stride-tuple?
"""
@generated function sview(A::AbstractMutableFixedSizePaddedArray{S,T,N,P,L}, inds...) where {S,T,N,P,L}
    @assert length(inds) == N
    SV = S.parameters
    s2 = Vector{Int}(undef, N)
    offset = 0
    for n ∈ 1:N
        # @show inds[n], n
        if inds[n] == Colon
            s2[n] = SV[n]
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

"""
This function is not safe -- make sure the underlying data is protected!!!
"""
@generated function Base.vec(a::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}) where {S,T,N,R,L}
    N == 1 || S.parameters[1] == R || throw("vec-ing multidimensional arrays with padding would lead to weird behavior.")
    :(PtrVector{$L,$T,$L}(pointer(a)))
end
@generated function Base.reshape(A::AbstractMutableFixedSizePaddedArray{S1,T,N,R,L}, ::Union{Val{S2},Static{S2}}) where {S1,S2,T,N,R,L}
    N == 1 || S1.parameters[1] == R || throw("Reshaping multidimensional arrays with padding would lead to weird behavior.")
    prod(S2.parameters) == prod(S1.parameters) || throw("Total length of reshaped array should equal original length\n\n$(S1)\n\n$(S2)\n\n.")
    N2 = length(S2.parameters)
    R2 = S2.parameters[1]
    :(PtrArray{$S2,$T,$N2,$R2,$L}(pointer(A)))
end
@generated function Base.reshape(A::AbstractArray, ::Static{S}) where {S}
    tupexpr = Expr(:tuple, S.parameters...)
    :(reshape(A, $tupexpr))
end

#@generated function Base.view(A::AbstractMutableFixedSizePaddedArray{S,T,N,P}, args...) where {S,T,N,P}
#    @assert length(args) == N
#    outdim = Vector{Int}
#end




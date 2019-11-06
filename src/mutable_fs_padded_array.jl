
@noinline function calc_NPL(SV::Core.SimpleVector, T, align_stride::Bool = true, pad_to_align_length::Bool = false)#true)
    nrow = (SV[1])::Int
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
    calc_NXL(SV, T, padded_rows, align_stride, pad_to_align_length)
end
function calc_NXL(SV::Core.SimpleVector, T, padded_rows::Int, align_stride::Bool = true, pad_to_align_length::Bool = false)
    L = padded_rows
    N = length(SV)
    X = Int[ 1 ]
    for n ∈ 2:N
        push!(X, L)
        L *= (SV[n])::Int
    end
    LA = VectorizationBase.align(L,T)
    if pad_to_align_length # why???
        LA += VectorizationBase.pick_vector_width(T)
    end
    N, Tuple{X...}, LA
end


@noinline function init_mutable_fs_padded_array_quote(SV::Core.SimpleVector, T)
    N, X, LA = calc_NPL(SV, T)
    quote
        $(Expr(:meta,:inline))
        FixedSizeArray{Tuple{$(SV...)},$T,$N,$X,$LA}(undef)
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
mutable struct FixedSizeArray{S,T,N,X,L} <: AbstractMutableFixedSizeArray{S,T,N,X,L}
    data::NTuple{L,T}
    @inline function FixedSizeArray{S,T,N,X,L}(::UndefInitializer) where {S,T,N,X,L}
        new()
    end
    @inline function FixedSizeArray{S,T,N,X,L}(data::NTuple{L,T}) where {S,T,N,X,L}
        new(data)
    end
end
@generated function FixedSizeArray{S,T}(::UndefInitializer) where {S,T}
    init_mutable_fs_padded_array_quote(S.parameters, T)
end
@generated function FixedSizeArray{S,T,N}(::UndefInitializer) where {S,T,N}
    init_mutable_fs_padded_array_quote(S.parameters, T)
end
const FixedSizeVector{M,T,L} = FixedSizeArray{Tuple{M},T,1,Tuple{1},L}
const FixedSizeMatrix{M,N,T,P,L} = FixedSizeArray{Tuple{M,N},T,2,Tuple{1,P},L}
@generated function FixedSizeVector{S,T}(::UndefInitializer) where {S,T}
    L = calc_padding(S,T)
    :(FixedSizeVector{$S,$T,$L}(undef))
end

@generated function FixedSizeMatrix{M,N,T}(::UndefInitializer) where {M,N,T}
    P = calc_padding(M,T)
    :(FixedSizeMatrix{$M,$N,$T,$P,$(P*N)}(undef))
end

# @inline FixedSizeVector(A::AbstractConstantFixedSizeVector{M,T,L}) where {M,T,L} = FixedSizeVector{M,T,L}(A.data)
# @inline FixedSizeMatrix(A::AbstractConstantFixedSizeArray{M,N,T,P,L}) where {M,N,T,P,L} = FixedSizeMatrix{M,N,T,P,L}(A.data)
@inline FixedSizeArray(A::AbstractConstantFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L} = FixedSizeArray{S,T,N,X,L}(A.data)

@generated function FixedSizeArray(::UndefInitializer, ::Val{S}, ::Type{T1}=Float64) where {S,T1}
    init_mutable_fs_padded_array_quote(Tuple{S...}.parameters, T1)
end
@generated function FixedSizeArray{S,T,N,X}(::UndefInitializer) where {S,T,N,X}
    L = simple_vec_prod(X.parameters) * last(S.parameters)::Int
    :(FixedSizeArray{$S,$T,$N,$X,$L}(undef))
end

@noinline function mutable_zero_quote(S::Core.SimpleVector, T)
    N,P,L = calc_NPL(S, T)
    quote
        $(Expr(:meta,:inline))
        ma = FixedSizeArray{$(Tuple{S...}),$T,$N,$P,$L}(undef)
        @inbounds for l ∈ 1:$L
            ma[l] = zero($T)
        end
        ma
    end
end
@generated function Base.zero(::Type{<:FixedSizeArray{S,T}}) where {T,S}
    mutable_zero_quote(S.parameters, T)
end
@generated function Base.zero(::Type{<:FixedSizeArray{S}}) where {S}
    mutable_zero_quote(S.parameters, Float64)
end
@generated function Base.zero(::Type{<:FixedSizeArray{S,Vec{W,T}}}) where {S,W,T}
    N, P, L = calc_NPL(S.parameters, Vec{W,T})
    quote
        $(Expr(:meta,:inline))
        ma = FixedSizeArray{$S,Vec{$W,$T},$N,$P,$L}(undef)
        @inbounds for l ∈ 1:$L
            ma[l] = SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T))
        end
        ma
    end
end
function zero!(A::AbstractMutableFixedSizeArray{S,T,N,P,L}) where {S,T,N,P,L}
    @inbounds for i ∈ 1:L
        A[i] = zero(T)
    end
end
function zero!(A::AbstractMutableFixedSizeArray{S,Vec{W,T},N,P,L}) where {S,W,T,N,P,L}
    ptrA = reinterpret(Ptr{T}, pointer(A))
    @inbounds for i ∈ 0:L-1
        VectorizationBase.vstore!(ptrA + i*sizeof(T), zero(T))
    end
end

@generated function Base.copy(A::AbstractFixedSizeArray{S,T,N,P,L}) where {S,T,N,P,L}
    quote
        B = FixedSizeArray{$S,$T,$N,$P,$L}(undef)
        @vvectorize $T for l ∈ 1:$L
            B[l] = A[l]
        end
        B
    end
end
@generated function Base.copyto!(B::AbstractMutableFixedSizeArray{S,T,N,P,L}, A::AbstractFixedSizeArray{S,T,N,P,L}) where {S,T,N,P,L}
    quote
        $(Expr(:meta,:inline))
        @vvectorize $T for l ∈ 1:$L
            B[l] = A[l]
        end
        B
    end
end
Base.similar(A::AbstractMutableFixedSizeArray{S,T,N,P,L}) where {S,T,N,P,L} = FixedSizeArray{S,T,N,P,L}(undef)
Base.similar(A::AbstractMutableFixedSizeArray{S,T1,N,P,L},::Type{T2}) where {S,T1,T2,N,P,L} = FixedSizeArray{S,T2,N}(undef)

FixedSizeArray(A::AbstractArray{T,N}) where {T,N} = copyto!(FixedSizeArray{Tuple{size(A)...},T}(undef), A)
FixedSizeVector(A::AbstractVector{T}) where {T} = copyto!(FixedSizeVector{length(A),T}(undef), A)
function FixedSizeMatrix(A::AbstractMatrix{T}) where {T}
    M,N = size(A)
    copyto!(FixedSizeMatrix{M,N,T}(undef), A)
end
FixedSizeArray{S}(A::AbstractArray{T,N}) where {S,T,N} = copyto!(FixedSizeArray{S,T}(undef), A)
FixedSizeArray{S,T}(A::AbstractArray{T,N}) where {S,T,N} = copyto!(FixedSizeArray{S,T}(undef), A)
FixedSizeArray{S,T,N}(A::AbstractArray{T,N}) where {S,T,N} = copyto!(FixedSizeArray{S,T}(undef), A)
FixedSizeArray{S,T,N,X}(A::AbstractArray{T,N}) where {S,T,N,X} = copyto!(FixedSizeArray{S,T,N,X}(undef), A)
FixedSizeArray{S,T,N,X,L}(A::AbstractArray{T,N}) where {S,T,N,X,L} = copyto!(FixedSizeArray{S,T,N,X,L}(undef), A)
FixedSizeVector{L}(A::AbstractVector{T}) where {L,T} = copyto!(FixedSizeVector{L,T}(undef), A)
FixedSizeMatrix{M,N}(A::AbstractMatrix{T}) where {M,N,T} = copyto!(FixedSizeMatrix{M,N,T}(undef), A)
FixedSizeArray{S,T1}(A::AbstractArray{T2,N}) where {S,T1,T2,N} = copyto!(FixedSizeArray{S,T1}(undef), A)
FixedSizeVector{L,T1}(A::AbstractVector{T2}) where {L,T1,T2} = copyto!(FixedSizeVector{L,T1}(undef), A)
FixedSizeMatrix{M,N,T1}(A::AbstractMatrix{T2}) where {M,N,T1,T2} = copyto!(FixedSizeMatrix{M,N,T1}(undef), A)

mutable_similar(A::AbstractArray) = similar(A)
mutable_similar(A::AbstractFixedSizeArray{S,T,N,R,L}) where {S,T,N,R,L} = FixedSizeArray{S,T,N,R,L}(undef)
mutable_similar(A::AbstractArray, T) = similar(A, T)
mutable_similar(A::AbstractFixedSizeArray{S,T1,N,R,L}, ::Type{T2}) where {S,T1,T2,N,R,L} = FixedSizeArray{S,T2,R,L}(undef)
function Base.fill!(A::AbstractMutableFixedSizeArray{S,T,N,R,L}, v::T) where {S,T,N,R,L}
    @inbounds for l ∈ 1:L
        A[l] = v
    end
    A
end

"""
This is only meant to make recursive algorithms easiesr to implement.
Wraps a pointer, while passing info on the size of the block and stride.
"""
struct PtrArray{S,T,N,X,L,V} <: AbstractMutableFixedSizeArray{S,T,N,X,L}
    ptr::Ptr{T}
end
@generated function PtrArray{S,T,N,X,L}(ptr::Ptr{T},::Val{V}=Val{false}()) where {S,T,N,X,L,V}
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$X,$L,$V}(ptr)
    end
end
@generated function PtrArray{S,T,N,X}(ptr::Ptr{T},::Val{V}=Val{false}()) where {S,T,N,X,V}
    L = simple_vec_prod(X.parameters) * last(S.parameters)::Int
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$X,$L,$V}(ptr)
    end
end
@generated function PtrArray{S,T,N}(ptr::Ptr{T},::Val{V}=Val{false}()) where {S,T,N,V}
    N2, X, L = calc_NPL(S.parameters, T)
    @assert N == N2 "length(S) == $(length(S.parameters)) != N == $N"
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$X,$L,$V}(ptr)
    end
end
@generated function PtrArray{S,T}(::UndefInitializer) where {S,T}
    N, X, L = calc_NPL(S.parameters, T, true, false)
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$X,$L,false}(SIMDPirates.alloca(Val{$L}(), $T))
    end
end
@generated function PtrArray{S,T}(ptr::Ptr{T}, ::Val{V} = Val{false}()) where {S,T,V}
    N, X, L = calc_NPL(S.parameters, T, true, false)
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$X,$L,$V}(ptr)
    end
end
@generated function PtrArray{S}(ptr::Ptr{T}) where {S,T}
    N, X, L = calc_NPL(S.parameters, T, true, false)
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$X,$L,true}(ptr)
    end
end
@generated function PtrArray{S}(ptr::Ptr{T}, ::Val{V}) where {S,T,V}
    N,X,L = calc_NPL(S.parameters,T,true,false)
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$X,$L,$V}(ptr)
    end
end
@generated function PtrArray{S}(sp::StackPointer) where {S}
    N,X,L = calc_NPL(S.parameters,Float64,true,false)
    quote
        $(Expr(:meta,:inline))
        sp + $(VectorizationBase.align(8L)), PtrArray{$S,Float64,$N,$X,$L,false}(pointer(sp, Float64))
    end
end
@generated function PtrArray{S,T}(sp::StackPointer) where {S,T}
    N,X,L = calc_NPL(S.parameters,T,true,false)
    quote
        $(Expr(:meta,:inline))
        sp + $(VectorizationBase.align(sizeof(T)*L)), PtrArray{$S,$T,$N,$X,$L,false}(pointer(sp, $T))
    end
end
@generated function PtrArray{S,T,N}(sp::StackPointer) where {S,T,N}
    N2, X, L = calc_NPL(S.parameters, T)
    @assert N == N2 "length(S) == $(length(S.parameters)) != N == $N"
    quote
        $(Expr(:meta,:inline))
        sp + $(VectorizationBase.align(sizeof(T)*L)), PtrArray{$S,$T,$N,$X,$L,false}(pointer(sp, $T))
    end
end
@generated function PtrArray{S,T,N,X}(sp::StackPointer) where {S,T,N,X}
    L = simple_vec_prod(X.parameters) * last(S.parameters)::Int
    quote
        $(Expr(:meta,:inline))
        ptr = Base.unsafe_convert(Ptr{$T}, sp.ptr)
        sp + $(VectorizationBase.align(sizeof(T)*L)), PtrArray{$S,$T,$N,$X,$L,false}(pointer(sp, $T))
#        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
    end
end
@generated function PtrArray{S,T,N,X,L}(sp::StackPointer) where {S,T,N,X,L}
    quote
        $(Expr(:meta,:inline))
        ptr = Base.unsafe_convert(Ptr{$T}, sp.ptr)
        sp + $(VectorizationBase.align(sizeof(T)*L)), PtrArray{$S,$T,$N,$X,$L,false}(pointer(sp, $T))
#        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
    end
end
@generated function PtrArray{S,T,N,X,L,V}(sp::StackPointer) where {S,T,N,X,L,V}
    quote
        $(Expr(:meta,:inline))
        ptr = Base.unsafe_convert(Ptr{$T}, sp.ptr)
        sp + $(VectorizationBase.align(sizeof(T)*L)), PtrArray{$S,$T,$N,$X,$L,$V}(pointer(sp, $T))
#        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
    end
end
@generated function PtrArray(A::AbstractMutableFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$X,$L,false}(pointer(A))
    end
end

const PtrVector{M,T,L,V} = PtrArray{Tuple{M},T,1,Tuple{1},L,V} # R and L will always be the same...
const PtrMatrix{M,N,T,P,L,V} = PtrArray{Tuple{M,N},T,2,Tuple{1,P},L,V}

@generated function PtrVector{P,T}(a::Ptr{T}) where {P,T}
    L = calc_padding(P, T)
    quote
        $(Expr(:meta,:inline))
        PtrArray{Tuple{$P},$T,1,Tuple{1},$L,false}(a)
    end
end
@generated function PtrVector{P,T}(a::StackPointer) where {P,T}
    L = calc_padding(P, T)
    quote
        $(Expr(:meta,:inline))
        a + $(VectorizationBase.align(L*sizeof(T))), PtrArray{Tuple{$P},$T,1,Tuple{1},$L,false}(pointer(a,$T))
    end
end
@generated function PtrVector{P,T}(::UndefInitializer) where {P,T,V}
    L = calc_padding(P, T)
    quote
        $(Expr(:meta,:inline))
        PtrArray{Tuple{$P},$T,1,Tuple{1},$L,false}(SIMDPirates.alloca(Val{$L}(),$T))
    end
end
@generated function PtrMatrix{M,N,T}(a::Ptr{T}) where {M,N,T}
    L = calc_padding(M, T)
    quote
        $(Expr(:meta,:inline))
        PtrArray{Tuple{$M,$N},$T,2,Tuple{1,$L},$(L*N),false}(a)
    end
end
@generated function PtrMatrix{M,N,T}(a::StackPointer) where {M,N,T}
    L = calc_padding(M, T)
    quote
        $(Expr(:meta,:inline))
        a + $(VectorizationBase.align(L*N*sizeof(T))), PtrArray{Tuple{$M,$N},$T,2,Tuple{1,$L},$(L*N),false}(pointer(a,$T))
    end
end
@generated function PtrMatrix{M,N,T}(::UndefInitializer) where {M,N,T}
    L = calc_padding(M, T)
    quote
        $(Expr(:meta,:inline))
        PtrArray{Tuple{$M,$N},$T,2,Tuple{1,$L},$(L*N),false}(SIMDPirates.alloca(Val{$(L*N)}(),$T))
    end
end

@inline Base.similar(sp::StackPointer, ::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L} = PtrArray{S,T,N,X,L,false}(sp)
@inline Base.pointer(A::PtrArray) = A.ptr
@inline Base.pointer(A::AbstractMutableFixedSizeArray{S,T}) where {S,T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@inline Base.pointer(A::AbstractMutableFixedSizeArray{S,NTuple{W,Core.VecElement{T}}}) where {S,T,W} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@inline Base.pointer(A::PtrArray{S,NTuple{W,Core.VecElement{T}}}) where {S,T,W} = Base.unsafe_convert(Ptr{T}, A.ptr)


@inline VectorizationBase.vectorizable(A::AbstractMutableFixedSizeArray) = VectorizationBase.Pointer(pointer(A))
@inline VectorizationBase.vectorizable(A::LinearAlgebra.Diagonal{T,<:AbstractMutableFixedSizeArray}) where {T} = VectorizationBase.Pointer(pointer(A.diag))
@inline VectorizationBase.vectorizable(A::LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizeArray}) where {T} = VectorizationBase.Pointer(pointer(A.parent))


# @inline Base.similar(sp::StackPointer, A::AbstractMutableFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L} = PtrArray{S,T,N,X,L,false}(sp)
@inline function Base.copy(sp::StackPointer, A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    sp, B = PtrArray{S,T,N,X,L,false}(sp)
    sp, copyto!(B, A)
end

isview(::Any) = false
isview(::PtrArray{S,T,N,X,L,V}) where {S,T,N,X,L,V} = V
isview(::Type{PtrArray{S,T,N,X,L,V}}) where {S,T,N,X,L,V} = V

@generated function Base.setindex!(A::AbstractMutableFixedSizeArray{S,T,N,X,L}, v, i::CartesianIndex{N}) where {S,T,N,X,L}
    ex = sub2ind_expr(X.parameters)
    R = isview(A) ? (S.parameters[1])::Int : (X.parameters[2])::Int
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d-> ( d == 1 ? i[d] > $R : i[d] > size(A,d)) d -> ThrowBoundsError("Dimension $d out of bounds") d -> nothing
        end
        # T = eltype(A)
        VectorizationBase.store!(pointer(A) + sizeof(T) * $ex, convert($T, v))
        v
    end
end
@generated function Base.setindex!(A::AbstractMutableFixedSizeArray{S,T,N,X,L}, v, i::Vararg{Integer,M}) where {S,T,N,X,L,M}
    if M == 1 && !isview(A)
        bound_check = :(i[1] > $L && ThrowBoundsError("index == $(i[1]) > $L == length of array"))
        ex = :(@inbounds i[1] - 1)
    else
        ex = sub2ind_expr(X.parameters)
        R = isview(A) ? (S.parameters[1])::Int : (X.parameters[2])::Int
        bound_check = :(Base.Cartesian.@nif $(N+1) d->( d == 1 ? i[d] > $R : i[d] > size(A,d)) d-> ThrowBoundsError("Dimension $d out of bounds") d -> nothing)
    end
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            $bound_check
        end
        VectorizationBase.store!(pointer(A) + sizeof(T) * $ex, convert($T,v))
        v
    end
end
@generated function Base.setindex!(
    A::AbstractMutableFixedSizeArray{S,NTuple{W,Core.VecElement{T}},N,X,L},
    v::NTuple{W,Core.VecElement{T}}, i::CartesianIndex{N}
) where {S,T,N,X,L,W}
    R = isview(A) ? (S.parameters[1])::Int : (X.parameters[2])::Int
    ex = sub2ind_expr(X.parameters)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->( d == 1 ? i[d] > $R : i[d] > size(A,d)) d -> ThrowBoundsError("Dimension $d out of bounds") d -> nothing
        end
        SIMDPirates.vstore!(pointer(A) + sizeof(NTuple{W,Core.VecElement{T}}) * $ex, v)
        v
    end
end
@generated function Base.setindex!(
    A::AbstractMutableFixedSizeArray{S,NTuple{W,Core.VecElement{T}},N,X,L},
    v::NTuple{W,Core.VecElement{T}}, i::Vararg{Integer,M}
) where {S,T,N,X,L,M,W}
    R = isview(A) ? (S.parameters[1])::Int : (X.parameters[2])::Int
    ex = sub2ind_expr(X.parameters)
    if M == 1
        ex = :(@inbounds i[1] - 1)
    else
        R = isview(A) ? (S.parameters[1])::Int : (X.parameters[2])::Int
        ex = sub2ind_expr(dims, R)
    end
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->( d == 1 ? i[d] > $R : i[d] > size(A,d)) d-> ThrowBoundsError("Dimension $d out of bounds") d -> nothing
        end
        SIMDPirates.vstore!(pointer(A) + sizeof(NTuple{W,Core.VecElement{T}}) * $ex, v)
        v
    end
end
# AbstractMutableFixedSizeArray

                                            
# @inline Base.@propagate_inbounds Base.getindex(A::AbstractMutableFixedSizeArray, i...) = A.data[i...]

@inline function Base.getindex(A::AbstractMutableFixedSizeVector{S,T,L}, i::Integer) where {S,T,L}
    @boundscheck i <= L || ThrowBoundsError("Index $i > full length $L.")
    VectorizationBase.load(pointer(A) + sizeof(T) * (i - 1))
end
@inline function Base.getindex(A::AbstractMutableFixedSizeArray, i::Integer)
    @boundscheck i <= full_length(A) || ThrowBoundsError("Index $i > full length $(full_length(A)).")
    T = eltype(A)
    VectorizationBase.load(pointer(A) + sizeof(T) * (i - 1))
end
@inline function Base.getindex(A::AbstractMutableFixedSizeVector{S,Vec{W,T},L}, i::Integer) where {S,T,L,W}
    @boundscheck i <= full_length(A) || ThrowBoundsError("Index $i > full length $L.")
    SIMDPirates.vload(Vec{W,T}, pointer(A) + sizeof(Vec{W,T}) * (i - 1))
end
@inline function Base.getindex(A::AbstractMutableFixedSizeArray{S,Vec{W,T},N,X,L}, i::Integer) where {S,T,L,W,N,X}
    @boundscheck i <= L || ThrowBoundsError("Index $i > full length $L.")
    SIMDPirates.vload(Vec{W,T}, pointer(A) + sizeof(Vec{W,T}) * (i - 1))
end
@inline function Base.getindex(A::LinearAlgebra.Adjoint{Union{},<: AbstractMutableFixedSizeArray{S,Vec{W,T},N,X,L}}, i::Integer) where {S,T,L,W,N,X}
    @boundscheck i <= L || ThrowBoundsError("Index $i > full length $L.")
    SIMDPirates.vload(Vec{W,T}, pointer(A.parent) + sizeof(Vec{W,T}) * (i - 1))
end


@generated function Base.getindex(A::AbstractMutableFixedSizeArray{S,T,N,X,L}, i::Vararg{<:Integer,N}) where {S,T,N,X,L}
    R = isview(A) ? (S.parameters[1])::Int : (X.parameters[2])::Int
    ex = sub2ind_expr(X.parameters)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > size(A,d)) d->ThrowBoundsError() d -> nothing
        end
        VectorizationBase.load(pointer(A) + $(sizeof(T)) * $ex )
    end
end
@generated function Base.getindex(A::AbstractMutableFixedSizeArray{S,T,N,X,L}, i::CartesianIndex{N}) where {S,T,N,X,L}
    R = isview(A) ? (S.parameters[1])::Int : (X.parameters[2])::Int
    ex = sub2ind_expr(X.parameters)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > size(A, d)) d->ThrowBoundsError() d -> nothing
        end
        VectorizationBase.load(pointer(A) + $(sizeof(T)) * $ex )
    end
end
@generated function Base.getindex(A::AbstractMutableFixedSizeArray{S,Vec{W,T},N,X,L}, i::Vararg{Int,N}) where {S,T,N,X,L,W}
    R = isview(A) ? (S.parameters[1])::Int : (X.parameters[2])::Int
    ex = sub2ind_expr(X.parameters)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > size(A,d)) d->ThrowBoundsError() d -> nothing
        end
        SIMDPirates.vload(Vec{W,T}, pointer(A) + $(sizeof(Vec{W,T})) * $ex )
    end
end
@generated function Base.getindex(A::AbstractMutableFixedSizeArray{S,Vec{W,T},N,X,L}, i::CartesianIndex{N}) where {S,T,N,X,L,W}
    R = isview(A) ? (S.parameters[1])::Int : (X.parameters[2])::Int
    ex = sub2ind_expr(X.parameters)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > size(A,d)) d->ThrowBoundsError() d -> nothing
        end
        SIMDPirates.vload(Vec{W,T}, pointer(A) + $(sizeof(Vec{W,T})) * $ex )
    end
end
@generated function Base.getindex(A::LinearAlgebra.Adjoint{Union{},<: AbstractMutableFixedSizeMatrix{M,N,Vec{W,T},X,L}}, i::Int, j::Int) where {M,N,W,T,X,L}
    ex = 
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            if (j > M) || (i > N)
                ThrowBoundsError("At least one of: $j > $M or $i > $N.")
            end
        end
        SIMDPirates.vload(Vec{W,T}, pointer(A.parent) + $(sizeof(Vec{W,T})) * ( (i-1)*$R + j-1 ))
    end
end
function Base.getindex(A::AbstractMutableFixedSizeVector{S,T,L}, i::Int, j::Int) where {S,T,L}
    @boundscheck begin
        (j != 1 || i > S || i < 1) && ThrowBoundsError()
    end
    @inbounds A[i]
end

@inline Base.unsafe_convert(::Type{Ptr{T}}, A::AbstractMutableFixedSizeArray) where {T} = Base.unsafe_convert(Ptr{T}, pointer(A))
@generated Base.strides(A::AbstractFixedSizeArray{S,T,N,X}) where {S,T,N,X} = tuple(X.parameters...)
@generated Base.stride(A::AbstractFixedSizeArray{S,T,N,X,L}, n::Int) where {S,T,N,X,L} = :(n > $N ? $L : @inbounds $(tuple(X.parameters...))[n])

@generated Base.size(::AbstractFixedSizeArray{S}) where {S} = tuple(S.parameters...)#to_tuple(S)

# Do we want this, or L?
@generated Base.length(::AbstractFixedSizeArray{S}) where {S} = simple_vec_prod(S.parameters)


"""
This function is not safe -- make sure the underlying data is protected!!!
"""
@generated function Base.vec(a::AbstractMutableFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    N == 1 || (S.parameters[1])::Int == (X.parameters[2])::Int || throw("vec-ing multidimensional arrays with padding would lead to weird behavior.")
    :(PtrVector{$L,$T,$L}(pointer(a)))
end
@noinline function reshape_quote(S1,S2,T,N,X,L)
    N == 1 || (S1.parameters[1])::Int == (X.parameters[2])::Int || throw("Reshaping multidimensional arrays with padding would lead to weird behavior.")
    prod(S2.parameters) == prod(S1.parameters) || throw("Total length of reshaped array should equal original length\n\n$(S1)\n\n$(S2)\n\n.")
    N2 = length(S2.parameters)
    R2 = (S2.parameters[1])::Int
    X2 = Vector{Int}(undef, N2); X2[1] = 1
    for n in 2:N2
        X2[n] = X2[n-1] * (S2.parameters[n-1])::Int
    end
    :(PtrArray{$S2,$T,$N2,$(Tuple{X2...}),$L}(pointer(A)))

end
@generated Base.reshape(A::AbstractMutableFixedSizeArray{S1,T,N,X,L}, ::Val{S2}) where {S1,S2,T,N,X,L} = reshape_quote(S1,S2,T,N,X,L)
@generated Base.reshape(A::AbstractMutableFixedSizeArray{S1,T,N,X,L}, ::Static{S2}) where {S1,S2,T,N,X,L} = reshape_quote(S1,S2,T,N,X,L)
@generated function Base.reshape(A::AbstractArray, ::Static{S}) where {S}
    tupexpr = Expr(:tuple, S.parameters...)
    :(reshape(A, $tupexpr))
end






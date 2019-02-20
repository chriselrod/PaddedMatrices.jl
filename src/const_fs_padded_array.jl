
struct ConstantFixedSizePaddedArray{S,T,N,P,L} <: AbstractConstantFixedSizePaddedArray{S,T,N,P,L}
    data::NTuple{L,T}
end
const ConstantFixedSizePaddedVector{S,T,P,L} = ConstantFixedSizePaddedArray{Tuple{S},T,1,P,L}
const ConstantFixedSizePaddedMatrix{M,N,T,P,L} = ConstantFixedSizePaddedArray{Tuple{M,N},T,2,P,L}

@generated function Base.size(::AbstractConstantFixedSizePaddedArray{S,T,N}) where {S,T,N}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:tuple, [S.parameters[n] for n ∈ 1:N]...))
    end
end
@generated function Base.length(A::AbstractConstantFixedSizePaddedArray{S,T,N}) where {S,T,N}
    L = S.parameters[1]
    for n ∈ 2:N
        L *= S.parameters[n]
    end
    quote
        $(Expr(:meta, :inline))
        $L
    end
end

"""
Returns zero based index. Don't forget to add one when using with arrays instead of pointers.
"""
function sub2ind_expr(S, P)
    N = length(S)
    N == 1 && return :(i[1] - 1)
    ex = :(i[$N] - 1)
    for i ∈ (N - 1):-1:2
        ex = :(i[$i] - 1 + $(S[i]) * $ex)
    end
    :(i[1] + $P * $ex)
end

@inline function Base.getindex(A::AbstractConstantFixedSizePaddedArray{S,T,1,L,L}, i::Int) where {S,T,L}
    @boundscheck i <= L || ThrowBoundsError("Index $i < full length $(full_length(A)).")
    @inbounds A.data[i]
end
@inline function Base.getindex(A::AbstractConstantFixedSizePaddedArray, i::Int)
    @boundscheck i <= full_length(A) || ThrowBoundsError("Index $i < full length $(full_length(A)).")
    @inbounds A.data[i]
end
@generated function Base.getindex(A::AbstractConstantFixedSizePaddedArray{S,T,N,P}, i::Vararg{<:Integer,N}) where {S,T,N,P}
    ex = sub2ind_expr(S, P)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > $sv[d]) d->ThrowBoundsError()) d -> nothing
        end
        @inbounds A.data[$ex + 1]
    end
end
@generated function Base.getindex(A::AbstractConstantFixedSizePaddedArray{S,T,N,P}, i::CartesianIndex{N}) where {S,T,N,P}
    ex = sub2ind_expr(S, P)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > $sv[d]) d->ThrowBoundsError()) d -> nothing
        end
        @inbounds A.data[$ex + 1]
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

struct vStaticPaddedArray{SPA}
    spa::SPA
    offset::Int
end
@inline VectorizationBase.vectorizable(A::AbstractConstantFixedSizePaddedArray) = vStaticPaddedArray(A,0)
@inline Base.:+(A::vStaticPaddedArray, i) = vStaticPaddedArray(A.spa, A.offset + i)
@inline Base.:+(i, A::vStaticPaddedArray) = vStaticPaddedArray(A.spa, A.offset + i)
@inline Base.:-(A::vStaticPaddedArray, i) = vStaticPaddedArray(A.spa, A.offset - i)

@generated function SIMDPirates.vload(::Type{Vec{N,T}}, A::vStaticPaddedArray, i::Integer)
    quote
        $(Expr(:meta, :inline))
        @inbounds $(Expr(:tuple, [:(Core.VecElement(A.spa.data[i + $n + A.offset])) for n ∈ 0:N-1]...))
    end
end
@generated function SIMDPirates.vload(::Type{SVec{N,T}}, A::StaticPaddedArray, i::Integer)
    quote
        $(Expr(:meta, :inline))
        @inbounds SVec($(Expr(:tuple, [:(Core.VecElement(A.spa.data[i + $n + A.offset])) for n ∈ 0:N-1]...)))
    end
end
@generated function SIMDPirates.vload(::Type{Vec{N,T}}, A::vStaticPaddedArray)
    quote
        $(Expr(:meta, :inline))
        @inbounds $(Expr(:tuple, [:(Core.VecElement(A.spa.data[$n + A.offset])) for n ∈ 1:N]...))
    end
end
@generated function SIMDPirates.vload(::Type{SVec{N,T}}, A::StaticPaddedArray)
    quote
        $(Expr(:meta, :inline))
        @inbounds SVec($(Expr(:tuple, [:(Core.VecElement(A.spa.data[$n + A.offset])) for n ∈ 1:N]...)))
    end
end

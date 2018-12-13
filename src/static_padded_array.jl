
struct StaticPaddedArray{S,T,N,P,L} <: AbstractPaddedArray{T,N}
    data::NTuple{L,T}
end

@generated function Base.size(::StaticPaddedArray{S,T,N}) where {S,T,N}
    quote
        $(Expr(:meta, :inline))
        $(ntuple(n -> S.parameters[n], Val(N)))
    end
end
@generated function Base.length(A::StaticPaddedArray{S,T,N}) where {S,T,N}
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

@generated function Base.getindex(A::StaticPaddedArray{S,T,N,P}, i::Vararg{<:Integer,N}) where {S,T,N,P}
    ex = sub2ind_expr(S, P)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > $sv[d]) d->ThrowBoundsError()) d -> nothing
        end
        A.data[$ex + 1]
    end
end
@generated function Base.getindex(A::StaticPaddedArray{S,T,N,P}, i::CartesianIndex{N}) where {S,T,N,P}
    ex = sub2ind_expr(S, P)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > $sv[d]) d->ThrowBoundsError()) d -> nothing
        end
        A.data[$ex + 1]
    end
end

@generated function Base.strides(A::StaticPaddedArray{S,T,N,P}) where {S,T,N,P}
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

@inline VectorizationBase.vectorizable(A::StaticPaddedArray) = A

@generated function SIMDPirates.vload(::Type{Vec{N,T}}, A::StaticPaddedArray, i::Integer)
    quote
        $(Expr(:meta, :inline))
        $(Expr(:tuple, [:(Core.VecElement(A.data[i + $n])) for n ∈ 0:N-1]...))
    end
end


struct LazyMap{F,S,T,N,X,L} <: AbstractFixedSizeArray{S,T,N,X,L}
    f::F
    ptr::Ptr{T}
end
struct VectorizableMap{F,T}
    f::F
    ptr::Ptr{T}
end
@inline Base.pointer(m::LazyMap) = m.ptr
@inline VectorizationBase.vectorizable(m::LazyMap{F,S,T}) where {F,S,T} = VectorizableMap{F,T}(m.f, m.ptr)
@inline Base.:+(m::VectorizableMap{F,T}, i::Integer) where {F,T} = VectorizableMap{F,T}(m.f, m.ptr + sizeof(T)*i)
@inline Base.:+(i::Integer, m::VectorizableMap{F,T}) where {F,T} = VectorizableMap{F,T}(m.f, m.ptr + sizeof(T)*i)
@inline Base.:-(m::VectorizableMap{F,T}, i::Integer) where {F,T} = VectorizableMap{F,T}(m.f, m.ptr - sizeof(T)*i)

@inline function LazyMap(f::F, A::AbstractMutableFixedSizeArray{S,T,N,X,L}) where {F,S,T,N,X,L}
    LazyMap{F,S,T,N,X,L}(f, pointer(A))
end

for (f, (m, sf)) ∈ LoopVectorization.SLEEFPiratesDict
    @eval @inline function LazyMap(f::typeof($f), A::AbstractMutableFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
        LazyMap{F,S,T,N,X,L}($m.$sf, pointer(A))
    end
end
    
@inline function SIMDPirates.vload(::Type{Vec{W,T}}, m::VectorizableMap{F,T}) where {W,F,T}
    m.f(vload(Vec{W,T}, m.ptr))
end
@inline function SIMDPirates.vload(::Type{Vec{W,T}}, m::VectorizableMap{F,T}, mask) where {W,F,T}
    m.f(vload(Vec{W,T}, m.ptr, mask))
end


@inline function Base.getindex(A::LazyMap{F,S,T,1,Tuple{1},L}, i::Int) where {F,S,T,N,X,L}
    @boundscheck i <= L || ThrowBoundsError("Index $i > full length $L.")
    A.f(VectorizationBase.load(pointer(A) + sizeof(T) * (i - 1)))
end
@inline function Base.getindex(A::LazyMap{F,S,T,N,X,L}, i::Int) where {F,S,T,N,X,L}
    @boundscheck i <= L || ThrowBoundsError("Index $i > full length $(full_length(A)).")
    A.f(VectorizationBase.load(pointer(A) + sizeof(T) * (i - 1)))
end
@generated function Base.getindex(A::LazyMap{F,S,T,N,X,L}, i::Vararg{<:Integer,N}) where {F,S,T,N,X,L}
    R = (S.parameters[1])::Int
    ex = sub2ind_expr(X.parameters)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > size(A,d)) d->ThrowBoundsError() d -> nothing
        end
        A.f(VectorizationBase.load(pointer(A) + $(sizeof(T)) * $ex ))
    end
end
@generated function Base.getindex(A::LazyMap{F,S,T,N,X,L}, i::CartesianIndex{N}) where {F,S,T,N,X,L}
    R = S.parameters[1])::Int
    ex = sub2ind_expr(X.parameters)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > size(A, d)) d->ThrowBoundsError() d -> nothing
        end
        A.f(VectorizationBase.load(pointer(A) + $(sizeof(T)) * $ex ))
    end
end

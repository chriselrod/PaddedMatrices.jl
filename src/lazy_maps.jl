
struct MappedStridedPointer{F, T, P <: VectorizationBase.AbstractPointer{T}}
    f::F
    ptr::P
end
@inline VectorizationBase.stridedpointer(A::LazyMap) = MappedStridedPointer(A.f, stridedpointer(PtrArray(A)))
# @inline VectorizationBase.vectorizable(m::LazyMap{F,S,T}) where {F,S,T} = MappedStridedPointer{F,T}(m.f, m.ptr)
# @inline Base.:+(m::MappedStridedPointer{F,T}, i::Integer) where {F,T} = MappedStridedPointer{F,T}(m.f, gep(m.ptr, i))
# @inline Base.:+(i::Integer, m::MappedStridedPointer{F,T}) where {F,T} = MappedStridedPointer{F,T}(m.f, gep(m.ptr, i))
# @inline Base.:-(m::MappedStridedPointer{F,T}, i::Integer) where {F,T} = MappedStridedPointer{F,T}(m.f, gep(m.ptr, -i))

@inline function LazyMap(f::F, A::AbstractStrideArray{S,T,N,X,SN,XN,V,L}) where {F,S,T,N,X,SN,XN,V,L}
    LazyMap{F,S,T,N,X,L}(f, pointer(A))
end
    
@inline function SIMDPirates.vload(::Type{Vec{W,T}}, m::MappedStridedPointer{F,T}) where {W,F,T}
    extract_data(m.f(vload(SVec{W,T}, m.ptr)))
end
@inline function SIMDPirates.vload(::Type{Vec{W,T}}, m::MappedStridedPointer{F,T}, i) where {W,F,T}
    extract_data(m.f(vload(SVec{W,T}, m.ptr, i)))
end
@inline function SIMDPirates.vload(::Type{Vec{W,T}}, m::MappedStridedPointer{F,T}, mask::Union{<:Unsigned,Vec{W,Bool}}) where {W,F,T}
    extract_data(m.f(vload(SVec{W,T}, m.ptr, mask)))
end
@inline function SIMDPirates.vload(::Type{Vec{W,T}}, m::MappedStridedPointer{F,T}, i, mask::Union{<:Unsigned,Vec{W,Bool}}) where {W,F,T}
    extract_data(m.f(vload(SVec{W,T}, m.ptr, i, mask)))
end
@inline function SIMDPirates.vload(::Type{SVec{W,T}}, m::MappedStridedPointer{F,T}) where {W,F,T}
    m.f(vload(SVec{W,T}, m.ptr))
end
@inline function SIMDPirates.vload(::Type{SVec{W,T}}, m::MappedStridedPointer{F,T}, i) where {W,F,T}
    m.f(vload(SVec{W,T}, m.ptr, i))
end
@inline function SIMDPirates.vload(::Type{SVec{W,T}}, m::MappedStridedPointer{F,T}, mask::Union{<:Unsigned,Vec{W,Bool}}) where {W,F,T}
    m.f(vload(SVec{W,T}, m.ptr, mask))
end
@inline function SIMDPirates.vload(::Type{SVec{W,T}}, m::MappedStridedPointer{F,T}, i, mask::Union{<:Unsigned,Vec{W,Bool}}) where {W,F,T}
    m.f(vload(SVec{W,T}, m.ptr, i, mask))
end
@inline function SIMDPirates.vload(::Val{W}, m::MappedStridedPointer{F,T}) where {W,F,T}
    m.f(vload(SVec{W,T}, m.ptr))
end
@inline function SIMDPirates.vload(::Val{W}, m::MappedStridedPointer{F,T}, i) where {W,F,T}
    m.f(vload(SVec{W,T}, m.ptr, i))
end
@inline function SIMDPirates.vload(::Val{W}, m::MappedStridedPointer{F,T}, mask::Union{<:Unsigned,Vec{W,Bool}}) where {W,F,T}
    m.f(vload(SVec{W,T}, m.ptr, mask))
end
@inline function SIMDPirates.vload(::Val{W}, m::MappedStridedPointer{F,T}, i, mask::Union{<:Unsigned,Vec{W,Bool}}) where {W,F,T}
    m.f(vload(SVec{W,T}, m.ptr, i, mask))
end


@inline function Base.getindex(A::LazyMap{F,S,T,1,Tuple{1},L}, i::Int) where {F,S,T,N,X,L}
    @boundscheck i <= L || ThrowBoundsError("Index $i > full length $L.")
    A.f(VectorizationBase.load(gep(pointer(A), (i - 1))))
end
@inline function Base.getindex(A::LazyMap{F,S,T,N,X,L}, i::Int) where {F,S,T,N,X,L}
    @boundscheck i <= L || ThrowBoundsError("Index $i > full length $(full_length(A)).")
    A.f(VectorizationBase.load(gep(pointer(A), (i - 1))))
end
@generated function Base.getindex(A::LazyMap{F,S,T,N,X,L}, i::Vararg{<:Integer,N}) where {F,S,T,N,X,L}
    R = (S.parameters[1])::Int
    ex = sub2ind_expr(X.parameters)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > size(A,d)) d->ThrowBoundsError() d -> nothing
        end
        A.f(VectorizationBase.load(gep(pointer(A), $ex )))
    end
end
@generated function Base.getindex(A::LazyMap{F,S,T,N,X,L}, i::CartesianIndex{N}) where {F,S,T,N,X,L}
    R = (S.parameters[1])::Int
    ex = sub2ind_expr(X.parameters)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > size(A, d)) d->ThrowBoundsError() d -> nothing
        end
        A.f(VectorizationBase.load(gep(pointer(A), $ex )))
    end
end

@inline Base.Broadcast.materialize(sp::StackPointer, A) = (sp, A)
@inline Base.Broadcast.materialize(sp::StackPointer, A::LazyMap) = copy(sp, A)
@inline Base.Broadcast.materialize(A::LazyMap) = copy(A)



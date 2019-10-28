
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

@inline function LazyMap(f::F, A::AbstractMutableFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
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




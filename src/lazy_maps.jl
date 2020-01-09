
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

const SLEEFPiratesDict = Dict{Symbol,Tuple{Symbol,Symbol}}(
    :sin => (:SLEEFPirates, :sin_fast),
    :sinpi => (:SLEEFPirates, :sinpi),
    :cos => (:SLEEFPirates, :cos_fast),
    :cospi => (:SLEEFPirates, :cospi),
    :tan => (:SLEEFPirates, :tan_fast),
    # :log => (:SLEEFPirates, :log_fast),
    :log => (:SIMDPirates, :vlog),
    :log10 => (:SLEEFPirates, :log10),
    :log2 => (:SLEEFPirates, :log2),
    :log1p => (:SLEEFPirates, :log1p),
    # :exp => (:SLEEFPirates, :exp),
    :exp => (:SIMDPirates, :vexp),
    :exp2 => (:SLEEFPirates, :exp2),
    :exp10 => (:SLEEFPirates, :exp10),
    :expm1 => (:SLEEFPirates, :expm1),
    :inv => (:SIMDPirates, :vinv), # faster than sqrt_fast
    :sqrt => (:SIMDPirates, :sqrt), # faster than sqrt_fast
    :rsqrt => (:SIMDPirates, :rsqrt),
    :cbrt => (:SLEEFPirates, :cbrt_fast),
    :asin => (:SLEEFPirates, :asin_fast),
    :acos => (:SLEEFPirates, :acos_fast),
    :atan => (:SLEEFPirates, :atan_fast),
    :sinh => (:SLEEFPirates, :sinh),
    :cosh => (:SLEEFPirates, :cosh),
    :tanh => (:SLEEFPirates, :tanh),
    :asinh => (:SLEEFPirates, :asinh),
    :acosh => (:SLEEFPirates, :acosh),
    :atanh => (:SLEEFPirates, :atanh),
    # :erf => :(SLEEFPirates.erf),
    # :erfc => :(SLEEFPirates.erfc),
    # :gamma => :(SLEEFPirates.gamma),
    # :lgamma => :(SLEEFPirates.lgamma),
    :trunc => (:SLEEFPirates, :trunc),
    :floor => (:SLEEFPirates, :floor),
    :ceil => (:SIMDPirates, :ceil),
    :abs => (:SIMDPirates, :vabs),
    :sincos => (:SLEEFPirates, :sincos_fast),
    # :pow => (:SLEEFPirates, :pow_fast),
    :^ => (:SLEEFPirates, :pow_fast),
    # :sincospi => (:SLEEFPirates, :sincospi_fast),
    # :pow => (:SLEEFPirates, :pow),
    # :hypot => (:SLEEFPirates, :hypot_fast),
    :mod => (:SLEEFPirates, :mod),
    # :copysign => :copysign
    :one => (:SIMDPirates, :vone),
    :zero => (:SIMDPirates, :vzero),
    :erf => (:SIMDPirates, :verf)
)
for (f, (m, sf)) ∈ LoopVectorization.SLEEFPiratesDict
    @eval @inline function LazyMap(f::typeof($f), A::AbstractMutableFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
        LazyMap{typeof($m.$sf),S,T,N,X,L}($m.$sf, pointer(A))
    end
    ff = get(Base.FastMath.fast_op, f, :null)
    if ff !== :null
        @eval @inline function LazyMap(f::typeof(Base.FastMath.$ff), A::AbstractMutableFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
            LazyMap{typeof($m.$sf),S,T,N,X,L}($m.$sf, pointer(A))
        end
    end
end
    
@inline function SIMDPirates.vload(::Type{Vec{W,T}}, m::VectorizableMap{F,T}) where {W,F,T}
    m.f(vload(Vec{W,T}, m.ptr))
end
@inline function SIMDPirates.vload(::Type{Vec{W,T}}, m::VectorizableMap{F,T}, mask::Union{<:Unsigned,Vec{W,Bool}}) where {W,F,T}
    m.f(vload(Vec{W,T}, m.ptr, mask))
end
@inline function SIMDPirates.vload(::Type{Vec{W,T}}, m::VectorizableMap{F,T}, i::Int) where {W,F,T}
    m.f(vload(Vec{W,T}, m.ptr, i * sizeof(T)))
end
@inline function SIMDPirates.vload(::Type{Vec{W,T}}, m::VectorizableMap{F,T}, i::Int, mask::Union{<:Unsigned,Vec{W,Bool}}) where {W,F,T}
    m.f(vload(Vec{W,T}, m.ptr, i * sizeof(T), mask))
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
    R = (S.parameters[1])::Int
    ex = sub2ind_expr(X.parameters)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > size(A, d)) d->ThrowBoundsError() d -> nothing
        end
        A.f(VectorizationBase.load(pointer(A) + $(sizeof(T)) * $ex ))
    end
end

@inline Base.Broadcast.materialize(sp::StackPointer, A) = (sp, A)
@inline Base.Broadcast.materialize(sp::StackPointer, A::LazyMap) = copy(sp, A)
@inline Base.Broadcast.materialize(A::LazyMap) = copy(A)



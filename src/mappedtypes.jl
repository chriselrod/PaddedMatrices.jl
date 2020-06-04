
struct MappedPairElement{F,T<:Real}
    x::T
    fx::T
    @inline MappedPairElement{F}(x::T, y::T) where {F,T} = new{F,T}(x, y)
end
Base.promote(x::MappedPairElement, y::MappedPairElement) = promote(x.x, y.x)
Base.promote(x::MappedPairElement, y) = promote(x.x, y)
Base.promote(x, y::MappedPairElement) = promote(x, y.x)
Base.:+(x::MappedPairElement, y::MappedPairElement) = Base.FastMath.add_fast(x.x, y.x)
Base.:-(x::MappedPairElement, y::MappedPairElement) = Base.FastMath.sub_fast(x.x, y.x)
Base.:*(x::MappedPairElement, y::MappedPairElement) = Base.FastMath.mul_fast(x.x, y.x)
Base.:/(x::MappedPairElement, y::MappedPairElement) = Base.FastMath.div_fast(x.x, y.x)
Base.:+(x::MappedPairElement, y::Number) = Base.FastMath.add_fast(x.x, y)
Base.:-(x::MappedPairElement, y::Number) = Base.FastMath.sub_fast(x.x, y)
Base.:*(x::MappedPairElement, y::Number) = Base.FastMath.mul_fast(x.x, y)
Base.:/(x::MappedPairElement, y::Number) = Base.FastMath.div_fast(x.x, y)
Base.:+(x::Number, y::MappedPairElement) = Base.FastMath.add_fast(x, y.x)
Base.:-(x::Number, y::MappedPairElement) = Base.FastMath.sub_fast(x, y.x)
Base.:*(x::Number, y::MappedPairElement) = Base.FastMath.mul_fast(x, y.x)
Base.:/(x::Number, y::MappedPairElement) = Base.FastMath.div_fast(x, y.x)

Base.:+(x::MappedPairElement{typeof(exp)}, y::MappedPairElement{typeof(exp)}) = MappedPairElement{typeof(exp)}(Base.FastMath.add_fast(x.x, y.x), Base.FastMath.mul_fast(x.fx, y.fx))
Base.:*(x::MappedPairElement{typeof(log)}, y::MappedPairElement{typeof(log)}) = MappedPairElement{typeof(log)}(Base.FastMath.mul_fast(x.x, y.x), Base.FastMath.add_fast(x.fx, y.fx))

struct MappedPairVec{F<:Function,W,T} <: AbstractStructVec{W,T}
    data::NTuple{W,Core.VecElement{T}}
    fdata::NTuple{W,Core.VecElement{T}}
    @inline MappedPairVec{F}(x::Vec{W,T}, y::Vec{W,T}) where {F,W,T} = new{F,W,T}(x, y)
end
SIMDPirates.promote_vtype(::Type{<:MappedPairVec{W,F}}, ::Type{<:AbstractSIMDVector{W}}) where {W,F} = SVec{W,F}
SIMDPirates.promote_vtype(::Type{<:AbstractSIMDVector{W}}, ::Type{<:MappedPairVec{W,F}}) where {W,F} = SVec{W,F}
SIMDPirates.promote_vtype(::Type{<:MappedPairVec{W,F}}, ::Type{<:MappedPairVec{W,F}}) where {W,F} = MappedPairVec{W,F}
@inline VectorizationBase.extract_data(v::MappedPairVec) = v.data
@inline MappedPairVec{F}(x::AbstractStructVec, y::AbstractStructVec) where {F} = MappedPairVec{F}(extract_data(x), extract_data(y))
@inline mapped(::Type{F}, x::T, fx::T) where {F,T<:Number} = MappedPairElement{F,T}(x, fx)
@inline mapped(::Type{F}, x::VectorizationBase._Vec{W,T}, fx::VectorizationBase._Vec{W,T}) where {F,W,T} = MappedPairVec{F}(x, fx)
@inline mapped(::Type{F}, x::AbstractStructVec{W,T}, fx::AbstractStructVec{W,T}) where {F,W,T} = MappedPairVec{F,W,T}(extract_data(x), extract_data(fx))
@inline Base.:+(x::MappedPairVec{typeof(exp)}, y::MappedPairVec{typeof(exp)}) = MappedPairVec{typeof(exp)}(vadd(x.x, y.x), vmul(x.fx, y.fx))
@inline Base.:*(x::MappedPairVec{typeof(log),W}, y::MappedPairVec{typeof(log)}) where {W} = MappedPairVec{typeof(log)}(vmul(x.x, y.x), vadd(x.fx, y.fx))

# Storring into the mapped part of a MappedPairArray is not allowed.
# To maintin the mapping, storing into it at all isn't allowed.
# Although one can take the Ptr.
struct MappedPairArray{F<:Function,S,T,N,X,SN,XN,V} <: AbstractStrideArray{S,T,N,X,SN,XN,V}
    ptr::Ptr{T}
    mappedptr::Ptr{T}
    size::NTuple{SN,UInt32}
    stride::NTuple{XN,UInt32}
end
@inline function MappedPairArray(::F, A::PtrArray{S,T,N,X,SN,XN,V}, mp::Ptr{T}) where {F,S,T,N,X,SN,XN,V}
    MappedPairArray{F,S,T,N,X,SN,XN,V}(pointer(A), mp, A.size, A.stride)
end
@inline function MappedPairArray(::F, A::PtrArray{S,T,N,X,0,0,V}, mp::Ptr{T}) where {F,S,T,N,X,V}
    MappedPairArray{F,S,T,N,X,0,0,V}(pointer(A), mp, tuple(), tuple())
end
@inline Base.broadcasted(::F, A::MappedPairArray{F,S,T,N,X,SN,XN,V}) where {F,S,T,N,X,SN,XN,V} = PtrArray{S,T,N,X,SN,XN,V}(A.mappedptr, A.size, A.stride)
@inline Base.pointer(A::MappedPairArray) = A.ptr

struct MappedPairStridedPointer{F<:Function,T,P<:AbstractStridedPointer{T}} <: AbstractStridedPointer{T}
    ptr::P
    mappedptr::Ptr{T}
    @inline MappedPairStridedPointer{F}(p::P, mp::Ptr{T}) where {F,T,P<:AbstractStridedPointer} = new{F,T,P}(p, mp)
end
@inline Base.pointer(A::MappedPairStridedPointer) = A.ptr.ptr
@inline VectorizationBase.offset(p::MappedPairStridedPointer, i...) = offset(p.ptr, i...)
@inline function VectorizationBase.stridedpointer(A::MappedPairStridedPointer{F}) where {F}
    MappedPairStridedPointer{F}(stridedpointer(PtrArray(A)), A.mappedptr)
end

@inline function VectorizationBase.vload(mpp::MappedPairStridedPointer{F}, i::Tuple) where {F}
    p = mpp.ptr
    o = offset(p, VectorizationBase.staticm1(i))
    mp = mpp.mappedptr
    mapped(F, vload(p.ptr, o), vload(mp, o))
end
@inline function VectorizationBase.vload(::Val{W}, mpp::MappedPairStridedPointer{F,T}, i::Tuple) where {W,F,T}
    p = mpp.ptr
    o = offset(p, VectorizationBase.staticm1(i))
    mp = mpp.mappedptr
    mapped(F, vload(Vec{W,T}, p.ptr, o), vload(Vec{W,T}, mp, o))
end
@inline function VectorizationBase.vload(::Type{Vec{W,T}}, mpp::MappedPairStridedPointer{F,T}, i::Tuple) where {W,F,T}
    p = mpp.ptr
    o = offset(p, VectorizationBase.staticm1(i))
    mp = mpp.mappedptr
    mapped(F, vload(Vec{W,T}, p.ptr, o), vload(Vec{W,T}, mp, o))
end

for (m,f,f⁻¹) ∈ [(:Base,:log,:exp), (:Base,:abs2,:sqrt), (:SLEEFPirates,:nlogit,:ninvlogit), (:SLEEFPirates,:logit,:invlogit)]
    for T ∈ [:MappedPairElement, :MappedPairVec]
        @eval @inline $m.$f(x::$T{typeof($f)}) = $T{typeof($f⁻¹)}(x.fx, x.x)
        @eval @inline $m.$f⁻¹(x::$T{typeof($f⁻¹)}) = $T{typeof($f)}(x.fx, x.x)
        @eval @inline Base.broadcasted(::typeof($f), x::$T{typeof($f)}) = $T{$f⁻¹}(A.fx, x.x)
        @eval @inline Base.broadcasted(::typeof($f⁻¹), x::$T{typeof($f⁻¹)}) = $T{$f}(A.fx, x.d)
    end
    @eval @inline (::typeof($f))(A::MappedPairArray{typeof($f),S,T,N,X,SN,XN,V}) where {S,T,N,X,SN,XN,V} = MappedPairArray{typeof($f⁻¹),S,T,N,X,SN,XN,V,L}(A.mappedptr, A.ptr, A.size, A.stride)
    @eval @inline (::typeof($f⁻¹))(A::MappedPairArray{typeof($f⁻¹),S,T,N,X,SN,XN,V}) where {S,T,N,X,SN,XN,V} = MappedPairArray{typeof($f),S,T,N,X,SN,XN,V,L}(A.mappedptr, A.ptr, A.size, A.stride)
    @eval @inline Base.broadcasted(::typeof($f), A::MappedPairArray{typeof($f),S,T,N,X,SN,XN,V}) where {S,T,N,X,SN,XN,V} = MappedPairArray{typeof($f⁻¹),S,T,N,X,SN,XN,V,L}(A.mappedptr, A.ptr, A.size, A.stride)
    @eval @inline Base.broadcasted(::typeof($f⁻¹), A::MappedPairArray{typeof($f⁻¹),S,T,N,X,SN,XN,V}) where {S,T,N,X,SN,XN,V} = MappedPairArray{typeof($f),S,T,N,X,SN,XN,V,L}(A.mappedptr, A.ptr, A.size, A.stride)
end









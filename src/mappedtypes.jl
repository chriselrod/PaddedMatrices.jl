
struct MappedElement{F,T<:Real} <: Real
    x::T
    fx::T
    @inline MappedElement{F}(x::T, y::T) where {F,T} = new{F,T}(x, y)
end
Base.promote(x::MappedElement, y::MappedElement) = promote(x.x, y.x)
Base.promote(x::MappedElement, y) = promote(x.x, y)
Base.promote(x, y::MappedElement) = promote(x, y.x)
Base.:+(x::MappedElement, y::MappedElement) = Base.FastMath.add_fast(x.x, y.x)
Base.:-(x::MappedElement, y::MappedElement) = Base.FastMath.sub_fast(x.x, y.x)
Base.:*(x::MappedElement, y::MappedElement) = Base.FastMath.mul_fast(x.x, y.x)
Base.:/(x::MappedElement, y::MappedElement) = Base.FastMath.div_fast(x.x, y.x)
Base.:+(x::MappedElement, y::Number) = Base.FastMath.add_fast(x.x, y)
Base.:-(x::MappedElement, y::Number) = Base.FastMath.sub_fast(x.x, y)
Base.:*(x::MappedElement, y::Number) = Base.FastMath.mul_fast(x.x, y)
Base.:/(x::MappedElement, y::Number) = Base.FastMath.div_fast(x.x, y)
Base.:+(x::Number, y::MappedElement) = Base.FastMath.add_fast(x, y.x)
Base.:-(x::Number, y::MappedElement) = Base.FastMath.sub_fast(x, y.x)
Base.:*(x::Number, y::MappedElement) = Base.FastMath.mul_fast(x, y.x)
Base.:/(x::Number, y::MappedElement) = Base.FastMath.div_fast(x, y.x)

Base.:+(x::MappedElement{typeof(exp)}, y::MappedElement{typeof{exp}}) = MappedElement{typeof(exp)}(Base.FastMath.add_fast(x.x, y.x), Base.FastMath.mul_fast(x.fx, y.fx))
Base.:*(x::MappedElement{typeof(log)}, y::MappedElement{typeof{log}}) = MappedElement{typeof(log)}(Base.FastMath.mul_fast(x.x, y.x), Base.FastMath.add_fast(x.fx, y.fx))

struct MappedVec{F<:Function,W,T} <: VectorizationBase.AbstractStructVec{W,T}
    data::NTuple{W,Core.VecElement{T}}
    fdata::NTuple{W,Core.VecElement{T}}
    @inline MappedVec{F}(x::Vec{W,T}, y::Vec{W,T}) where {F,W,T} = new{F,W,T}(x, y)
end
@inline VectorizationBase.extract_data(v::MappedVec) = v.data
@inline MappedVec{F}(x::AbstractStructVec, y::AbstractStructVec) where {F} = MappedVec{F}(extract_data(x), extract_data(y))
@inline mapped(::Type{F}, x::T, fx::T) where {F,T<:Number} = MappedElement{F,T}(x, fx)
@inline mapped(::Type{F}, x::Vec{W,T}, fx::Vec{W,T}) where {F,W,T} = MappedVec{F,W,T}(x, fx)
@inline mapped(::Type{F}, x::AbstractStructVec{W,T}, fx::AbstractStructVec{W,T}) where {F,W,T} = MappedVec{F,W,T}(extract_data(x), extract_data(fx))
@inline Base.:+(x::MappedVec{typeof(exp)}, y::MappedVec{typeof{exp}}) = MappedVec{typeof(exp)}(vadd(x.x, y.x), vmul(x.fx, y.fx))
@inline Base.:*(x::MappedVec{typeof(log),W}, y::MappedVec{typeof{log},W}) where {W} = MappedVec{typeof(log)}(vmul(x.x, y.x), vadd(x.fx, y.fx))

# Storring into the mapped part of a MappedArray is not allowed.
# To maintin the mapping, storing into it at all isn't allowed.
# Although one can take the Ptr.
struct MappedArray{F<:Function,S,T,N,X,SN,XN,V,L} <: AbstractStrideArray{S,T,N,X,SN,XN,V,L}
    ptr::Ptr{T}
    mappedptr::Ptr{T}
    size::NTuple{SN,UInt32}
    stride::NTuple{XN,UInt32}
end
@inline function MappedArray(::F, A::PtrArray{S,T,N,X,SN,XN,V,L}, mp::Ptr{T}) where {F,S,T,N,X,SN,XN,V,L}
    MappedArray{F,S,T,N,X,SN,XN,V,L}(pointer(A), mp, A.size, A.stride)
end
@inline function MappedArray(::F, A::PtrArray{S,T,N,X,0,0,V,L}, mp::Ptr{T}) where {F,S,T,N,X,V,L}
    MappedArray{F,S,T,N,X,0,0,V,L}(pointer(A), mp, tuple(), tuple()
end
@inline Base.broadcasted(::F, A::MappedArray{F,S,T,N,X,SN,XN,V,L}) where {F,S,T,N,X,SN,XN,V,L} = PtrArray{S,T,N,X,SN,XN,V,L}(A.mappedptr, A.size, A.stride)
@inline Base.pointer(A::MappedArray) = A.ptr

struct MappedStridedPointer{F<:Function,T,P<:AbstractStridedPointer{T}} <: AbstractStridedPointer{T}
    ptr::P
    mappedptr::Ptr{T}
    @inline MappedStridedPointer{F}(p::P, mp::Ptr{T}) where {T,P<:AbstractStridedPointer} = new{F,T,P}(p, mp)
end
@inline Base.pointer(A::MappedStridedPointer) = A.ptr.ptr
@inline VectorizationBase.offset(p::MappedStridedPointer, i...) = offset(p.ptr, i...)
@inline function VectorizationBase.stridedpointer(A::MappedStridedPointer{F}) where {F}
    MappedStridedPointer{F}(stridedpointer(PtrArray(A)), A.mappedptr)
end

@inline function VectorizationBase.load(mpp::MappedStridedPointer{F}, i::Tuple) where {F}
    p = mpp.ptr
    o = offset(p, i...)
    mp = mpp.mappedptr
    MappedElement{F}(load(p.ptr, o), load(mp, o))
end
@inline function VectorizationBase.vload(::Val{W}, mpp::MappedStridedPointer{F,T}, i...) where {W,F,T}
    p = mpp.ptr
    o = offset(p, i...)
    mp = mpp.mappedptr
    MappedVec{F}(vload(Vec{W,T}, p.ptr, o), vload(Vec{W,T}, mp, o))
end
@inline function VectorizationBase.vload(::Type{Vec{W,T}}, mpp::MappedStridedPointer{F,T}, i...) where {W,F,T}
    p = mpp.ptr
    o = offset(p, i...)
    mp = mpp.mappedptr
    MappedVec{F}(vload(Vec{W,T}, p.ptr, o), vload(Vec{W,T}, mp, o))
end

for (m,f,f⁻¹) ∈ [(:Base,:log,:exp), (:Base,:abs2,:sqrt), (:SLEEFPirates,:nlogit,:ninvlogit), (:SLEEFPirates,:logit,:invlogit)]
    for T ∈ [:MappedElement, :MappedVec]
        @eval @inline $m.$f(x::$T{typeof($f)}) = $T{typeof($f⁻¹)}(x.fx, x.x)
        @eval @inline $m.$f⁻¹(x::$T{typeof($f⁻¹)}) = $T{typeof($f)}(x.fx, x.x)
        @eval @inline Base.broadcasted(::$f, x::$T{$f}) = $T{$f⁻¹}(A.fx, x.x)
        @eval @inline Base.broadcasted(::$f⁻¹, x::$T{$f⁻¹}) = $T{$f}(A.fx, x.d)
    end
    @eval @inline (::typeof($f))(A::MappedArray{$f,S,T,N,X,SN,XN,V,L}) where {S,T,N,X,SN,XN,V,L} = MappedArray{$f⁻¹,S,T,N,X,SN,XN,V,L}(A.mappedptr, A.ptr, A.size, A.stride)
    @eval @inline (::typeof($f⁻¹)(A::MappedArray{$f⁻¹,S,T,N,X,SN,XN,V,L}) where {S,T,N,X,SN,XN,V,L} = MappedArray{$f,S,T,N,X,SN,XN,V,L}(A.mappedptr, A.ptr, A.size, A.stride)
    @eval @inline Base.broadcasted(::$f, A::MappedArray{$f,S,T,N,X,SN,XN,V,L}) where {S,T,N,X,SN,XN,V,L} = MappedArray{$f⁻¹,S,T,N,X,SN,XN,V,L}(A.mappedptr, A.ptr, A.size, A.stride)
    @eval @inline Base.broadcasted(::$f⁻¹, A::MappedArray{$f⁻¹,S,T,N,X,SN,XN,V,L}) where {S,T,N,X,SN,XN,V,L} = MappedArray{$f,S,T,N,X,SN,XN,V,L}(A.mappedptr, A.ptr, A.size, A.stride)
end









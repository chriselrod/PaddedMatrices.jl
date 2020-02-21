
# abstract type AbstractVecStridedPointer
# Only aligned loads (along vector-width boundaries) are legal.
struct VecStridedPointer{T,X,L} <: VectorizationBase.AbstractStaticStridedPointer{T,X}
    ptr::NTuple{L,Core.VecElement{T}}
    offset::UInt32
end
@inline VectorizationBase.offset(ptr::VecStridedPointer, i::Integer) = (i % UInt32) + ptr.offset
@inline VectorizationBase.gep(ptr::VecStridedPointer{T,X,L}, i::Integer) where {T,X,L} = VecStridedPointer{T,X,L}(ptr.ptr, (i % UInt32) + ptr.offset)
@inline VecStridedPointer{X}(ptr::NTuple{L,Core.VecElement{T}}) where {T,X,L} = VecStridedPointer{T,X,L}(ptr, 0x00000000)
@inline VectorizationBase.load(ptr::VecStridedPointer) = @inbounds ptr.ptr[1 + ptr.offset].value
@inline function SIMDPirates.vload(::Val{W}, ptr::VecStridedPointer{T,X,L}) where {W,T,X,L}
    SVec(ntuple(Val(W)) do w @inbounds ptr.ptr[w+ptr.offset].value end)
end
@inline function SIMDPirates.vload(::Val{W}, ptr::VecStridedPointer{T,X,L}, ::Unsigned) where {W,T,X,L}
    SVec(ntuple(Val(W)) do w @inbounds ptr.ptr[w+ptr.offset].value end)
end

@inline Base.pointer(A::StrideArray) = pointer(A.data)
@inline Base.pointer(A::FixedSizeArray{S,T}) where {S,T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@inline Base.pointer(A::ConstantArray) = stridedpointer(flatvector(A))
@inline Base.pointer(A::PtrArray) = A.data

@inline VectorizationBase.stridedpointer(A::AbstractStrideArray{S,T,N,X,SN,0}) where {S,T,N,X,SN} = VectorizationBase.StaticStridedPointer{T,X}(pointer(A))
@inline VectorizationBase.stridedpointer(A::AbstractStrideArray{S,T,N,X,SN,0}) where {S,T,N,X<:Tuple{1,Vararg},SN} = VectorizationBase.StaticStridedPointer{T,X}(pointer(A))
@inline VectorizationBase.stridedpointer(A::ConstantArray{S,T,N,X,L}) where {S,T,N,X,L} = VectorizationBase.VecStridedPointer{T,X,L}(A.data, 0x00000000))

@inline VectorizationBase.stridedpointer(A::AbstractStrideArray{S,T,N,<:Tuple{1,Vararg}}) where {S,T,N} = VectorizationBase.PackedStridedPointer{T,N}(pointer(A), tailstrides(A))
@generated function VectorizationBase.stridedpointer(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X}
    Xv = tointvec(X.parameters)
    ret = if last(Xv) == 1
        :(VectorizationBase.RowMajorStridedPointer{$T,$N}(pointer(A), revtailstrides(A)))
    # elseif first(Xv) == 1
        # :(VectorizationBase.PackedStridedPointer{$T,$N}(pointer(A), tailstrides(A)))
    else
        :(VectorizationBase.SparseStridedPointer{$T,$N}(pointer(A), strides(A)))
    end
    Expr(:block, Expr(:meta, :inline), ret)
end


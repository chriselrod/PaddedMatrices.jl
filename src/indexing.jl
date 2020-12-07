



@noinline ThrowBoundsError(A, i) = (println("A of length $(length(A))."); throw(BoundsError(A, i)))
                                
Base.IndexStyle(::Type{<:AbstractStrideArray}) = IndexCartesian()
Base.IndexStyle(::Type{<:AbstractStrideVector{<:Any,<:Any,<:Any,1}}) = IndexLinear()
@generated function Base.IndexStyle(::Type{<:AbstractStrideArray{S,D,T,N,C,B,R}}) where {S,D,T,N,C,B,R}
    # if is column major || is a transposed contiguous vector
    if all(D) && ((isone(C) && R === ntuple(identity, Val(N))) || (C === 2 && R === (2,1) && S <: Tuple{One,Integer}))
        :(IndexLinear())
    else
        :(IndexCartesian())
    end          
end

@inline function Base.getindex(A::PtrArray, i::Vararg{Integer,K}) where {K}
    @boundscheck checkbounds(A, i...)
    vload(stridedpointer(A), i)
end
@inline function Base.getindex(A::AbstractStrideArray, i::Vararg{Integer,K}) where {K}
    @boundscheck checkbounds(A, i...)
    GC.@preserve A vload(stridedpointer(A), i)
end
Base.@propagate_inbounds Base.getindex(A::AbstractStrideVector, i::Int, j::Int) = A[i]
@inline function Base.setindex!(A::PtrArray, v, i::Vararg{Integer,K}) where {K}
    @boundscheck checkbounds(A, i...)
    vstore!(stridedpointer(A), v, i)
    v
end
@inline function Base.setindex!(A::AbstractStrideArray, v, i::Vararg{Integer,K}) where {K}
    @boundscheck checkbounds(A, i...)
    GC.@preserve A vstore!(stridedpointer(A), v, i)
    v
end


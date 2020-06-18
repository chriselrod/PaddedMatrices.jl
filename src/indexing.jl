@noinline ThrowBoundsError(A, i) = (println("A of length $(length(A))."); throw(BoundsError(A, i)))
                                
Base.IndexStyle(::Type{<:AbstractStrideArray}) = IndexCartesian()
Base.IndexStyle(::Type{<:AbstractStrideVector{<:Any,<:Any,1}}) = IndexLinear()
@generated function Base.IndexStyle(::Type{<:AbstractStrideArray{S,T,N,X}}) where {S,T,N,X}
    x = 1
    for n ∈ 1:N
        Xₙ = (X.parameters[n])::Int
        x == Xₙ || return IndexCartesian()
        Sₙ = (S.parameters[n])::Int
        Sₙ == -1 && return IndexCartesian()
        x *= Sₙ
    end
    IndexLinear()
end

@inline function Base.getindex(
    A::AbstractStrideArray{S,T,1}, i::NTuple{2,<:Integer}
) where {S,T,N}
    @boundscheck begin
        first(i) > length(A) && ThrowBoundsError(A, i)
    end
    GC.@preserve A vload(stridedpointer(A), (vsub(first(i), 1),))
end
@inline function Base.getindex(
    A::AbstractPtrStrideArray{S,T,1}, i::NTuple{2,<:Integer}
) where {S,T,N}
    @boundscheck begin
        first(i) > length(A) && ThrowBoundsError(A, i)
    end
    GC.@preserve A vload(stridedpointer(A), (vsub(first(i), 1),))
end
@inline function Base.getindex(
    A::AbstractStrideArray{S,T,N}, i::NTuple{N,<:Integer}
) where {S,T,N}
    @boundscheck begin
        any(i .> size(A)) && ThrowBoundsError(A, i)
    end
    GC.@preserve A vload(stridedpointer(A), staticm1(i))
end
Base.@propagate_inbounds Base.getindex(A::AbstractStrideArray, i::CartesianIndex) = getindex(A, i.I)
Base.@propagate_inbounds function Base.getindex(A::AbstractStrideArray, i::Vararg{<:Number})
    getindex(A, i)
end
Base.@propagate_inbounds Base.getindex(A::AbstractPtrStrideArray{S,T,N}, i::Vararg{<:Number,N}) where {S,T,N} = getindex(A, i)
@inline function Base.getindex(A::AbstractPtrStrideArray{S,T}, i::Integer) where {S,T}
    @boundscheck i > length(A) && ThrowBoundsError(A, i)
    GC.@preserve A vload(pointer(A), vmul(sizeof(T), vsub(i, 1)))
end
@inline function Base.getindex(A::AbstractStrideArray{S,T}, i::Integer) where {S,T}
    @boundscheck i > length(A) && ThrowBoundsError(A, i)
    GC.@preserve A vload(pointer(A), vmul(sizeof(T), vsub(i, 1)))
end
@inline function Base.getindex(A::AbstractStrideArray{S,T,1}, i::Integer) where {S,T}
    @boundscheck i > length(A) && ThrowBoundsError(A, i)
    GC.@preserve A vload(pointer(A), vmul(sizeof(T), vsub(i, 1)))
end

@inline function Base.setindex!(
    A::AbstractStrideArray{S,T,N}, v, i::NTuple{N,<:Integer}
) where {S,T,N}
    @boundscheck begin
        any(i .> size(A)) && ThrowBoundsError(A, i)
    end
   # vstore!(stridedpointer(A), v, i)
    GC.@preserve A vnoaliasstore!(stridedpointer(A), v, staticm1(i))
end
Base.@propagate_inbounds Base.setindex!(A::AbstractStrideArray, v, i::CartesianIndex) = setindex!(A, v, i.I)

Base.@propagate_inbounds function Base.setindex!(
    A::AbstractStrideArray{S,T,N}, v, i::Vararg{<:Integer,N}
) where {S,T,N}
    setindex!(A, v, i)
end

@inline function Base.setindex!(A::AbstractStrideArray{S,T}, v, i::Integer) where {S,T}
    @boundscheck i > length(A) && ThrowBoundsError(A, i)
   # vstore!(pointer(A), v, i - 1)
    GC.@preserve A vnoaliasstore!(pointer(A), v, vmul(sizeof(T), vsub(i, 1)))
end



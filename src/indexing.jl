@noinline ThrowBoundsError(A, i) = throw(BoundsError(A, i))
                                
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
    vload(stridedpointer(A), (first(i),))
end
@inline function Base.getindex(
    A::AbstractPtrStrideArray{S,T,1}, i::NTuple{2,<:Integer}
) where {S,T,N}
    @boundscheck begin
        first(i) > length(A) && ThrowBoundsError(A, i)
    end
    vload(stridedpointer(A), (first(i),))
end
@inline function Base.getindex(
    A::AbstractStrideArray{S,T,N}, i::NTuple{N,<:Integer}
) where {S,T,N}
    @boundscheck begin
        any(i .> size(A)) && ThrowBoundsError(A, i)
    end
    vload(stridedpointer(A), i)
end
Base.@propagate_inbounds Base.getindex(A::AbstractStrideArray, i::CartesianIndex) = @show getindex(A, i.I)
Base.@propagate_inbounds function Base.getindex(A::AbstractStrideArray, i::Vararg{<:Number})
    getindex(A, i)
end
Base.@propagate_inbounds Base.getindex(A::AbstractPtrStrideArray, i::Vararg{<:Number}) = getindex(A, i)
@inline function Base.getindex(A::AbstractStrideArray, i::Integer)
    @boundscheck i > length(A) && ThrowBoundsError(A, i)
    vload(pointer(A), i - 1)
end
@inline function Base.getindex(A::AbstractStrideArray{S,T,1}, i::Integer) where {S,T}
    @boundscheck i > length(A) && ThrowBoundsError(A, i)
    vload(pointer(A), i - 1)
end

@inline function Base.setindex!(
    A::AbstractStrideArray{S,T,N}, v, i::NTuple{N,<:Integer}
) where {S,T,N}
    @boundscheck begin
        any(i .> size(A)) && ThrowBoundsError(A, i)
    end
    vstore!(stridedpointer(A), v, i)
end
Base.@propagate_inbounds Base.setindex!(A::AbstractStrideArray, v, i::CartesianIndex) = setindex!(A, v, i.I)

Base.@propagate_inbounds function Base.setindex!(
    A::AbstractStrideArray{S,T,N}, v, i::Vararg{<:Integer,N}
) where {S,T,N}
    setindex!(A, v, i)
end

@inline function Base.setindex!(A::AbstractStrideArray, v, i::Integer)
    @boundscheck i > length(A) && ThrowBoundsError(A, i)
    vstore!(pointer(A), v, i - 1)
end



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

@inline function Base.getindex(A::AbstractStrideArray, i::Tuple)
    @boundscheck begin
        any(i .> size(A)) && ThrowBoundsError(A, i)
    end
    vload(stridedpointer(A), i)
end
@inline Base.getindex(A::AbstractStrideArray, i::CartesianIndex) = getindex(A, i.I)
@inline Base.getindex(A::AbstractStrideArray, i::Vararg{<:Number}) = getindex(A, i)
@inline Base.getindex(A::AbstractPtrStrideArray, i::Vararg{<:Number}) = getindex(A, i)
@inline function Base.getindex(A::AbstractStrideArray, i::Integer)
    @boundscheck i > length(A) && ThrowBoundsError(A, i)
    vload(pointer(A), i - 1)
end

@inline function Base.setindex!(A::AbstractStrideArray, v, i::Tuple)
    @boundscheck begin
        any(i .> size(A)) && ThrowBoundsError(A, i)
    end
    vstore!(stridedpointer(A), v, i)
end
@inline Base.setindex!(A::AbstractStrideArray, v, i::CartesianIndex) = setindex!(A, v, i.I)
@inline Base.setindex!(A::AbstractStrideArray, v, i...) = setindex!(A, v, i)
@inline function Base.setindex!(A::AbstractStrideArray, v, i::Integer)
    @boundscheck i > length(A) && ThrowBoundsError(A, i)
    vstore!(pointer(A), v, i - 1)
end



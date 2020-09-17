@noinline ThrowBoundsError(A, i) = (println("A of length $(length(A))."); throw(BoundsError(A, i)))

function isoneto(x::NTuple{N,Int}) where {N}
    y = ntuple(identity, Val{N}())
    x === y
end

Base.IndexStyle(::Type{<:AbstractStrideArray{S,T,N,X,C,B,O,D}}) where {S,T,N,X,C,B,O,D} = (all(isequal(dense), D) && isoneto(O)) ? IndexLinear() : IndexCartesian()
Base.IndexStyle(::Type{<:AbstractStrideVector{<:Any,<:Any,1}}) = IndexLinear()
# @generated function Base.IndexStyle(::Type{<:AbstractStrideArray{S,T,N,X}}) where {S,T,N,X}
#     x = 1
#     for n ∈ 1:N
#         Xₙ = (X.parameters[n])::Int
#         x == Xₙ || return IndexCartesian()
#         Sₙ = (S.parameters[n])::Int
#         Sₙ == -1 && return IndexCartesian()
#         x *= Sₙ
#     end
#     IndexLinear()
# end

# Avoid method ambiguities
for AT ∈ [:AbstractStrideArray, :AbstractPtrStrideArray, :StrideArray, :FixedSizeArray]
    loadexprs = :(vload(stridedpointer(A), (vsub(i, 1),)))
    loadexprt = :(vload(stridedpointer(A), staticm1(i)))
    loadexprv = :(vload(stridedpointer(A), (vsub(first(i), 1),)))
    if AT !== :AbstractPtrStrideArray
        loadexprs = :(GC.@preserve A $loadexprs)
        loadexprt = :(GC.@preserve A $loadexprt)
        loadexprv = :(GC.@preserve A $loadexprv)
    end
    @eval begin
        @inline function Base.getindex(
            A::$AT{S,T,1}, i::NTuple{2,<:Integer}
        ) where {S,T}
            @boundscheck begin
                checkbounds(A, first(i))
            end
            $loadexprv
        end

        @inline function Base.getindex(
            A::$AT{S,T,N}, i::NTuple{N,<:Integer}
        ) where {S,T,N}
            @boundscheck begin
                checkbounds(A, i...)
            end
            $loadexprt
        end
        Base.@propagate_inbounds Base.getindex(A::$AT, i::Vararg{<:Integer}) = getindex(A, i)
        @inline function Base.getindex(A::$AT{S,T}, i::Integer) where {S,T}
            @boundscheck checkbounds(A, i)
            $loadexprs
        end
        @inline function Base.getindex(A::$AT{S,T,1}, i::Integer) where {S,T}
            @boundscheck checkbounds(A, i)
            $loadexprs
        end

    end
end
Base.@propagate_inbounds Base.getindex(A::AbstractStrideArray, i::CartesianIndex) = getindex(A, i.I)

@inline function Base.setindex!(
    A::AbstractStrideArray{S,T,N}, v, i::NTuple{N,<:Integer}
) where {S,T,N}
    @boundscheck begin
        checkbounds(A, i...)
    end
    GC.@preserve A vnoaliasstore!(stridedpointer(A), v, staticm1(i))
end

@inline function Base.setindex!(
    A::AbstractPtrStrideArray{S,T,N}, v, i::NTuple{N,<:Integer}
) where {S,T,N}
    @boundscheck begin
        checkbounds(A, i...)
    end
    vnoaliasstore!(stridedpointer(A), v, staticm1(i))
end
Base.@propagate_inbounds Base.setindex!(A::AbstractStrideArray, v, i::CartesianIndex) = setindex!(A, v, i.I)

Base.@propagate_inbounds function Base.setindex!(
    A::AbstractStrideArray{S,T,N}, v, i::Vararg{<:Integer,N}
) where {S,T,N}
    setindex!(A, v, i)
end

@inline function Base.setindex!(A::AbstractStrideArray{S,T}, v, i::Integer) where {S,T}
    @boundscheck checkbounds(A, i)
    GC.@preserve A vnoaliasstore!(pointer(A), v, vmul(sizeof(T), vsub(i, 1)))
end

@inline function Base.setindex!(A::AbstractPtrStrideArray{S,T}, v, i::Integer) where {S,T}
    @boundscheck checkbounds(A, i)
    vnoaliasstore!(pointer(A), v, vmul(sizeof(T), vsub(i, 1)))
end



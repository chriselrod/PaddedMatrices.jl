function inddimprod(X, i, xi, A::Symbol = :A)
    Xᵢ = (X[i])::Int
    if Xᵢ == 1
        ind = :(i[$i] - 1)
    elseif Xᵢ == -1
        ind = :((i[$i] - 1) * ($A.size[$xi] % Int) )
        xi += 1
    else
        ind = :((i[$i] - 1) * $Xᵢ )
    end
    ind, xi
end

"""
Converts Cartesian one-based index into linear one-based index.
Just subtract 1 for a zero based index.
"""
@noinline function sub2ind_expr(X::Core.SimpleVector, A::Symbol = :A)#, N::Int = length(X))
    N = length(X)
    ind1, x1 = inddimprod(X, 1, 1, A)
    if N == 1
        inds = ind1
    else
        inds = Expr(:call, :+, ind1)
        for n ∈ 2:N
            indi, x1 = inddimprod(X, n, x1, A)
            push!(inds.args, indi)
        end
    end
    Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,@__FILE__), inds)
end
@generated function sub2ind(A::AbstractStrideArray{S,T,N,X}, i::NTuple{N}) where {S,T,N,X}
    sub2ind_expr(X.parameters, :A)
end

function calc_padding(nrow::Int, T)
    W = VectorizationBase.pick_vector_width(T)
    W > nrow ? VectorizationBase.nextpow2(nrow) : VectorizationBase.align(nrow, T)
end

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
        any(size(A) .> i) && ThrowBoundsError(A, i)
    end
    load(stridedpointer(A), i .- 1)
end
@inline Base.getindex(A::AbstractStrideArray, i::CartesianIndex) = getindex(A, i.I)
@inline Base.getindex(A::AbstractStrideArray, i...) = getindex(A, i)
@inline function Base.getindex(A::AbstractStrideArray, i::Integer)
    @boundscheck i > length(A) && ThrowBoundsError(A, i)
    load(pointer(A), i - 1)
end

@inline function Base.setindex!(A::AbstractStrideArray, v, i::Tuple)
    @boundscheck begin
        any(size(A) .> i) && ThrowBoundsError(A, i)
    end
    store!(stridedpointer(A), v, i .- 1)
end
@inline Base.setindex!(A::AbstractStrideArray, v, i::CartesianIndex) = setindex!(A, v, i.I)
@inline Base.setindex!(A::AbstractStrideArray, v, i...) = setindex!(A, v, i)
@inline function Base.setindex!(A::AbstractStrideArray, v, i::Integer)
    @boundscheck i > length(A) && ThrowBoundsError(A, i)
    store!(pointer(A), v, i - 1)
end



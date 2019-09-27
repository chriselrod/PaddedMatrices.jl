
struct StaticUnitRange{L,S} <: AbstractFixedSizePaddedVector{L,Int,L} end
Base.getindex(::StaticUnitRange{L,S}, i::Integer) where {L,S} = Int(i+S-1)
Base.size(::StaticUnitRange{L}) where {L} = (L,)
Base.length(::StaticUnitRange{L}) where {L} = L
Base.IndexStyle(::Type{<:StaticUnitRange}) = Base.IndexLinear()
@generated StaticUnitRange(::Val{Start}, ::Val{Stop}) where {Start,Stop} = StaticUnitRange{Stop-Start+1,Start}()
macro StaticRange(rq)
    @assert rq.head == :call
    @assert rq.args[1] == :(:)
    :(StaticUnitRange(Val{$(rq.args[2])}(), Val{$(rq.args[3])}()))
end


@generated function Base.getindex(A::AbstractMutableFixedSizePaddedArray{SV,T,N,R}, I...) where {SV,T,N,R}
    S = Int[S.parameters...]
    offset = 0
    stride = 1
    dims = Int[]
    for (i, TT) in enumerate(I)
        if TT <: Integer

        elseif TT <: StaticUnitRange
            L, U = first(TT.parameters)::Int, last(TT.parameters)::Int - 1
            push!(dims, U - L)
            offset += stride * L
        elseif TT === Colon
            push!(dims, S[i])
        else

        end
        stride *= i == 1 ? R : S[i]
    end
end

function ∂getindex()


end



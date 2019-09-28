
struct StaticUnitRange{L,S} <: AbstractFixedSizeVector{L,Int,L} end
Base.getindex(::StaticUnitRange{L,S}, i::Integer) where {L,S} = Int(i+S)
Base.size(::StaticUnitRange{L}) where {L} = (L,)
Base.length(::StaticUnitRange{L}) where {L} = L

Base.IndexStyle(::Type{<:StaticUnitRange}) = Base.IndexLinear()
@generated StaticUnitRange(::Val{Start}, ::Val{Stop}) where {Start,Stop} = StaticUnitRange{Stop-Start+1,Start-1}()
macro StaticRange(rq)
    @assert rq.head == :call
    @assert rq.args[1] == :(:)
    :(StaticUnitRange(Val{$(rq.args[2])}(), Val{$(rq.args[3])}()))
end


@generated function Base.getindex(A::AbstractMutableFixedSizeArray{SV,T,N,XV}, I...) where {SV,T,N,XV}
    S = Int[SV.parameters...]
    X = Int[XV.parameters...]
    offset = 0
    stride = 1
    dims = Int[]
    for (i, TT) in enumerate(I)
        if TT <: Integer

        elseif TT <: StaticUnitRange
            L, S = first(TT.parameters)::Int, last(TT.parameters)::Int
            push!(dims, L)
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





staticrangelength(::Type{Static{R}}) where R = 1 + last(R) - first(R)

struct ViewAdjoint{SP,SV,XP,XV}
    offset::Int
end

function generalized_getindex_quote(SV, XV, T, @nospecialize(inds), partial::Bool = false)
    N = length(SV)
    s2 = Int[]
    x2 = Int[]
    offset::Int = 0
    offset_expr = Expr[]
    for n ∈ 1:N
        xvn = (XV[n])::Int
        if inds[n] == Colon
            push!(s2, (SV[n])::Int)
            push!(x2, xvn)
        elseif inds[n] <: Integer
            push!(offset_expr, :($(sizeof(T)) * $xvn * (inds[n] - 1)))
        else
            @assert inds[n] <: Static
            push!(s2, staticrangelength(inds[n]))
            push!(x2, xvn)
            offset += sizeof(T) * (first(static_type(inds[n])) - 1) * xvn
        end
    end
    S2 = Tuple{s2...}
    X2 = Tuple{x2...}
    if partial
        if length(offset_expr) == 0 && offset == 0
            exo = 0
            ex = :(pointer(A))
        elseif offset == 0
            exo = length(offset_expr) > 1 ? Expr(:call, :+, offset_expr...) : offset_expr
            ex = :(pointer(A) + _offset)
        else
            exo = Expr(:call, :+, offset, offset_expr...)
            ex = :(pointer(A) + _offset)
        end
        partial_expr = :(ViewAdjoint{$(Tuple{SV...}),$S2,$(Tuple{XV...}),$X2}(_offset))
        length(s2) == 0 && return :( Expr(:meta,:inline); _offset = $ex; ( VectorizationBase.load( $ex ), $partial_expr ) )
        return quote
            $(Expr(:meta,:inline))
            _offset = $exo
            PtrArray{$S2,$T,$(length(s2)),$X2,$L,true}($ex), $partial_expr
        end
    else
        if length(offset_expr) == 0 && offset == 0
            ex = :(pointer(A))
        elseif offset == 0
            ex = Expr(:call, :+, :(pointer(A)), offset_expr...)
        else
            ex = Expr(:call, :+, :(pointer(A)), offset, offset_expr...)
        end
        length(s2) == 0 && return :( Expr(:meta,:inline); VectorizationBase.load( $ex ) )
        return quote
            $(Expr(:meta,:inline))
            PtrArray{$S2,$T,$(length(s2)),$X2,$L,true}($ex)
        end
    end
end
"""
Note that the stride will currently only be correct when N <= 2.
Perhaps R should be made into a stride-tuple?
"""
@generated function Base.getindex(A::AbstractMutableFixedSizeArray{S,T,N,X,L}, inds...) where {S,T,N,X,L}
    @assert length(inds) == N
    generalized_getindex_quote(S.parameters, X.parameters, T, inds, false)
end

function ∂getindex(A::AbstractMutableFixedSizeArray{S,T,N,X,L}, inds...) where {S,T,N,X,L}
    @assert length(inds) == N
    generalized_getindex_quote(S.parameters, X.parameters, T, inds, true)
end
@generated function RESERVED_INCREMENT_SEED_RESERVED(
    sp::StackPointer,
    a::AbstractMutableFixedSizeArray{SV,T,NV,XV},
    b::ViewAdjoint{SP,SV,XP,XV},
    c::AbstractMutableFixedSizeArray{SP,T,NP,XP}
) where {SP,SV,XP,XV,T,NP,NV}
    LV = last(SV.parameters)::Int * last(XV.parameters)::Int
    # q = quote
        # d = PtrArray{$SV,$T,$NV,$XV,$LV,true}(pointer(c) + b.offset)
    # end
    if (XV.parameters[1])::Int > 1
        SVT = tuple(SV.parameters...)
        return quote
            d = PtrArray{$SV,$T,$NV,$XV,$LV,true}(pointer(c) + b.offset)
            @inbounds @nloops $NV i j -> $SVT[j] begin
                @nref $NV d i = @nref $NV a i
            end
            sp, c
        end
    end
    if NV == 1
        return quote
            ptra = pointer(a)
            ptrc = pointer(c) + b.offset
            @vvectorize $T 4 for r in 1:$(SV.parameters[1])
                ptrc[r] = ptra[r] + ptrc[r]
            end
            sp, c
        end
    elseif NV == 2
        if NP == 2 && (first(SV.parameters)::Int == first(SP.parameters)::Int)
            L = ((SV.parameters[2])::Int) * (XP.parameters[2])::Int
            return quote
                ptra = pointer(a)
                ptrc = pointer(c) + b.offset
                @vvectorize $T 4 for r in 1:$L
                    ptrc[r] = ptra[r] + ptrc[r]
                end
            end
            sp, c
        else
            P = (XV.parameters[2])::Int
            return quote
                ptra = pointer(a)
                ptrc = pointer(c) + b.offset
                for c in 1:$(SV.parameters[2])
                    @vvectorize $T 4 for r in 1:$(SV.parameters[1])
                        ptrc[r] = ptra[r] + ptrc[r]
                    end
                    ptra += $P; ptrc += $P
                end
                sp, c
            end
        end
    elseif NV ≥ 3
        SVT = tuple(SV.parameters...)
        return quote
            @nloops $(NV-1) i j -> 0:$SVT[j+1]-1 begin
                offa = $(Expr(:call, :(+), [:( $(Symbol(:i_,n-1)) * $((XV.parameters[n])::Int) ) for n in 2:NV]...))
                offc = b.offset + offa
                @vectorize $T for i_0 ∈ 1:$(ST[1])
                    c[n + offc] = a[n + offa] + c[n + offc]
                end
            end
            sp, c
        end
    end
end
@generated function RESERVED_INCREMENT_SEED_RESERVED(
    sp::StackPointer,
    a::AbstractMutableFixedSizeArray{SV,T,NV,XV},
    b::ViewAdjoint{SP,SV,XP,XV}
#    c::AbstractMutableFixedSizeArray{SP,T,NP,XP}
) where {SP,SV,XP,XV,T,NV}
    NP = length(SP.parameters)
    LP = last(SP.parameters)::Int * last(XP.parameters)::Int
    quote
#        $(Expr(:meta,:inline))
        sp, c = PtrArray{$SP,$T,$NP,$XP}(sp)
        @inbounds for l in 1:$LP
            c[l] = zero($T)
        end
        RESERVED_INCREMENT_SEED_RESERVED(sp, a, b, c)
    end
end


# macro copy(expr)
    # @assert expr.head == :ref
    # q = Expr(:call, :(PaddedMatrices.sview), expr.args[1])
    # for n ∈ 2:length(expr.args)
        # original_ind = expr.args[n]
        # if original_ind isa Expr && original_ind.args[1] == :(:)
            # new_ind = :(PaddedMatrices.Static{$original_ind}())
        # else
            # new_ind = original_ind
        # end
        # push!(q.args, new_ind)
    # end
    # esc(q)
# end


# struct StaticUnitRange{L,S} <: AbstractFixedSizeVector{L,Int,L} end
# Base.getindex(::StaticUnitRange{L,S}, i::Integer) where {L,S} = Int(i+S)
# Base.size(::StaticUnitRange{L}) where {L} = (L,)
# Base.length(::StaticUnitRange{L}) where {L} = L

# Base.IndexStyle(::Type{<:StaticUnitRange}) = Base.IndexLinear()
# @generated StaticUnitRange(::Val{Start}, ::Val{Stop}) where {Start,Stop} = StaticUnitRange{Stop-Start+1,Start-1}()
# macro StaticRange(rq)
    # @assert rq.head == :call
    # @assert rq.args[1] == :(:)
    # :(StaticUnitRange(Val{$(rq.args[2])}(), Val{$(rq.args[3])}()))
# end


# @generated function Base.getindex(A::AbstractMutableFixedSizeArray{SV,T,N,XV}, I...) where {SV,T,N,XV}
    # S = Int[SV.parameters...]
    # X = Int[XV.parameters...]
    # offset = 0
    # stride = 1
    # dims = Int[]
    # for (i, TT) in enumerate(I)
        # if TT <: Integer

        # elseif TT <: StaticUnitRange
            # L, S = first(TT.parameters)::Int, last(TT.parameters)::Int
            # push!(dims, L)
            # offset += stride * L
        # elseif TT === Colon
            # push!(dims, S[i])
        # else

        # end
        # stride *= i == 1 ? R : S[i]
    # end
# end




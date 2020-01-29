

staticrangelength(::Type{Static{R}}) where {R} = 1 + last(R) - first(R)
staticrangelength(::Type{VectorizationBase.StaticUnitRange{L,U}}) where {L,U} = 1 + U - L

struct ViewAdjoint{SP,SV,XP,XV}
    offset::Int
end

function generalized_getindex_quote(SV, XV, T, @nospecialize(inds), partial::Bool = false, scalarview::Bool = false)
    N = length(SV)
    s2 = Int[]
    x2 = Int[]
    offset::Int = 0
    offset_expr = Expr[]
    size_expr = Expr(:tuple)
    size_T = sizeof(T)
    for n ∈ 1:N
        xvn = (XV[n])::Int
        if inds[n] <: Integer
            push!(offset_expr, :($size_T * $xvn * (inds[$n] - 1)))
        else
            push!(x2, xvn)
            if inds[n] == Colon
                push!(s2, (SV[n])::Int)
            elseif inds[n] <: Static
                push!(s2, staticrangelength(inds[n]))
                offset += sizeof(T) * (first(static_type(inds[n])) - 1) * xvn
            elseif inds[n] <: VectorizationBase.StaticUnitRange
                push!(s2, staticrangelength(inds[n]))
                offset += sizeof(T) * (first(static_type(inds[n])) - 1) * xvn
            elseif inds[n] <: AbstractRange{<:Integer}
                push!(s2, -1)
                push!(offset_expr, :($size_T * $xvn * @inbounds( first(inds[$n]) - 1 )))
                push!(size_expr.args, Expr(:call, :length, :(@inbounds(inds[$n]))))
            elseif inds[n] <: Base.OneTo
                push!(s2, -1)
                push!(size_expr.args, Expr(:call, :length, :(@inbounds(inds[$n]))))
            else
                throw("Indices of type $(inds[n]) not currently supported.")
            end
        end
    end
    S2 = Tuple{s2...}
    X2 = Tuple{x2...}
    L = prod(x2) * last(s2)
    if partial
        @assert length(size_expr.args) == 0
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
        if length(s2) == 0
            if scalarview
                return :( Expr(:meta,:inline); VectorizationBase.Pointer( $ex ) )
            else
                return :( Expr(:meta,:inline); VectorizationBase.load( $ex ) )
            end
        end
        if length(size_expr.args) == 0
            return quote
                $(Expr(:meta,:inline))
                PtrArray{$S2,$T,$(length(s2)),$X2,$L,true}($ex)
            end
        else
            return quote
                $(Expr(:meta,:inline))
                DSFSPtrArray{$S2,$T,$(length(s2)),$X2,$(length(size_expr))}($ex, $size_expr)
            end
        end
    end
end
"""
Note that the stride will currently only be correct when N <= 2.
Perhaps R should be made into a stride-tuple?
"""
@generated function Base.getindex(A::AbstractMutableFixedSizeArray{S,T,N,X,L}, inds...) where {S,T,N,X,L}
    @assert length(inds) == N
    generalized_getindex_quote(S.parameters, X.parameters, T, inds, false, false)
end
@generated function Base.view(A::AbstractMutableFixedSizeArray{S,T,N,X,L}, inds...) where {S,T,N,X,L}
    @assert length(inds) == N
    generalized_getindex_quote(S.parameters, X.parameters, T, inds, false, true)
end

@generated function ∂getindex(A::AbstractMutableFixedSizeArray{S,T,N,X,L}, inds...) where {S,T,N,X,L}
    @assert length(inds) == N
    generalized_getindex_quote(S.parameters, X.parameters, T, inds, true)
end
@generated function RESERVED_INCREMENT_SEED_RESERVED!(
    c::AbstractMutableFixedSizeArray{SP,T,NP,XP,LP},
    b::ViewAdjoint{SP,SV,XP,XV},
    a::AbstractMutableFixedSizeArray{SV,T,NV,XA,LV}
) where {SP,SV,XP,XA,XV,T,NP,NV,LP,LV}
    if (XA.parameters[1])::Int > 1
        # LV = last(SV.parameters)::Int * last(XV.parameters)::Int
        SVT = tuple(SV.parameters...)
        q = quote
            d = PtrArray{$SV,$T,$NV,$XV,$LV,true}(pointer(c) + b.offset)
            @inbounds @nloops $NV i j -> $SVT[j] begin
                @nref $NV d i += @nref $NV a i
            end
            nothing
        end
    end
    if NV == 1
        q = quote
            ptra = pointer(a)
            ptrc = pointer(c) + b.offset
            @vvectorize $T 4 for r in 1:$(SV.parameters[1])
                ptrc[r] = ptra[r] + ptrc[r]
            end
            nothing
        end
    elseif NV == 2
        if NP == 2 && (first(SV.parameters)::Int == first(SP.parameters)::Int) && (XV.parameters[2])::Int == (XA.parameters[2])::Int
            L = ((SV.parameters[2])::Int) * (XP.parameters[2])::Int
            q = quote
                ptra = pointer(a)
                ptrc = pointer(c) + b.offset
                @vvectorize $T 4 for r in 1:$L
                    ptrc[r] = ptra[r] + ptrc[r]
                end
                nothing
            end
        else
            PV = (XV.parameters[2])::Int * sizeof(T) # Take strides along the view's second axis
            PA = (XA.parameters[2])::Int * sizeof(T)
            q = quote
                ptra = pointer(a)
                ptrc = pointer(c) + b.offset
                for c in 1:$(SV.parameters[2])
                    @vvectorize $T 4 for r in 1:$(SV.parameters[1])
                        ptrc[r] = ptra[r] + ptrc[r]
                    end
                    ptra += $PA; ptrc += $PV
                end
                nothing
            end
        end
    elseif NV ≥ 3
        SVT = tuple(SV.parameters...)
        q = quote
            @nloops $(NV-1) i j -> 0:$SVT[j+1]-1 begin
                offc = $(Expr(:call, :(+), :(b.offset), [:( $(Symbol(:i_,n-1)) * $((XV.parameters[n])::Int) ) for n in 2:NV]...))
                offa = $(Expr(:call, :(+), [:( $(Symbol(:i_,n-1)) * $((XA.parameters[n])::Int) ) for n in 2:NV]...))
                @vectorize $T for i_0 ∈ 1:$(ST[1])
                    c[n + offc] = a[n + offa] + c[n + offc]
                end
            end
            nothing
        end
    end
    if c <: UninitializedArray
        # then we zero-initialize first
        qi = quote
            @inbounds for l in 1:$LP
                c[l] = zero($T)
            end
        end
        pushfirst!(q.args, qi)
    end
    q
end




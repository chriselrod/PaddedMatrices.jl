
@inline flatvector(A::AbstractFixedSizeArray{S,T,1,Tuple{1},false,L}) where {S,T,L} = A
@inline function flatvector(A::AbstractFixedSizeArray{S,T,N,<:Tuple{1,Vararg},false,L}) where {S,T,N,L}
    PtrArray{Tuple{L},T,1,Tuple{1},0,0,false,L}(pointer(A),tuple(),tuple())
end

@inline flatvector(A::StrideArray{S,T,1,Tuple{1},0,0,L}) where {S,T,L} = A
@inline function flatvector(A::StrideArray{S,T,N,<:Tuple{1,Vararg},0,0,L}) where {S,T,N,L}
    StrideArray{Tuple{L},T,1,Tuple{1},0,0,L}(A.data, tuple(), tuple())
end

@inline flatvector(A::StrideArray{S,T,1,Tuple{1},SN,XN,-1}) where {S,T,SN,XN} = A
@inline function flatvector(A::StrideArray{S,T,N,<:Tuple{1,Vararg},SN,XN,-1}) where {S,T,N,SN,XN}
    StrideArray{Tuple{-1},T,1,Tuple{1},1,0,-1}(A.data, prod(size(A)), tuple())
end

@inline flatvector(A::AbstractStrideArray{Tuple{-1},T,1,Tuple{1},1,0,false,-1}) where {T} = A
@inline function flatvector(A::AbstractStrideArray{S,T,N,<:Tuple{1,Vararg},SN,XN,false,-1}) where {S,T,N,SN,XN}
    PtrArray{Tuple{-1},T,1,Tuple{1},1,0,-1}(pointer(A), prod(size(A)), tuple())
end

@inline flatvector(A::ConstantArray{S,T,1,Tuple{1},L}) where {S,T,L} = A
@inline function flatvector(A::ConstantArray{S,T,N,X,L}) where {S,T,N,X,L}
    ConstantArray{Tuple{L},T,1,Tuple{1},L}(A.data)
end

@inline flatvector(a::Number) = a


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
    for n ∈ 1:N
        xvn = (XV[n])::Int
        if inds[n] <: Integer
            push!(offset_expr, :($xvn * (inds[$n] - 1)))
        else
            push!(x2, xvn)
            if inds[n] == Colon
                push!(s2, (SV[n])::Int)
            elseif inds[n] <: Static
                push!(s2, staticrangelength(inds[n]))
                offset += (first(static_type(inds[n])) - 1) * xvn
            elseif inds[n] <: VectorizationBase.StaticUnitRange
                push!(s2, staticrangelength(inds[n]))
                offset += (first(static_type(inds[n])) - 1) * xvn
            elseif inds[n] <: AbstractRange{<:Integer}
                push!(s2, -1)
                push!(offset_expr, :($xvn * @inbounds( first(inds[$n]) - 1 )))
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
            ex = :(gep(pointer(A), _offset))
        else
            exo = Expr(:call, :+, offset, offset_expr...)
            ex = :(gep(pointer(A), _offset))
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
            if length(offset_expr) == 1
                ex = Expr(:call, :gep, :(pointer(A)), first(offset_expr))
            else
                ex = Expr(:call, :gep, :(pointer(A)), Expr(:call, :+, offset_expr...))
            end
        else
            ex = Expr(:call, :gep, :(pointer(A)), Expr(:call, :+, offset, offset_expr...))
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



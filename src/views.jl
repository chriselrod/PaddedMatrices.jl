
@inline flatvector(A::AbstractFixedSizeArray{S,T,1,Tuple{1},false}) where {S,T} = A
@generated function flatvector(A::AbstractFixedSizeArray{S,T,N,X,false}) where {S,T,N,X<:Tuple{1,Vararg}}
    L = last(S.parameters)::Int * last(X.parameters)::Int
    Expr(
        :block,
        Expr(:meta,:inline),
        :(PtrArray{Tuple{$L},$T,1,Tuple{1},0,0,false}(pointer(A),tuple(),tuple()))
    )
end

@inline flatvector(A::StrideArray{S,T,1,Tuple{1},0,0}) where {S,T} = A
@generated function flatvector(A::StrideArray{S,T,N,X,0,0}) where {S,T,N,X<:Tuple{1,Vararg}}
    L = last(S.parameters)::Int * last(X.parameters)::Int
    Expr(
        :block,
        Expr(:meta,:inline),
        :(StrideArray{Tuple{$L},$T,1,Tuple{1},0,0}(A.ptr, tuple(), tuple(), A.parent))
    )
end

@inline flatvector(A::StrideArray{S,T,1,Tuple{1},SN,XN}) where {S,T,SN,XN} = A
@inline function flatvector(A::StrideArray{S,T,N,<:Tuple{1,Vararg},SN,XN}) where {S,T,N,SN,XN}
    StrideArray{Tuple{-1},T,1,Tuple{1},1,0}(A.ptr, (prod(size(A)),), tuple(), A.parent)
end

@inline flatvector(A::AbstractStrideArray{Tuple{-1},T,1,Tuple{1},1,0,false}) where {T} = A
@inline function flatvector(A::AbstractStrideArray{S,T,N,<:Tuple{1,Vararg},SN,XN,false}) where {S,T,N,SN,XN}
    PtrArray{Tuple{-1},T,1,Tuple{1},1,0}(pointer(A), (prod(size(A)),), tuple())
end

@inline flatvector(A::ConstantArray{S,T,1,Tuple{1}}) where {S,T} = A
@inline function flatvector(A::ConstantArray{S,T,N,X,L}) where {S,T,N,X,L}
    ConstantArray{Tuple{L},T,1,Tuple{1},L}(A.data)
end

@inline flatvector(a::Number) = a


staticrangelength(::Type{Static{R}}) where {R} = 1 + last(R) - first(R)
staticrangelength(::Type{VectorizationBase.StaticUnitRange{L,U}}) where {L,U} = 1 + U - L

# struct ViewAdjoint{SP,SV,XP,XV,SN,XN}
#     offset::Int
#     size::NTuple{SN,Int}
#     stride::NTuple{XN,Int}
# end

static_type(::Type{StaticUnitRange{L,U}}) where {L,U} = (L,U)
function generalized_getindex_quote(SV, XV, @nospecialize(inds))
    N = length(SV)
    s2 = Int[]
    x2 = Int[]
    sti = 0; xti = 0
    st2 = Expr(:tuple)
    xt2 = Expr(:tuple)
    offset::Int = 0
    offset_expr = Expr[]
    size_expr = Expr(:tuple)
    for n ∈ 1:N
        svn = (SV[n])::Int
        svn == -1 && (sti += 1)
        xvn = (XV[n])::Int
        xvn == -1 && (xti += 1)
        if inds[n] <: Integer
            push!(offset_expr, :($xvn * (inds[$n] - 1)))
        else
            push!(x2, xvn)
            if xvn == -1
                push!(xt2.args, Expr(:ref, :Astride, xti))
            end
            if inds[n] == Colon
                push!(s2, svn)
                if svn == -1
                    push!(st2.args, Expr(:ref, :Asize, sti))
                end
            # elseif inds[n] <: Static
                # push!(s2, staticrangelength(inds[n]))
                # offset += (first(static_type(inds[n])) - 1) * xvn
            elseif inds[n] <: VectorizationBase.StaticUnitRange
                push!(s2, staticrangelength(inds[n]))
                offset += (first(static_type(inds[n])) - 1) * xvn
            elseif inds[n] <: AbstractRange{<:Integer}
                push!(s2, -1)
                push!(offset_expr, :($xvn * @inbounds( first(inds[$n]) - 1 )))
                push!(st2.args, Expr(:call, :length, :(@inbounds(inds[$n]))))
            elseif inds[n] <: Base.OneTo
                push!(s2, -1)
                push!(st2.args, Expr(:call, :length, :(@inbounds(inds[$n]))))
            else
                throw("Indices of type $(inds[n]) not currently supported.")
            end
        end
    end
    S2 = Tuple{s2...}
    X2 = Tuple{x2...}
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
    SN = length(st2.args)
    XN = length(xt2.args)
    q = quote
        $(Expr(:meta,:inline))
        _offset = $ex
    end
    SN > 0 && push!(q.args, Expr(:(=), :Asize, Expr(:(.), :A, QuoteNode(:size))))
    XN > 0 && push!(q.args, Expr(:(=), :Astride, Expr(:(.), :A, QuoteNode(:stride))))
    # arraydef = if length(s2) == 0
    #     scalarview ? :(VectorizationBase.Pointer) : :(VectorizationBase.load)
    # else
    #     :(PtrArray{$S2,$T,$(length(s2)),$X2,$SN,$XN,true})
    # end
    q, S2, length(s2), X2, SN, XN, st2, xt2
    # partial_expr = :(ViewAdjoint{$(Tuple{SV...}),$S2,$(Tuple{XV...}),$X2,$SN,$XN}(_offset, $st2, $xt2))
    # push!(q.args, :(@inbounds $arraydef( _offset, $st2, $xt2), $partial))
    # push!(q.args, :(@inbounds $arraydef( _offset, $st2, $xt2)))
    # q
end
"""
Note that the stride will currently only be correct when N <= 2.
Perhaps R should be made into a stride-tuple?
"""
@generated function Base.getindex(A::AbstractPtrStrideArray{S,T,N,X}, inds::Vararg{<:Any,N}) where {S,T,N,X,V}
    @assert length(inds) == N
    q, S2, Nsub, X2, SN, XN, st2, xt2 = generalized_getindex_quote(S.parameters, X.parameters, T, inds, false)
    arraydef = if iszero(Nsub)
        :(VectorizationBase.vload)
    else
        :(PtrArray{$S2,$T,$Nsub,$X2,$SN,$XN,true})
    end
    push!(q.args, :(@inbounds $arraydef( _offset, $st2, $xt2 )))
    q
end
Base.@propagate_inbounds function Base.getindex(A::StrideArray{S,T,N}, inds::Vararg{<:Any,N}) where {S,T,N}
    q, S2, Nsub, X2, SN, XN, st2, xt2 = generalized_getindex_quote(S.parameters, X.parameters, T, inds, false)
    if iszero(Nsub)
        push!(q.args, :(@inbounds GC.@preserve A vload( _offset, $st2, $xt2, parent(A) )))
    else
        push!(q.args, :(@inbounds StrideArray{$S2,$T,$Nsub,$X2,$SN,$XN,true}( _offset, $st2, $xt2, parent(A) )))
    end
    q
end
Base.@propagate_inbounds function Base.getindex(A::AbstractStrideArray{S,T,N}, inds::Vararg{<:Any,N}) where {S,T,N}
    view(A, inds...)
end
@generated function Base.view(
    A::AbstractPtrStrideArray{S,T,N,X}, inds::Vararg{<:Any,N}
# ) where {S,T,N,X,SN,XN,V}
) where {V,S,T,N,X}
    @assert length(inds) == N
    generalized_getindex_quote(S.parameters, X.parameters, T, inds, true)
    # @show generalized_getindex_quote(S.parameters, X.parameters, T, inds, false, true)
end



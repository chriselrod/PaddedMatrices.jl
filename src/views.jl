
# @inline function gep_no_offset(ptr::VectorizationBase.AbstractStridedPointer, i::Tuple)
    # VectorizationBase.gep(pointer(ptr), VectorizationBase.tdot(ptr, i, VectorizationBase.strides(ptr), VectorizationBase.nopromote_axis_indicator(ptr)))
# end
# @inline function similar_with_offset(sptr::StridedPointer{T,N,C,B,R,X,O}, ptr::Ptr{T}) where {T,N,C,B,R,X,O}
#     StridedPointer{T,N,C,B,R,X}(ptr, sptr.strd, zerotuple(Val{N}()))
# end
_extract(::Type{StaticInt{N}}) where {N} = N::Int
_extract(_) = missing
@generated function Base.view(A::PtrArray{S,D,T,N,C,B,R,X,O}, i::Vararg{Union{Integer,AbstractRange,Colon},K}) where {K,S,D,T,N,C,B,R,X,O}
    @assert ((K == N) || isone(K))

    inds = Expr(:tuple)
    Nnew = 0
    s = Expr(:tuple)
    x = Expr(:tuple)
    o = Expr(:tuple)
    Rnew = Expr(:tuple)
    Dnew = Expr(:tuple)
    Cnew = -1
    Bnew = -1
    sortp = ArrayInterface.rank_to_sortperm(R)
    still_dense = true
    densev = Vector{Bool}(undef, K)
    for k ∈ 1:K
        iₖ = Expr(:ref, :i, k)
        if i[k] === Colon
            Nnew += 1
            push!(inds.args, Expr(:ref, :o, k))
            push!(s.args, Expr(:ref, :s, k))
            push!(x.args, Expr(:ref, :x, k))
            push!(o.args, :(One()))
            if k == C
                Cnew = Nnew
            end
            if k == B
                Bnew = Nnew
            end
            push!(Rnew.args, R[k])
        else
            push!(inds.args, Expr(:call, :first, iₖ))
            if i[k] <: AbstractRange
                Nnew += 1
                push!(s.args, Expr(:call, :static_length, iₖ))
                push!(x.args, Expr(:ref, :x, k))
                push!(o.args, :(One()))
                if k == C
                    Cnew = Nnew
                end
                if k == B
                    Bnew = Nnew
                end
                push!(Rnew.args, R[k])
            end
        end
        spₙ = sortp[k]
        if still_dense & D[spₙ]
            ispₙ = i[spₙ]
            still_dense = (ispₙ <: AbstractUnitRange) || (ispₙ === Colon)
            densev[spₙ] = still_dense
            if still_dense
                still_dense = (((ispₙ === Colon)::Bool || (ispₙ <: Base.Slice)::Bool) ||
                               ((ispₙ <:  ArrayInterface.OptionallyStaticUnitRange{<:StaticInt,<:StaticInt})::Bool &&
                                (ArrayInterface.known_length(ispₙ) === _extract(S.parameters[spₙ]))::Bool))
            end
        else
            still_dense = false
        end
    end
    for k ∈ 1:K
        iₖt = i[k]
        if (iₖt === Colon) || (iₖt <: AbstractVector)
            push!(Dnew.args, densev[k])
        end
    end    
    quote
        $(Expr(:meta,:inline))
        sp = A.ptr
        s = A.size
        x = sp.strd
        o = sp.offsets
        new_sp = StridedPointer{$T,$Nnew,$Cnew,$Bnew,$Rnew}(gep(sp, $inds), $x, $o)
        PtrArray(new_sp, $s, DenseDims{$Dnew}())
    end
end

@inline function Base.view(A::StrideArray, i::Vararg{Union{Integer,AbstractRange,Colon},K}) where {K}
    StrideArray(view(A.ptr, i...), A.data)
end

@inline function Base.vec(A::PtrArray{S,D,T,N,C,0}) where {S,D,T,N,C}
    @assert all(D) "All dimensions must be dense for a vec view. Try `vec(copy(A))` instead."
    sp = StridedPointer(pointer(A), (VectorizationBase.static_sizeof(T),), (One(),))
    PtrArray(sp, (static_length(A),), DenseDims((true,)))
end





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
            @avx @nloops $NV i j -> 1:$SVT[j] begin
                @nref $NV d i += @nref $NV a i
            end
            nothing
        end
    end
    if NV == 1
        q = quote
            ptra = pointer(a)
            ptrc = pointer(c) + b.offset
            @avx for r in 1:$(SV.parameters[1])
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
                @avx for r in 1:$L
                    ptrc[r] = ptra[r] + ptrc[r]
                end
                nothing
            end
        else
            PV = (XV.parameters[2])::Int # Take strides along the view's second axis
            PA = (XA.parameters[2])::Int
            q = quote
                ptra = pointer(a)
                ptrc = pointer(c) + b.offset
                for c in 1:$(SV.parameters[2])
                    @avx for r in 1:$(SV.parameters[1])
                        ptrc[r] = ptra[r] + ptrc[r]
                    end
                    ptra = gep(ptra, $PA); ptrc = gep(ptrc, $PV)
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
                ctemp = gep(pointer(c), offc)
                atemp = gep(pointer(a), offa)
                @avx for i_0 ∈ 1:$(ST[1])
                    ctemp[i_0] = atemp[i_0] + ctemp[i_0]
                end
            end
            nothing
        end
    end
    if c <: UninitializedArray
        # then we zero-initialize first
        qi = quote
            @avx for l in 1:$LP
                c[l] = zero($T)
            end
        end
        pushfirst!(q.args, qi)
    end
    q
end




@generated function PtrArray{S}(sp::StackPointer) where {S}
    N,X,L = calc_NPL(S.parameters,Float64,true,false)
    quote
        $(Expr(:meta,:inline))
        sp + $(align(8L)), PtrArray{$S,Float64,$N,$X,$L,false}(pointer(sp, Float64))
    end
end
@generated function PtrArray{S,T}(sp::StackPointer) where {S,T}
    N,X,L = calc_NPL(S.parameters,T,true,false)
    quote
        $(Expr(:meta,:inline))
        sp + $(align(sizeof(T)*L)), PtrArray{$S,$T,$N,$X,$L,false}(pointer(sp, $T))
    end
end
@generated function PtrArray{S,T,N}(sp::StackPointer) where {S,T,N}
    N2, X, L = calc_NPL(S.parameters, T)
    @assert N == N2 "length(S) == $(length(S.parameters)) != N == $N"
    quote
        $(Expr(:meta,:inline))
        sp + $(align(sizeof(T)*L)), PtrArray{$S,$T,$N,$X,$L,false}(pointer(sp, $T))
    end
end
@generated function PtrArray{S,T,N,X}(sp::StackPointer) where {S,T,N,X}
    L = simple_vec_prod(X.parameters) * last(S.parameters)::Int
    quote
        $(Expr(:meta,:inline))
        sp + $(align(sizeof(T)*L)), PtrArray{$S,$T,$N,$X,$L,false}(pointer(sp, $T))
#        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
    end
end
# @generated function PtrArray{S,T,N,X,L}(sp::StackPointer) where {S,T,N,X,L}
#     quote
#         $(Expr(:meta,:inline))
#         sp + $(align(sizeof(T)*L)), PtrArray{$S,$T,$N,$X,$L,false}(pointer(sp, $T))
# #        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
#     end
# end
# @generated function PtrArray{S,T,N,X,L,V}(sp::StackPointer) where {S,T,N,X,L,V}
#     quote
#         $(Expr(:meta,:inline))
#         sp + $(align(sizeof(T)*L)), PtrArray{$S,$T,$N,$X,$L,$V}(pointer(sp, $T))
# #        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
#     end
# end
@inline function PtrArray(sp::StackPointer, sz::Tuple)
    A = PtrArray(pointer(sp, Float64), sz)
    sp + align(memory_length(A)), A
end
@inline function PtrArray{T}(sp::StackPointer, sz::Tuple) where {T}
    A = PtrArray(pointer(sp, T), sz)
    sp + align(memory_length(A)), A
end
@inline (sp::StackPointer)(::typeof(allocarray), ::Type{T}, s) where {T} = PtrArray{T}(sp, s)


# @generated function PtrVector{P,T}(a::StackPointer) where {P,T}
#     L = calc_padding(P, T)
#     quote
#         $(Expr(:meta,:inline))
#         a + $(align(L*sizeof(T))), PtrArray{Tuple{$P},$T,1,Tuple{1},$L,false}(pointer(a,$T))
#     end
# end
# @generated function PtrMatrix{M,N,T}(a::StackPointer) where {M,N,T}
#     L = calc_padding(M, T)
#     quote
#         $(Expr(:meta,:inline))
#         a + $(align(L*N*sizeof(T))), PtrArray{Tuple{$M,$N},$T,2,Tuple{1,$L},$(L*N),false}(pointer(a,$T))
#     end
# end
@inline Base.similar(sp::StackPointer, ::AbstractFixedSizeArray{S,T,N,X}) where {S,T,N,X} = PtrArray{S,T,N,X,false}(sp)
@inline function (sp::StackPointer)(::typeof(similar), ::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,X,L}}) where {S,T,N,X,L}
    sp, A = PtrArray{S,T,N,X}(sp)
    sp, A'
end

@inline function (sp::StackPointer)(::typeof(copy), A::AbstractFixedSizeArray{S,T,N,X}) where {S,T,N,X}
    sp, B = PtrArray{S,T,N,X}(sp)
    sp, copyto!(B, A)
end


## Reverse diff expr support
SIMDPirates.vfmadd!(C::AbstractArray, A, B) = mul!(C, A, B, Val{1}(), Val{1}())
SIMDPirates.vfnmadd!(C::AbstractArray, A, B) = mul!(C, A, B, Val{-1}(), Val{1}())
SIMDPirates.vfmadd!(C::ZeroInitializedArray, A, B) = mul!(C.data, A, B, Val{1}(), Val{0}())
SIMDPirates.vfnmadd!(C::ZeroInitializedArray, A, B) = mul!(C.data, A, B, Val{-1}(), Val{0}())

@inline function (sp::StackPointer)(::typeof(*), A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}) where {Ta, Tb}
    T = promote_type(Ta, Tb)
    sp, C = PtrArray{T}(sp, maybestaticsize(A, Val{1}()), maybestaticsize(B, Val{2}()))
    sp, mul!(C, A, B)
end
@inline function (sp::StackPointer)(::typeof(vnmul), A::AbstractMatrix, B::AbstractMatrix)
    T = promote_type(Ta, Tb)
    sp, C = PtrArray{T}(sp, (maybestaticsize(A, Val{1}()), maybestaticsize(B, Val{2}())))
    sp, mul!(C, A, B, Val{-1}())
end


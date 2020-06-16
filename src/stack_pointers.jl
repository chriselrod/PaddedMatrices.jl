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
@generated function PtrArray{S,T,N,X,L}(sp::StackPointer) where {S,T,N,X,L}
    quote
        $(Expr(:meta,:inline))
        sp + $(align(sizeof(T)*L)), PtrArray{$S,$T,$N,$X,$L,false}(pointer(sp, $T))
#        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
    end
end
@generated function PtrArray{S,T,N,X,L,V}(sp::StackPointer) where {S,T,N,X,L,V}
    quote
        $(Expr(:meta,:inline))
        sp + $(align(sizeof(T)*L)), PtrArray{$S,$T,$N,$X,$L,$V}(pointer(sp, $T))
#        PtrArray{$S,$T,$N,$R,$L,$P}(ptr)
    end
end
@inline function PtrArray(sp::StackPointer, sz::Vararg{<:Any,N}) where {N}
    A = PtrArray(pointer(sp), az)
    sp + align(memory_length(A)), A
end

@generated function PtrVector{P,T}(a::StackPointer) where {P,T}
    L = calc_padding(P, T)
    quote
        $(Expr(:meta,:inline))
        a + $(align(L*sizeof(T))), PtrArray{Tuple{$P},$T,1,Tuple{1},$L,false}(pointer(a,$T))
    end
end
@generated function PtrMatrix{M,N,T}(a::StackPointer) where {M,N,T}
    L = calc_padding(M, T)
    quote
        $(Expr(:meta,:inline))
        a + $(align(L*N*sizeof(T))), PtrArray{Tuple{$M,$N},$T,2,Tuple{1,$L},$(L*N),false}(pointer(a,$T))
    end
end
@inline Base.similar(sp::StackPointer, ::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L} = PtrArray{S,T,N,X,L,false}(sp)
@inline function StackPointers.stack_pointer_call(::typeof(similar), sp::StackPointer, ::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,X,L}}) where {S,T,N,X,L}
    sp, A = PtrArray{S,T,N,X,L,false}(sp)
    sp, A'
end

@inline function StackPointers.stack_pointer_call(::typeof(copy), sp::StackPointer, A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    sp, B = PtrArray{S,T,N,X,L,false}(sp)
    sp, copyto!(B, A)
end


## Reverse diff expr support
SIMDPirates.vfmadd!(C::AbstractArray, A, B) = mul!(C, A, B, Val{1}(), Val{1}())
SIMDPirates.vfnmadd!(C::AbstractArray, A, B) = mul!(C, A, B, Val{-1}(), Val{1}())
SIMDPirates.vfmadd!(C::ZeroInitializedArray, A, B) = mul!(C.data, A, B, Val{1}(), Val{0}())
SIMDPirates.vfnmadd!(C::ZeroInitializedArray, A, B) = mul!(C.data, A, B, Val{-1}(), Val{0}())

@inline function StackPointers.stack_pointer_call(::typeof(*), sp::StackPointer, A::AbstractMatrix, B::AbstractMatrix)
    sp, C = PtrArray(sp, maybestaticsize(A, Val{1}()), maybestaticsize(B, Val{2}()))
    sp, mul!(C, A, B)
end
@inline function StackPointers.stack_pointer_call(::typeof(vnmul), sp::StackPointer, A::AbstractMatrix, B::AbstractMatrix)
    sp, C = PtrArray(sp, maybestaticsize(A, Val{1}()), maybestaticsize(B, Val{2}()))
    sp, mul!(C, A, B, Val{-1}())
end


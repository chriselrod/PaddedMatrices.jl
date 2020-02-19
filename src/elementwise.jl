

# To be replaced with a set of @eval loops....

# for sptr ∈ (false, true)


# end

# function elementwise_op_quote(S, T, N, X, L, f, op, eq, sp, alloc::Bool, aisscalar::Bool, bisscalar::Bool)
    # Aref = aisscalar ? QuoteNode(:A) : :(Expr(:ref, :A, :l))
    # Bref = bisscalar ? QuoteNode(:B) : :(Expr(:ref, :B, :l))
# end
function elementwise_op_func_quote(m, f, op, eq, sp, alloc::Bool, aisscalar::Bool, bisscalar::Bool)
    args = Expr[]
    if aisscalar
        push!(args, :(A::T))
        Aref = QuoteNode(:A)
    else
        push!(args, :(A::AbstractFixedSizeArray{S,T,N,X,L}))
        Aref = :(Expr(:ref, :A, :l))
    end
    if bisscalar
        push!(args, :(B::T))
        Bref = QuoteNode(:B)
    else
        push!(args, :(B::AbstractFixedSizeArray{S,T,N,X,L}))
        Bref = :(Expr(:ref, :B, :l))
    end
    if sp && alloc
        allocq = :(pushfirst!(q.args, Expr(:(=), Expr(:tuple,:sp,:C), Expr(:call, Expr(:curly, :PtrArray,:S,:T,:N,:X,:L),:sp))))
    elseif alloc
        allocq = :(pushfirst!(q.args, Expr(:(=), :C, Expr(:call, Expr(:curly, :FixedSizeArray, :S, :T,:N, :X, :L), :undef))))
    else
        pushfirst!(args, :(C::AbstractMutableFixedSizeArray{S,T,N,X,L}))
        allocq = nothing
    end
    if sp
        pushfirst!(args, :(sp::StackPointer))
        ret = :(push!(q.args, :((sp,C))))
    else
        ret = :(push!(q.args, :C))
    end
    quote
        @generated function $m.$f($(args...)) where {S,T,N,X,L}
            # elementwise_op_quote(S.parameters, T, N, X.parameters, L, $(QuoteNode(op)), $(QuoteNode(eq)), $sp, $alloc, $aisscalar, $bisscalar)
            q = Expr(:block,
             Expr(:macrocall, Symbol("@vvectorize"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), T,
                  Expr(:for,
                       Expr(:(=), :l, Expr(:call, :(:), 1, L)),
                       Expr(:block, Expr($(QuoteNode(eq)), Expr(:ref, :C, :l), Expr(:call, $(QuoteNode(op)), $Aref, $Bref)))
                       )
                  )
             )
            $allocq
            4VectorizationBase.pick_vector_width(T) < L || pushfirst!(q.args, Expr(:meta,:inline))
            $ret
            q
        end
    end
end

function add! end
function sub! end
function elementwise_mul! end
function vfmadd! end
function vfnmadd! end

for sptr ∈ (true,false)
    for (m,f,op,eq,alloc) ∈ (
        (:Base,:+,:+,:(=),true),
        (:PaddedMatrices,:add!,:+,:(=),false),
        (:Base,:-,:-,:(=),true),
        (:PaddedMatrices,:sub!,:-,:(=),false),
        (:Base,:*,:*,:(=),true),
        (:PaddedMatrices,:elementwise_mul!,:*,:(=),false),
        (:PaddedMatrices,:vfmadd!,:*,:(+=),false),
        (:PaddedMatrices,:vfnmadd!,:*,:(-=),false)
    )
        for aisscalar ∈ (true,false)
            for bisscalar ∈ (true,false)
                aisscalar && bisscalar && continue
                eval(elementwise_op_func_quote(m, f, op, eq, sptr, alloc, aisscalar, bisscalar))
            end
        end
    end
end

@inline Base.:*(A::Diagonal{T,<:AbstractFixedSizeVector{N,T,P}}, B::AbstractFixedSizeVector{N,T,P}) where {N,T,P} = A.diag * B
@inline Base.:*(sp::StackPointer, A::Diagonal{T,<:AbstractFixedSizeVector{N,T,P}}, B::AbstractFixedSizeVector{N,T,P}) where {N,T,P} = sp * A.diag * B

@inline Base.:*(A::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeVector{N,T,P}}, B::Diagonal{T,<:AbstractFixedSizeVector{N,T,P}}) where {N,T,P} = (A.parent * B.diag)'
@inline function Base.:*(sp::StackPointer, A::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeVector{N,T,P}}, B::Diagonal{T,<:AbstractFixedSizeVector{N,T,P}}) where {N,T,P}
    (sp, c) = sp * A.parent * B.diag
    sp, c'
end

# @generated function Base.:+(
#     sp::StackPointer,
#     A::AbstractFixedSizeArray{S,T,N,X,LA},
#     B::AbstractFixedSizeArray{S,T,N,X,LB}
# ) where {S,T<:Number,N,X,LA,LB}
#     L = min(LA,LB)
#     @assert first(X.parameters)::Int == 1
#     quote
#         mv = PtrArray{$S,$T,$N,$X,$L}(pointer(sp,$T))
#         @vvectorize $T for i ∈ 1:$L
#             mv[i] = A[i] + B[i]
#         end
#         sp + $(VectorizationBase.align(sizeof(T)*L)), mv
#     end
# end
# @generated function Base.:+(
#     sp::StackPointer,
#     A::AbstractFixedSizeVector{N,T,LA},
#     B::AbstractFixedSizeVector{N,T,LB}
# ) where {N,T<:Number,LA,LB}
#     L = min(LA,LB)
#     quote
#         mv = PtrVector{$N,$T,$L}(pointer(sp,$T))
#         @vvectorize $T for i ∈ 1:$L
#             mv[i] = A[i] + B[i]
#         end
#         sp + $(VectorizationBase.align(sizeof(T)*L)), mv
#     end
# end
@inline function Base.:+(
    A::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,P,L}},
    B::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,P,L}}
) where {S,T,N,P,L}
    (A' + B')'
end
@inline function Base.:+(
    sp::StackPointer,
    A::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,XA,LA}},
    B::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,XB,LB}}
) where {S,T,N,XA,XB,LA,LB}
    (sp2, C) = (+(sp, A', B'))
    sp2, C'
end
# @generated function Base.:+(a::T, B::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
#     quote
#         # $(Expr(:meta,:inline))
#         mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
#         @vvectorize $T for i ∈ 1:$L
#             mv[i] = a + B[i]
#         end
#         ConstantFixedSizeArray(mv)
#     end
# end
# @generated function SIMDPirates.vadd(
#     sp::StackPointer, a::T,
#     B::AbstractFixedSizeArray{S,T,N,X,L}
# ) where {S,T<:Number,N,X,L}
#     quote
#         # $(Expr(:meta,:inline))
#         (sp, mv) = PtrArray{$S,$T,$N,$X,$L}(sp)
#         @vvectorize $T for i ∈ 1:$L
#             mv[i] = a + B[i]
#         end
#         sp, mv
#     end
# end
# @generated function Base.:+(A::AbstractFixedSizeArray{S,T,N,X,L}, b::T) where {S,T<:Number,N,X,L}
#     quote
#         # $(Expr(:meta,:inline))
#         mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
#         @vvectorize $T for i ∈ 1:$L
#             mv[i] = A[i] + b
#         end
#         ConstantFixedSizeArray(mv)
#     end
# end
# @generated function Base.:+(a::T, Badj::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,X,L}}) where {S,T<:Number,N,X,L}
#     quote
#         $(Expr(:meta,:inline))
#         mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
#         @vvectorize $T for i ∈ 1:$L
#             mv[i] = a + B[i]
#         end
#         ConstantFixedSizeArray(mv)'
#     end
# end
# @generated function Base.:+(Aadj::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,X,L}}, b::T) where {S,T<:Number,N,X,L}
#     quote
#         $(Expr(:meta,:inline))
#         mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
#         @vvectorize $T for i ∈ 1:$L
#             mv[i] = A[i] + b
#         end
#         ConstantFixedSizeArray(mv)'
#     end
# end
# @generated function Base.:-(A::AbstractFixedSizeArray{S,T,N,X,L}, B::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
#     quote
#         # $(Expr(:meta,:inline))
#         mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
#         @vvectorize $T for i ∈ 1:$L
#             mv[i] = A[i] - B[i]
#         end
#         ConstantFixedSizeArray(mv)
#     end
# end
# @generated function diff!(C::AbstractMutableFixedSizeArray{S,T,N,X,L}, A::AbstractFixedSizeArray{S,T,N,X,L}, B::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
#     quote
#         @vvectorize $T for i ∈ 1:$L
#             C[i] = A[i] - B[i]
#         end
#         C
#     end
# end
# @generated function Base.:-(A::AbstractFixedSizeArray{S,T,N,X,L}, b::T) where {S,T<:Number,N,L,X}
#     quote
#         # $(Expr(:meta,:inline))
#         mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
#         @vvectorize $T for i ∈ 1:$L
#             mv[i] = A[i] - b
#         end
#         ConstantFixedSizeArray(mv)
#     end
# end
# @generated function Base.:-(a::T, B::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
#     quote
#         # $(Expr(:meta,:inline))
#         mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
#         @vvectorize $T for i ∈ 1:$L
#             mv[i] = a - B[i]
#         end
#         ConstantFixedSizeArray(mv)
#     end
# end








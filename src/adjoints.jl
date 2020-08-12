@inline rev(t::NTuple{0}) = t
@inline rev(t::NTuple{1}) = t
@inline rev(t::NTuple) = reverse(t)
# reverse_tuple_type(t) = reverse_tuple_type(t.parameters)
# function reverse_tuple_type(t::Core.SimpleVector)
#     tnew = Expr(:curly, :Tuple)
#     N = length(t)
#     for n ∈ 0:N-1
#         push!(tnew.args, t[N - n])
#     end
#     tnew
# end
# @generated function Base.adjoint(A::AbstractStrideArray{S,T,N,X,SN,XN,V}) where {S,T<:Real,N,X,SN,XN,V}
#     Snew = reverse_tuple_type(S)
#     Xnew = reverse_tuple_type(X)
#     Expr(:block, Expr(:meta,:inline), :(PtrArray{$Snew,$T,$N,$Xnew,$SN,$XN,$V}(pointer(A), rev(A.size), rev(A.stride))))
# end
# @generated function Base.transpose(A::AbstractStrideArray{S,T,N,X,SN,XN,V}) where {S,T,N,X,SN,XN,V}
#     Snew = reverse_tuple_type(S)
#     Xnew = reverse_tuple_type(X)
#     Expr(:block, Expr(:meta,:inline), :(PtrArray{$Snew,$T,$N,$Xnew,$SN,$XN,$V}(pointer(A), rev(A.size), rev(A.stride))))
# end
@generated function Base.adjoint(A::ConstantArray{S,T,N,X,L}) where {S,T<:Real,N,X,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(ConstantArray{$Snew,$T,$N,$Xnew,$L}(A.data)))
end
@generated function Base.transpose(A::ConstantArray{S,T,N,X,L}) where {S,T,N,X,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(ConstantArray{$Snew,$T,$N,$Xnew,$L}(A.data)))
end
@generated function Base.adjoint(A::StrideArray{S,T,N,X,0,0,V}) where {S,T<:Real,N,X,V}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(StrideArray{$Snew,$T,$N,$Xnew,0,0,$V}(A.ptr, rev(A.size), rev(A.stride), A.parent)))
end
@generated function Base.transpose(A::StrideArray{S,T,N,X,0,0,V}) where {S,T,N,X,V}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(StrideArray{$Snew,$T,$N,$Xnew,0,0,$V}(A.ptr, rev(A.size), rev(A.stride), A.parent)))
end
# @generated function Base.adjoint(A::StrideArray{S,T,N,X,SN,XN}) where {S,T<:Real,N,X,SN,XN}
#     Snew = reverse_tuple_type(S)
#     Xnew = reverse_tuple_type(X)
#     Expr(:block, Expr(:meta,:inline), :(StrideArray{$Snew,$T,$N,$Xnew,$SN,$XN}(A.data, rev(A.size), rev(A.stride))))
# end
# @generated function Base.transpose(A::StrideArray{S,T,N,X,SN,XN}) where {S,T,N,X,SN,XN}
#     Snew = reverse_tuple_type(S)
#     Xnew = reverse_tuple_type(X)
#     Expr(:block, Expr(:meta,:inline), :(StrideArray{$Snew,$T,$N,$Xnew,$SN,$XN}(A.data, rev(A.size), rev(A.stride))))
# end


# For vectors

# @inline function Base.adjoint(A::FixedSizeArray{Tuple{S},T,1,Tuple{X},L,SN,XN,V}) where {S,T<:Real,X,L,SN,XN,V}
    # FixedSizeArray{Tuple{1,S},T,2,Tuple{0,X},L,SN,XN,V}(pointer(A), A.size, A.stride, A.data)
# end
@inline function Base.transpose(A::FixedSizeArray{Tuple{S},T,1,Tuple{X},L,SN,XN,V}) where {S,T<:Real,X,L,SN,XN,V}
    FixedSizeArray{Tuple{1,S},T,2,Tuple{0,X},L,SN,XN,V}(pointer(A), A.size, A.stride, A.data)
end
# @inline function Base.adjoint(A::AbstractFixedSizeArray{Tuple{S},T,1,Tuple{1},V}) where {S,T<:Real,V}
    # PtrArray{Tuple{1,S},T,2,Tuple{0,1},0,0,V}(pointer(A), tuple(), tuple())
# end
@inline function Base.transpose(A::AbstractFixedSizeArray{Tuple{S},T,1,Tuple{1},V}) where {S,T,V}
    PtrArray{Tuple{1,S},T,2,Tuple{0,1},0,0,V}(pointer(A), tuple(), tuple())
end
# @inline function Base.adjoint(A::AbstractStrideArray{Tuple{-1},T,1,Tuple{1},1,0,V}) where {T<:Real,V}
    # PtrArray{Tuple{1,-1},T,2,Tuple{0,1},1,0,V}(pointer(A), A.size, tuple())
# end
@inline function Base.transpose(A::AbstractStrideArray{Tuple{-1},T,1,Tuple{1},1,0,V}) where {T,V}
    PtrArray{Tuple{1,-1},T,2,Tuple{0,1},1,0,V}(pointer(A), A.size, tuple())
end

# @inline function Base.adjoint(A::ConstantArray{Tuple{S},T,1,Tuple{1},L}) where {S,T<:Real,L}
    # ConstantArray{Tuple{1,S},T,2,Tuple{0,1},L}(A.data)
# end
@inline function Base.transpose(A::ConstantArray{Tuple{S},T,1,Tuple{1},L}) where {S,T,L}
    ConstantArray{Tuple{1,S},T,2,Tuple{0,1},L}(A.data)
end
# @inline function Base.adjoint(A::StrideArray{Tuple{S},T,1,Tuple{1},0,0,V}) where {S,T<:Real,V}
    # StrideArray{Tuple{1,S},T,2,Tuple{0,1},0,0,V}(A.ptr, A.size, 0, A.parent)
# end
@inline function Base.transpose(A::StrideArray{Tuple{S},T,1,Tuple{1},0,0,V}) where {S,T,V}
    StrideArray{Tuple{1,S},T,2,Tuple{0,1},0,0,V}(A.ptr, A.size, 0, A.parent)
end
# @inline function Base.adjoint(A::StrideArray{Tuple{-1},T,1,Tuple{1},1,0,V}) where {T<:Real,V}
    # StrideArray{Tuple{1,-1},T,2,Tuple{0,1},1,0,V}(A.ptr, A.size, 0, A.parent)
# end
@inline function Base.transpose(A::StrideArray{Tuple{-1},T,1,Tuple{1},1,0,V}) where {T,V}
    StrideArray{Tuple{1,-1},T,2,Tuple{0,1},1,0,V}(A.ptr, A.size, 0, A.parent)
end


Base.PermutedDimsArray(A::AbstractStrideArray{<:Any,<:Any,N}, perm::NTuple{N}) where {N} = PermutedDimsArray(A, Static(perm))
Base.PermutedDimsArray(A::AbstractArray, ::Static{perm}) where {perm} = PermutedDimsArray(A, perm)
function permuted_dims_array_expr(Sv,N,Xv,perm)
    S = tointvec(Sv)::Vector{Int}; X = tointvec(Xv)::Vector{Int}
    Sunknown = cumsum(S .== -1)
    Xunknown = cumsum(X .== -1)
    Sp = Expr(:curly, :Tuple)
    Xp = Expr(:curly, :Tuple)
    sp = Expr(:tuple)
    xp = Expr(:tuple)
    si = xi = 0;
    for n in 1:N
        p = perm[n]
        Sn = S[p]
        push!(Sp.args, Sn)
        if Sn == -1
            push!(sp.args, Expr(:ref, :st, Sunknown[p]))
        end
        Xn = X[p]
        push!(Xp.args, Xn)
        if Xn == -1
            push!(xp.args, Expr(:ref, :xt, Xunknown[p]))
        end
    end
    Sp, Xp, sp, xp
end
@generated function Base.PermutedDimsArray(A::PtrArray{S,T,N,X}, ::Static{perm}) where {S,T,N,X,perm}
     Sp, Xp, sp, xp = permuted_dims_array_expr(S, N, X, perm)
    quote
        $(Expr(:meta,:inline))
        st = A.size; xt = A.stride;
        PtrArray{$Sp,$T,$N,$Xp}(A.ptr, $sp, $xp)
    end
end
@generated function Base.PermutedDimsArray(A::FixedSizeArray{S,T,N,X}, ::Static{perm}) where {S,T,N,X,perm}
    Sp, Xp, sp, xp = permuted_dims_array_expr(S, N, X, perm)
    quote
        $(Expr(:meta,:inline))
        st = A.size; xt = A.stride;
        FixedSizeArray{$Sp,$T,$N,$Xp}(A.ptr, $sp, $xp, A.data)
    end
end
@generated function Base.PermutedDimsArray(A::StrideArray{S,T,N,X}, ::Static{perm}) where {S,T,N,X,perm}
    Sp, Xp, sp, xp = permuted_dims_array_expr(S, N, X, perm)
    quote
        $(Expr(:meta,:inline))
        st = A.size; xt = A.stride;
        StrideArray{$Sp,$T,$N,$Xp}(A.ptr, $sp, $xp, A.data)
    end
end

@generated function Base.transpose(A::PtrArray{S,T,N,X}) where {S,T,N,X}
     Sp, Xp, sp, xp = permuted_dims_array_expr(S, N, X, N:-1:1)
    quote
        $(Expr(:meta,:inline))
        st = A.size; xt = A.stride;
        PtrArray{$Sp,$T,$N,$Xp}(A.ptr, $sp, $xp)
    end
end
@generated function Base.transpose(A::FixedSizeArray{S,T,N,X}) where {S,T,N,X}
    Sp, Xp, sp, xp = permuted_dims_array_expr(S, N, X, N:-1:1)
    quote
        $(Expr(:meta,:inline))
        st = A.size; xt = A.stride;
        FixedSizeArray{$Sp,$T,$N,$Xp}(A.ptr, $sp, $xp, A.data)
    end
end
@generated function Base.transpose(A::StrideArray{S,T,N,X}) where {S,T,N,X}
    Sp, Xp, sp, xp = permuted_dims_array_expr(S, N, X, N:-1:1)
    quote
        $(Expr(:meta,:inline))
        st = A.size; xt = A.stride;
        StrideArray{$Sp,$T,$N,$Xp}(A.ptr, $sp, $xp, A.data)
    end
end
@inline Base.adjoint(A::AbstractStrideArray{<:Any,<:Real}) = transpose(A)


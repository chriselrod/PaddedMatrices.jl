@inline rev(t::NTuple{0}) = t
@inline rev(t::NTuple{1}) = t
@inline rev(t::NTuple) = reverse(t)
reverse_tuple_type(t) = reverse_tuple_type(t.parameters)
function reverse_tuple_type(t::Core.SimpleVector)
    tnew = Expr(:curly, :Tuple)
    N = length(t)
    for n ∈ 0:N-1
        push!(tnew.args, t[N - n])
    end
    tnew
end
@generated function Base.adjoint(A::AbstractFixedSizeArray{S,T,N,X,V,L}) where {S,T,N,X,V,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), Expr(:call, :(PtrArray{$Snew,$T,$N,$Xnew,0,0,$V,$L}), :(pointer(A)), tuple(), tuple()))
end
@generated function Base.transpose(A::AbstractFixedSizeArray{S,T,N,X,V,L}) where {S,T,N,X,V,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), Expr(:call, :(PtrArray{$Snew,$T,$N,$Xnew,0,0,$V,$L}), :(pointer(A)), tuple(), tuple()))
end
@generated function Base.adjoint(A::AbstractStrideArray{S,T,N,X,SN,XN,V,L}) where {S,T,N,X,SN,XN,V,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(PtrArray{$Snew,$T,$N,$Xnew,$SN,$XN,$V,$L}(pointer(A), rev(A.size), rev(A.stride))))
end
@generated function Base.transpose(A::AbstractStrideArray{S,T,N,X,SN,XN,V,L}) where {S,T,N,X,SN,XN,V,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(PtrArray{$Snew,$T,$N,$Xnew,$SN,$XN,$V,$L}(pointer(A), rev(A.size), rev(A.stride))))
end
@generated function Base.adjoint(A::ConstantArray{S,T,N,X,L}) where {S,T,N,X,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(ConstantArray{$Snew,$T,$N,$Xnew,$L}(A.data)))
end
@generated function Base.transpose(A::ConstantArray{S,T,N,X,L}) where {S,T,N,X,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(ConstantArray{$Snew,$T,$N,$Xnew,$L}(A.data)))
end
@generated function Base.adjoint(A::StrideArray{S,T,N,X,0,0,L}) where {S,T,N,X,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(StrideArray{$Snew,$T,$N,$Xnew,0,0,$L}(A.data, rev(A.size), rev(A.stride))))
end
@generated function Base.transpose(A::StrideArray{S,T,N,X,0,0,L}) where {S,T,N,X,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(StrideArray{$Snew,$T,$N,$Xnew,0,0,$L}(A.data, rev(A.size), rev(A.stride))))
end
@generated function Base.adjoint(A::StrideArray{S,T,N,X,SN,XN,L}) where {S,T,N,X,SN,XN,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(StrideArray{$Snew,$T,$N,$Xnew,$SN,$XN,$L}(A.data, rev(A.size), rev(A.stride))))
end
@generated function Base.transpose(A::StrideArray{S,T,N,X,SN,XN,L}) where {S,T,N,X,SN,XN,L}
    Snew = reverse_tuple_type(S)
    Xnew = reverse_tuple_type(X)
    Expr(:block, Expr(:meta,:inline), :(StrideArray{$Snew,$T,$N,$Xnew,$SN,$XN,$L}(A.data, rev(A.size), rev(A.stride))))
end


# For vectors

@inline function Base.adjoint(A::AbstractFixedSizeArray{Tuple{S},T,1,Tuple{1},V,L}) where {S,T,V,L}
    PtrArray{Tuple{1,S},T,2,Tuple{0,1},0,0,V,L}(pointer(A) tuple(), tuple())
end
@inline function Base.transpose(A::AbstractFixedSizeArray{Tuple{S},T,1,Tuple{1},V,L}) where {S,T,V,L}
    PtrArray{Tuple{1,S},T,2,Tuple{0,1},0,0,V,L}(pointer(A)), tuple(), tuple()))
end
@inline function Base.adjoint(A::AbstractStrideArray{Tuple{-1},T,1,Tuple{1},1,0,V,-1}) where {T,V}
    PtrArray{Tuple{1,-1},T,2,Tuple{0,1},1,0,V,-1}(pointer(A), A.size, tuple())
end
@inline function Base.transpose(A::AbstractStrideArray{Tuple{-1},T,1,Tuple{1},1,0,V,-1}) where {T,V}
    PtrArray{Tuple{1,-1},T,2,Tuple{0,1},1,0,V,-1}(pointer(A), A.size, tuple())
end

@inline function Base.adjoint(A::ConstantArray{Tuple{S},T,1,Tuple{1},L}) where {S,T,L}
    ConstantArray{Tuple{1,S},T,2,Tuple{0,1},L}(A.data)
end
@inline function Base.transpose(A::ConstantArray{Tuple{S},T,1,Tuple{1},L}) where {S,T,L}
    ConstantArray{Tuple{1,S},T,2,Tuple{0,1},L}(A.data)
end
@inline function Base.adjoint(A::StrideArray{Tuple{S},T,1,Tuple{1},0,0,L}) where {S,T,L}
    StrideArray{Tuple{1,S},T,2,Tuple{0,1},0,0,L}(A.data, A.size, 0)
end
@inline function Base.transpose(A::StrideArray{Tuple{S},T,1,Tuple{1},0,0,L}) where {S,T,L}
    StrideArray{Tuple{1,S},T,2,Tuple{0,1},0,0,L}(A.data, A.size, 0)
end
@inline function Base.adjoint(A::StrideArray{Tuple{-1},T,1,Tuple{1},1,0,-1}) where {T}
    StrideArray{Tuple{1,-1},T,2,Tuple{0,1},1,0,-1}(A.data, A.size, 0)
end
@inline function Base.transpose(A::StrideArray{Tuple{-1},T,1,Tuple{1},1,0,-1}) where {T}
    StrideArray{Tuple{1,-1},T,2,Tuple{0,1},1,0,-1}(A.data, A.size, 0)
end



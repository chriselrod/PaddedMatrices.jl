
LoopVectorization.maybestaticlength(A::AbstractStrideArray{S,T,N,X,SN,XN,V,-1}) where {S,T,N,X,SN,XN,V} = length(A)
LoopVectorization.maybestaticlength(::AbstractStrideArray{S,T,N,X,SN,XN,V,L}) where {S,T,N,X,SN,XN,V,L} = Static{L}()
LoopVectorization.maybestaticsize(::AbstractStrideArray{<:Tuple{M,Vararg}}, ::Val{1}) where {M} = Static{M}()
LoopVectorization.maybestaticsize(::AbstractStrideArray{<:Tuple{M,N,Vararg}}, ::Val{2}) where {M,N} = Static{N}()
LoopVectorization.maybestaticsize(::AbstractStrideArray{<:Tuple{M,N,K,Vararg}}, ::Val{3}) where {M,N,K} = Static{K}()
LoopVectorization.maybestaticsize(::AbstractStrideArray{<:Tuple{M,N,K,L,Vararg}}, ::Val{4}) where {M,N,K,L} = Static{L}()
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{-1,Vararg}}, ::Val{1}) = @inbounds A.size[1]
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,-1,Vararg}}, ::Val{2}) where {M} = @inbounds A.size[1]
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{-1,-1,Vararg}}, ::Val{2}) where {M} = @inbounds A.size[2]
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,N,-1,Vararg}}, ::Val{3}) where {M,N} = size(A,3)
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,N,K,-1,Vararg}}, ::Val{4}) where {M,N,K} = size(A,4)
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,N,Vararg}}, ::Val{1:2}) where {M,N} = (Static{M}(),Static{N}())
LoopVectorization.maybestaticsize(A::AbstractStrideArray{Tuple{M}}, ::Val{1:2}) where {M} = (Static{M}(),Static{1}())
@generated function LoopVectorization.maybestaticsize(A::AbstractStrideArray{S}, ::Val{I}) where {S,I}
    M = (S.parameters[I])::Int
    M == -1 ? :(size(A, $I)) : Static{M}()
end


@generated Base.size(A::AbstractFixedSizeArray{S}) where {S} = Expr(:tuple, S.parameters...)
@generated Base.strides(A::AbstractFixedSizeArray{S,T,N,X}) where {S,T,N,X} = Expr(:tuple, X.parameters...)
tup_sv_quote(T) = tup_sv_quote(T.parameters)
function tup_sv_rev_quote(T::Core.SimpleVector, s, trunc = 0)
    t = Expr(:tuple)
    N = length(T)
    i = 0
    for n ∈ 1:N-trunc
        Tₙ = (T[n])::Int
        if Tₙ == -1
            i += 0
            pushfirst!(t.args, Expr(:call, :%, Expr(:ref, Expr(:(.), :A, QuoteNode(s)), i), Int))
        else
            pushfirst!(t.args, Tₙ)
        end
    end
    Expr(:block, Expr(:meta, :inline), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), t))
end
function tup_sv_quote(T::Core.SimpleVector, s, start = 1)
    t = Expr(:tuple)
    N = length(T)
    i = 0
    for n ∈ start:N
        Tₙ = (T[n])::Int
        if Tₙ == -1
            i += 0
            push!(t.args, Expr(:call, :%, Expr(:ref, Expr(:(.), :A, QuoteNode(s)), i), Int))
        else
            push!(t.args, Tₙ)
        end
    end
    Expr(:block, Expr(:meta, :inline), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), t))
end
@generated Base.size(A::AbstractStrideArray{S}) where {S} = tup_sv_quote(S, :size)
@generated Base.strides(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X} = tup_sv_quote(X, :stride)
@generated tailstrides(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X<:Tuple{1,Vararg}} = tup_sv_quote(X, :stride, 2)
@generated revtailstrides(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X<:Tuple{1,Vararg}} = tup_sv_rev_quote(X, :stride, 1)


@inline LinearAlgebra.stride1(::AbstractStrideArray{S,T,N,<:Tuple{X,Vararg}}) where {S,T,N,X} = X
@inline LinearAlgebra.stride1(A::AbstractStrideArray{S,T,N,<:Tuple{-1,Vararg}}) where {S,T,N} = @inbounds A.stride[1]

LinearAlgebra.checksquare(::AbstractStrideMatrix{M,M}) where {M} = M
LinearAlgebra.checksquare(A::AbstractStrideMatrix{M,-1}) where {M} = ((@assert M == @inbounds A.size[1]); M)
LinearAlgebra.checksquare(A::AbstractStrideMatrix{-1,M}) where {M} = ((@assert M == @inbounds A.size[1]); M)
function LinearAlgebra.checksquare(A::AbstractStrideMatrix{-1,-1})
    M, N = @inbounds A.size[1], A.size[2]
    @assert M == N
    M
end
LinearAlgebra.checksquare(::AbstractStrideMatrix{M,N}) where {M,N} = DimensionMismatch("Matrix is not square: dimensions are ($M,$N).")

# FIXME: Need to clean up this mess
@inline val_length(::AbstractFixedSizeArray{S,T,N,X,V,L}) where {S,T,N,X,V,L} = Val{L}()

@inline is_sized(::NTuple) = true
@inline is_sized(::Type{<:NTuple}) = true
@inline is_sized(::Number) = true
@inline is_sized(::Type{<:Number}) = true
@inline type_length(::NTuple{N}) where {N} = N
@inline type_length(::Type{<:NTuple{N}}) where {N} = N
@inline param_type_length(::NTuple{N}) where {N} = N
@inline param_type_length(::Type{<:NTuple{N}}) where {N} = N
@inline type_length(::Number) = 1
@inline type_length(::Type{<:Number}) = 1
@inline type_length(x::AbstractArray) = length(x)
@inline param_type_length(::Number) = 1
@inline param_type_length(::Type{<:Number}) = 1

@generated function type_length(nt::NT) where {NT <: NamedTuple}
    P = first(NT.parameters)
    q = quote s = 0 end
    for p ∈ P
        push!(q.args, :(s += type_length(nt.$p)))
    end
    q
end


@inline is_sized(::AbstractFixedSizeArray) = true
@inline is_sized(::Type{<:AbstractFixedSizeArray}) = true
@generated is_sized(::AbstractStrideArray{S,T,N,X}) where {S,T,N,X} = !(anyneg1(S.parameters) || anyneg1(X.parameters))
@generated is_sized(::Type{<:AbstractStrideArray{S,T,N,X}}) where {S,T,N,X} = !(anyneg1(S.parameters) || anyneg1(X.parameters))
function simple_vec_prod(sv::Core.SimpleVector)
    p = 1
    for n ∈ 1:length(sv)
        p *= (sv[n])::Int
    end
    p
end
anyneg1(sv::Core.SimpleVector) = any(s -> s == -1, tointvec(sv))
tointvec(sv) = tointvec(sv.parameters)
function tointvec(sv::Core.SimpleVector)
    v = Vector{Int}(undef, length(sv))
    for i ∈ eachindex(v)
        v[i] = sv[i]
    end
    v
end

@inline type_length(::AbstractFixedSizeArray{S,T,N,X,V,L}) where {S,T,N,X,V,L} = L
@inline type_length(::Type{<:AbstractFixedSizeArray{S,T,N,X,V,L}}) where {S,T,N,X,V,L} = L
@generated param_type_length(::AbstractFixedSizeArray{S}) where {S} = simple_vec_prod(S.parameters)
@generated param_type_length(::Type{<:AbstractFixedSizeArray{S}}) where {S} = simple_vec_prod(S.parameters)
@inline is_sized(::Any) = false




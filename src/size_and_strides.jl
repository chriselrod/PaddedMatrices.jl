
function sv_pos_product(SV::Core.SimpleVector)
    L = 1
    SV = S.parameters
    for i in eachindex(SV)
        sv = (SV[i])::Int
        sv > 0 || return -1
        L *= sv
    end
    L
end

@generated function LoopVectorization.maybestaticlength(A::AbstractStrideArray{S,T,N,X,SN,XN,V}) where {S,T,N,X,SN,XN,V}
    L = sv_pos_product(S.parameters)
    L > 0 ? Static{L}() : :(length(A))
end
LoopVectorization.maybestaticsize(::AbstractStrideArray{<:Tuple{M,Vararg}}, ::Val{1}) where {M} = Static{M}()
LoopVectorization.maybestaticsize(::AbstractStrideArray{<:Tuple{M,N,Vararg}}, ::Val{2}) where {M,N} = Static{N}()
LoopVectorization.maybestaticsize(::AbstractStrideArray{<:Tuple{M,N,K,Vararg}}, ::Val{3}) where {M,N,K} = Static{K}()
LoopVectorization.maybestaticsize(::AbstractStrideArray{<:Tuple{M,N,K,L,Vararg}}, ::Val{4}) where {M,N,K,L} = Static{L}()
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{-1,Vararg}}, ::Val{1}) = @inbounds A.size[1]
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,-1,Vararg}}, ::Val{2}) where {M} = @inbounds A.size[1]
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{-1,-1,Vararg}}, ::Val{2}) = @inbounds A.size[2]
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,N,-1,Vararg}}, ::Val{3}) where {M,N} = size(A,3)
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,N,K,-1,Vararg}}, ::Val{4}) where {M,N,K} = size(A,4)
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{-1,N,Vararg}}, ::Val{1:2}) where {N} = (A.size[1],Static{N}())
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,-1,Vararg}}, ::Val{1:2}) where {M} = (Static{M}(),A.size[1])
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{-1,-1,Vararg}}, ::Val{1:2}) = (A.size[1], A.size[2])
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,N,Vararg}}, ::Val{1:2}) where {M,N} = (Static{M}(),Static{N}())
LoopVectorization.maybestaticsize(A::AbstractStrideArray{Tuple{M}}, ::Val{1:2}) where {M} = (Static{M}(),Static{1}())
@generated function LoopVectorization.maybestaticsize(A::AbstractStrideArray{S}, ::Val{I}) where {S,I}
    M = (S.parameters[I])::Int
    M == -1 ? :(size(A, $I)) : Static{M}()
end


@generated Base.size(A::AbstractFixedSizeArray{S}) where {S} = Expr(:tuple, S.parameters...)
@generated Base.strides(A::AbstractFixedSizeArray{S,T,N,X}) where {S,T,N,X} = Expr(:tuple, X.parameters...)
@inline Base.stride(A::AbstractStrideArray, i::Int) = strides(A)[i]
tup_sv_quote(T) = tup_sv_quote(T.parameters)
function tup_sv_rev_quote(T::Core.SimpleVector, s, trunc = 0)
    t = Expr(:tuple)
    N = length(T)
    i = 0
    for n ∈ 1:N-trunc
        Tₙ = (T[n])::Int
        if Tₙ == -1
            i += 0
            pushfirst!(t.args, Expr(:ref, Expr(:(.), :A, QuoteNode(s)), i))
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
            i += 1
            push!(t.args, Expr(:ref, Expr(:(.), :A, QuoteNode(s)), i))
        else
            push!(t.args, Tₙ)
        end
    end
    Expr(:block, Expr(:meta, :inline), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), t))
end
@generated Base.size(A::AbstractStrideArray{S}) where {S} = tup_sv_quote(S.parameters, :size)
@generated Base.strides(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X} = tup_sv_quote(X.parameters, :stride)
@generated tailstrides(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X<:Tuple{1,Vararg}} = tup_sv_quote(X.parameters, :stride, 2)
@generated revtailstrides(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X<:Tuple{1,Vararg}} = tup_sv_rev_quote(X.parameters, :stride, 1)


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

@generated type_length(::AbstractFixedSizeArray{S}) where {S} = simple_vec_prod(S.parameters)
@generated type_length(::Type{<:AbstractFixedSizeArray{S}}) where {S} = simple_vec_prod(S.parameters)
@generated param_type_length(::AbstractFixedSizeArray{S}) where {S} = simple_vec_prod(S.parameters)
@generated param_type_length(::Type{<:AbstractFixedSizeArray{S}}) where {S} = simple_vec_prod(S.parameters)
@inline is_sized(::Any) = false


@generated function Base.axes(A::AbstractStrideArray{S}) where {S}
    retexpr = Expr(:tuple,)
    SV = S.parameters
    ioff = 1
    for n in 1:length(SV)
        sn = (SV[n])::Int
        if sn == -1
            push!(retexpr.args, :(Base.OneTo(A.size[$ioff])))
            ioff += 1
        else
            push!(retexpr.args, :(VectorizationBase.StaticUnitRange{1,$sn}()))
        end
    end
    retexpr
end

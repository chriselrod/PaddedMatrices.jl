
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
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{-1,Vararg}}, ::Val{1}) = @inbounds size_tuple(A)[1]
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,-1,Vararg}}, ::Val{2}) where {M} = @inbounds size_tuple(A)[1]
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{-1,-1,Vararg}}, ::Val{2}) = @inbounds size_tuple(A)[2]
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,N,-1,Vararg}}, ::Val{3}) where {M,N} = size(A,3)
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,N,K,-1,Vararg}}, ::Val{4}) where {M,N,K} = size(A,4)
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{-1,N,Vararg}}, ::Val{1:2}) where {N} = (size_tuple(A)[1],Static{N}())
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{M,-1,Vararg}}, ::Val{1:2}) where {M} = (Static{M}(),size_tuple(A)[1])
LoopVectorization.maybestaticsize(A::AbstractStrideArray{<:Tuple{-1,-1,Vararg}}, ::Val{1:2}) = (size_tuple(A)[1], size_tuple(A)[2])
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

function makeref(s, i)
    # ref = Expr(:ref, Expr(:(.), :A, QuoteNode(s === :bytestride ? :stride : s)), i)
    f = s === :size ? :size_tuple : :stride_tuple
    ref = Expr(:ref, Expr(:call, f, :A), i)
    if s === :stride
        ref = Expr(:call, :>>>, ref, Expr(:call, :(VectorizationBase.intlog2), Expr(:call, :sizeof, :T)))
    end
    ref
end
function tup_sv_rev_quote(T::Core.SimpleVector, s, trunc = 0)
    t = Expr(:tuple)
    N = length(T)
    i = 0
    for n ∈ 1:N-trunc
        Tₙ = (T[n])::Int
        if Tₙ == -1
            i += 1
            pushfirst!(t.args, makeref(s, i))
        elseif s === :bytestride
            pushfirst!(t.args, Expr(:call, :vmul, Expr(:call, :sizeof, :T, Tₙ)))
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
            push!(t.args, makeref(s, i))
        elseif s === :bytestride
            push!(t.args, Expr(:call, :vmul, Expr(:call, :sizeof, :T, Tₙ)))
        else
            push!(t.args, Tₙ)
        end
    end
    Expr(:block, Expr(:meta, :inline), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), t))
end
@generated Base.size(A::AbstractStrideArray{S}) where {S} = tup_sv_quote(S.parameters, :size)
@generated Base.strides(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X} = tup_sv_quote(X.parameters, :stride)
@generated tailstrides(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X<:Tuple{1,Vararg}} = tup_sv_quote(X.parameters, :stride, 2)
@generated revtailstrides(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X} = tup_sv_rev_quote(X.parameters, :stride, 1)
@generated bytestrides(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X} = tup_sv_quote(X.parameters, :bytestride)
@generated revbytestrides(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X} = tup_sv_rev_quote(X.parameters, :bytestride)
@generated tailbytestrides(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X<:Tuple{1,Vararg}} = tup_sv_quote(X.parameters, :bytestride, 2)
@generated revtailbytestrides(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X} = tup_sv_rev_quote(X.parameters, :bytestride, 1)

@inline LinearAlgebra.stride1(::AbstractStrideArray{S,T,N,<:Tuple{X,Vararg}}) where {S,T,N,X} = X
@inline LinearAlgebra.stride1(A::AbstractStrideArray{S,T,N,<:Tuple{-1,Vararg}}) where {S,T,N} = @inbounds A.stride[1]

LinearAlgebra.checksquare(::AbstractStrideMatrix{M,M}) where {M} = M
LinearAlgebra.checksquare(A::AbstractStrideMatrix{M,-1}) where {M} = ((@assert M == @inbounds size_tuple(A)[1]); M)
LinearAlgebra.checksquare(A::AbstractStrideMatrix{-1,M}) where {M} = ((@assert M == @inbounds size_tuple(A)[1]); M)
function LinearAlgebra.checksquare(A::AbstractStrideMatrix{-1,-1})
    M, N = @inbounds size_tuple(A)[1], size_tuple(A)[2]
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

# @generated type_length(::AbstractFixedSizeArray{S}) where {S} = simple_vec_prod(S.parameters)
# @generated type_length(::Type{<:AbstractFixedSizeArray{S}}) where {S} = simple_vec_prod(S.parameters)
# @generated param_type_length(::AbstractFixedSizeArray{S}) where {S} = simple_vec_prod(S.parameters)
# @generated param_type_length(::Type{<:AbstractFixedSizeArray{S}}) where {S} = simple_vec_prod(S.parameters)

@generated function memory_length(::Type{<:AbstractFixedSizeArray{S,T,N,X}}) where {S,T,N,X}
    Xvec = Int[X.parameters...]
    Xmax, i = findmax(Xvec)
    (Xmax * S.parameters[i])::Int
end
@inline memory_length(::A) where {A <: AbstractStrideArray} = memory_length(A)
@generated memory_length_val(::Type{A}) where {A} = Val{memory_length(A)}()
@generated function number_elements(::Type{<:AbstractFixedSizeArray{S,T,N,X}}) where {S,T,N,X}
    simple_vec_prod(S.parameters)
end
# @inline number_elements(::A) where {A <: AbstractStrideArray} = number_elements(A)
@inline number_elements(::AbstractStrideArray) = length(A)
@generated number_elements_val(::Type{A}) where {A} = Val{number_elements(A)}()

@inline is_sized(::Any) = false


@generated function Base.axes(A::AbstractStrideArray{S}) where {S}
    retexpr = Expr(:tuple,)
    SV = S.parameters
    ioff = 1
    for n in 1:length(SV)
        sn = (SV[n])::Int
        if sn == -1
            push!(retexpr.args, :(Base.OneTo(size_tuple(A)[$ioff])))
            ioff += 1
        else
            push!(retexpr.args, :(VectorizationBase.StaticUnitRange{1,$sn}()))
        end
    end
    Expr(:block, Expr(:meta,:inline), retexpr)
end

@inline function Base.eachindex(::Base.IndexLinear, ::A) where {A <: AbstractFixedSizeArray}
    Static{1}():Static(number_elements_val(A))
end



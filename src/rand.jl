using VectorizedRNG: local_pcg

Random.rand!(A::AbstractMutableFixedSizeArray) = (rand!(local_pcg(), flatvector(A)); A)
Random.rand!(A::AbstractMutableFixedSizeArray{S,T}, l::T, u::T) where {S,T} = (rand!(local_pcg(), flatvector(A), l, u); A)
# Specific to avoid ambiguities
Random.randn!(A::AbstractMutableFixedSizeArray) = (randn!(local_pcg(), flatvector(A)); A)
function Random.randn!(
    A::AbstractMutableFixedSizeArray{S,T,N,X,L},
    B::Union{T,<:AbstractMutableFixedSizeArray{S,T,N,X,L}}
) where {S,T,N,X,L}
    randn!(local_pcg(), flatvector(A), flatvector(B)); A
end
function Random.randn!(
    A::AbstractMutableFixedSizeArray{S,T,N,X,L},
    B::Union{T,<:AbstractMutableFixedSizeArray{S,T,N,X,L}},
    C::Union{T,<:AbstractMutableFixedSizeArray{S,T,N,X,L}}
) where {S,T,N,X,L}
    randn!(local_pcg(), flatvector(A), flatvector(B), flatvector(C)); A
end

# @generated function Random.randn(rng::VectorizedRNG.AbstractPCG{P}, ::Type{<:ConstantFixedSizeArray{S,T}}) where {P,S,T<:Union{Float32,Float64}}
#     N,X,L = calc_NPL(S.parameters, T)
#     quote
#         A = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
#         $(rand_mutable_fixed_size_expr(L, T, P, :randn))
#         ConstantFixedSizeArray(A)
#     end
# end
# function Random.randn(::Type{<:ConstantFixedSizeArray{S,T}}) where {S,T<:Union{Float32,Float64}}
#     randn(local_pcg(), ConstantFixedSizeArray{S,T})
# end
@generated function Random.randn(rng::VectorizedRNG.AbstractPCG, ::Static{S}) where {S}
    if isa(S, Integer)
        ST = Tuple{S}
    else
        ST = S
    end
    quote
        $(Expr(:meta,:inline))
        #randn(rng, ConstantFixedSizeArray{$ST,Float64})
        randn(rng, FixedSizeArray{$ST,Float64})
    end
end
@generated function Random.randn(::Static{S}) where {S}
    if isa(S, Integer)
        ST = Tuple{S}
    else
        ST = S
    end
    quote
        $(Expr(:meta,:inline))
        randn(local_pcg(), FixedSizeArray{$ST,Float64})
    end
end


Random.randexp!(A::AbstractMutableFixedSizeArray) = randexp!(local_pcg(), flatvector(A))
function Random.randexp!(
    A::AbstractMutableFixedSizeArray{S,T,N,X,L},
    B::Union{T,<:AbstractMutableFixedSizeArray{S,T,N,X,L}},
    C::Union{T,<:AbstractMutableFixedSizeArray{S,T,N,X,L}}
) where {P,S,T<:Union{Float32,Float64},N,X,L}
    randexp!(local_pcg(), flatvector(A), flatvector(B), flatvector(C)); A
end
function Random.randexp!(
    A::AbstractMutableFixedSizeArray{S,T,N,X,L},
    B::Union{T,<:AbstractMutableFixedSizeArray{S,T,N,X,L}}
) where {P,S,T<:Union{Float32,Float64},N,X,L}
    randexp!(local_pcg(), flatvector(A), flatvector(B)); A
end

function Random.rand(::Type{<: FixedSizeArray{S,T}}) where {S,T}
    rand!(FixedSizeArray{S,T}(undef))
end
function Random.randn(::Type{<: FixedSizeArray{S,T}}) where {S,T}
    randn!(FixedSizeArray{S,T}(undef))
end
function Random.randexp(::Type{<: FixedSizeArray{S,T}}) where {S,T}
    randexp!(FixedSizeArray{S,T}(undef))
end
function Random.rand(rng, ::Type{<: FixedSizeArray{S,T}}) where {S,T}
    A = FixedSizeArray{S,T}(undef)
    rand!(rng, flatvector(A)); A
end
function Random.randn(rng, ::Type{<: FixedSizeArray{S,T}}) where {S,T}
    A = FixedSizeArray{S,T}(undef)
    randn!(rng, flatvector(A)); A
end
function Random.randexp(rng, ::Type{<: FixedSizeArray{S,T}}) where {S,T}
    A = FixedSizeArray{S,T}(undef)
    randexp!(rng, flatvector(A)); A
end

function Random.rand(::Type{<: FixedSizeArray{S,T}}, l::T, u::T) where {S,T}
    rand!(FixedSizeArray{S,T}(undef), l, u)
end
function Random.randn(::Type{<: FixedSizeArray{S,T}}, μ::T, σ::T) where {S,T}
    randn!(FixedSizeArray{S,T}(undef), μ, σ)
end
function Random.randexp(::Type{<: FixedSizeArray{S,T}}, β::T, l::T) where {S,T}
    randexp!(FixedSizeArray{S,T}(undef), β, l)
end
function Random.rand(rng, ::Type{<: FixedSizeArray{S,T}}, l::T, u::T) where {S,T}
    A = FixedSizeArray{S,T}(undef)
    rand!(rng, flatvector(A), l, u); A
end
function Random.randn(rng, ::Type{<: FixedSizeArray{S,T}}, μ::T, σ::T) where {S,T}
    A = FixedSizeArray{S,T}(undef)
    randn!(rng, flatvector(A), μ, σ); A
end
function Random.randexp(rng, ::Type{<: FixedSizeArray{S,T}}, β::T, l::T) where {S,T}
    A = FixedSizeArray{S,T}(undef)
    randexp!(rng, flatvector(A), β, l); A
end
function Random.randn(::Type{<: FixedSizeArray{S,T}}, σ::T) where {S,T}
    randn!(FixedSizeArray{S,T}(undef), σ)
end
function Random.randexp(::Type{<: FixedSizeArray{S,T}}, β::T) where {S,T}
    randexp!(FixedSizeArray{S,T}(undef), β)
end
function Random.randn(rng, ::Type{<: FixedSizeArray{S,T}}, σ::T) where {S,T}
    A = FixedSizeArray{S,T}(undef)
    randn!(rng, flatvector(A), σ); A
end
function Random.randexp(rng, ::Type{<: FixedSizeArray{S,T}}, β::T) where {S,T}
    A = FixedSizeArray{S,T}(undef)
    randexp!(rng, flatvector(A), β); A
end


compatible(::Type{T}, x::T) where {T} = x
compatible(::Type{T}, x::AbstractArray{T}) where {T} = x
compatible(::Type{T}, x::AbstractArray) where {T} = convert(Array{T}, x)
compatible(::Type{T}, x) where {T} = convert(T, x)
function rand_expr(expr, R, args...)
    # @show expr.args
    N = length(expr.args)
    n = 2
    randtypes = Set((:Float32,:Float64,:Int,:Int32,:Int64,:UInt,:UInt32,:UInt64))
    arg2 = expr.args[2]
    if arg2 ∈ randtypes
        T = arg2
        n += 1
    elseif arg2 isa Expr && arg2.head === :($)
        T = esc(first(arg2.args))
        n += 1
    else
        T = Float64
    end
    return :( $(expr.args[1])( $(R){Tuple{$(esc.(expr.args[n:end])...)}, $T}, $([:(PaddedMatrices.compatible($T,$arg)) for arg in args]...)  )  )
end

# macro Mutable(expr)
#     rand_expr(expr, :FixedSizeArray)
# end
# macro Constant(expr)
#     rand_expr(expr, :ConstantFixedSizeArray)
# end

macro Mutable(expr, args...)
    rand_expr(expr, :FixedSizeArray, args...)
end
# macro Constant(expr, args...)
    # rand_expr(expr, :ConstantFixedSizeArray, args...)
# end


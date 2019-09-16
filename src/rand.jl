function calculate_L_from_size(S, T)
    nrow = S[1]
    N = length(S)
    W, Wshift = VectorizationBase.pick_vector_width_shift(nrow, T)
    rem = nrow & (W - 1)
    P = rem == 0 ? nrow : nrow + W - rem
    L = P
    for n ∈ 2:N
        L *= S[n]
    end
    P, L
end
function rand_iter_expr(V, mulwith_B, addto_C, rand_extract_expr, iter_offset, rrepiter = 0)
    if addto_C == -1 && mulwith_B == -1
        rem_expr = rand_extract_expr
    elseif addto_C == -1
        if mulwith_B == 0
            rem_expr = :( vmul(vB, $rand_extract_expr) )
        elseif mulwith_B == 1
            rem_expr = :( vmul(vload($V, ptr_B + $iter_offset), $rand_extract_expr) )
        end
    elseif mulwith_B == -1
        if addto_C == 0
            rem_expr = :( vadd(vC, $rand_extract_expr) )
        elseif addto_C == 1
            rem_expr = :( vadd(vload($V, ptr_C + $iter_offset), $rand_extract_expr) )
        end
    else
        vB = mulwith_B == 0 ? :vB : Symbol(:vB_, rrepiter)
        vC = addto_C == 0 ? :vC : Symbol(:vB_, rrepiter)
        rem_expr = :(vmuladd($vB, $rand_extract_expr, $vC))
    end
    rem_expr
end
function rand_mutable_fixed_size_expr(L, T, P, randfunc, mulwith_B::Int = -1, addto_C::Int = -1, args...)
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(L, T)
    V = Vec{W,T}
    PW = P*W
    if PW > L
        P = (L+W-1) >> Wshift
        PW = P*W
    end
    unifs = gensym(:unifs)
    nrep, r = divrem(L, PW)
    float_q = :($randfunc(rng, NTuple{$P,$V}, $(args...), Val{VectorizedRNG.RXS_M_XS}()))
    store_expr = quote end
    pointer_quote = quote ptr_A = pointer(A) end
    if mulwith_B == 1
        push!(pointer_quote.args, :(ptr_B = pointer(B)))
        for p ∈ 1:P
            vB = Symbol(:vB_,p)
            push!(store_expr.args, :($vB = vload($V, ptr_B + $(sizeof(T)*W) * (i*$(P) + $(p-1)))))
        end
    elseif mulwith_B == 0
        push!(pointer_quote.args, :(vB = vbroadcast($V, B)))
    end
    if addto_C == 1
        push!(pointer_quote.args, :(ptr_C = pointer(C)))
        for p ∈ 1:P
            vC = Symbol(:vC_,p)
            push!(store_expr.args, :($vC = vload($V, ptr_C + $(sizeof(T)*W) * (i*$(P) + $(p-1)))))
        end
    elseif addto_C == 0
        push!(pointer_quote.args, :(vC = vbroadcast($V, C)))
    end
    if mulwith_B >= 0 && addto_C >= 0
        for p ∈ 1:P
            vB = mulwith_B == 0 ? :vB : Symbol(:vB_,p)
            vC = addto_C == 0 ? :vC : Symbol(:vC_,p)
            push!(store_expr.args, :(vstore!(ptr_A + $(sizeof(T)*W) * (i*$(P) + $(p-1)), vmuladd($vB, $unifs[$p], $vC))))
        end
    elseif mulwith_B >= 0 && addto_C == -1
        for p ∈ 1:P
            vB = mulwith_B == 0 ? :vB : Symbol(:vB_,p)
            push!(store_expr.args, :(vstore!(ptr_A + $(sizeof(T)*W) * (i*$(P) + $(p-1)), vmul($vB, $unifs[$p]))))
        end
    elseif addto_C == 1 && mulwith_B == -1
        for p ∈ 1:P
            vC = addto_C == 0 ? :vC : Symbol(:vC_,p)
            push!(store_expr.args, :(vstore!(ptr_A + $(sizeof(T)*W) * (i*$(P) + $(p-1)), vadd($vC, $unifs[$p]))))
        end
    else
        for p ∈ 1:P
            push!(store_expr.args, :(vstore!(ptr_A + $(sizeof(T)*W) * (i*$(P) + $(p-1)), $unifs[$p])))
        end
    end
    if nrep > 0
        q = quote
            $pointer_quote
            for i ∈ 0:$(nrep-1)
                $unifs = $float_q
                $store_expr
            end
        end
    else
        q = pointer_quote
    end
    if r > 0
        rrep, rrem = divrem(r, W)
        Nremv = rrep + ( rrem > 0 )
        u_sym = gensym(:u_rem)
        push!(q.args, :($u_sym = $randfunc(rng, NTuple{$Nremv,$V}, $(args...))))#VectorizedRNG.RXS_M_XS)))
        for rrepiter ∈ 0:rrep-1
            iter_offset = sizeof(T)*W*(nrep*P + rrepiter)
            rand_extract_expr = :($u_sym[$(rrepiter+1)])
            rem_expr = rand_iter_expr(V, mulwith_B, addto_C, rand_extract_expr, iter_offset, rrepiter)
            push!(q.args, :(vstore!(ptr_A + $iter_offset, $rem_expr)))
        end
        if rrem > 0
            iter_offset = sizeof(T)*W*(nrep*P + rrep)
            rem_expr = rand_iter_expr(V, mulwith_B, addto_C, :($u_sym[$Nremv]), iter_offset, 0)
            mask = VectorizationBase.mask(T, rrem)
            push!(q.args, :( vstore!(ptr_A + $iter_offset, $rem_expr, $mask) ))
        end
    end
    push!(q.args, :A)
    q
end


@generated function Random.rand!(rng::VectorizedRNG.AbstractPCG{P}, A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}) where {P,S,T<:Union{Float32,Float64},N,R,L}
    rand_mutable_fixed_size_expr(L, T, P, :rand)
end
Random.rand!(A::AbstractMutableFixedSizePaddedArray) = rand!(VectorizedRNG.GLOBAL_vPCG, A)
@generated function Random.rand!(rng::VectorizedRNG.AbstractPCG{P}, A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}, l::T, u::T) where {P,S,T<:Union{Float32,Float64},N,R,L}
    rand_mutable_fixed_size_expr(L, T, P, :rand, -1, -1, :l, :u)
end
Random.rand!(A::AbstractMutableFixedSizePaddedArray{S,T}, l::T, u::T) where {S,T} = rand!(VectorizedRNG.GLOBAL_vPCG, A, l, u)
@generated function Random.rand(rng::VectorizedRNG.AbstractPCG{P}, ::Type{ <: ConstantFixedSizePaddedArray{S,T}}) where {P,S,T<:Union{Float32,Float64}}
    N,R,L = calc_NPL(S.parameters, T)
    quote
        A = MutableFixedSizePaddedArray{$S,$T,$N,$R,$L}(undef)
        $(rand_mutable_fixed_size_expr(L, T, P, :rand))
        ConstantFixedSizePaddedArray(A)
    end
end
function Random.rand(::Type{ <: ConstantFixedSizePaddedArray{S,T}}) where {S,T<:Union{Float32,Float64}}
    rand(VectorizedRNG.GLOBAL_vPCG, ConstantFixedSizePaddedArray{S,T})
end

# @generated function Random.randn!(rng::VectorizedRNG.AbstractPCG{P}, A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}) where {S,P,T<:Union{Float32,Float64},N,R,L}
#     rand_mutable_fixed_size_expr(L, T, P, :randn)
# end
@generated function Random.randn!(rng::VectorizedRNG.AbstractPCG{P}, A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}) where {S,P,T<:Union{Float32,Float64},N,R,L}
    rand_mutable_fixed_size_expr(L, T, P, :randn)
end
@generated function Random.randn!(
    rng::VectorizedRNG.AbstractPCG{P},
    A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L},
    C::Union{T,<:AbstractMutableFixedSizePaddedArray{S,T,N,R,L}},
    B::Union{T,<:AbstractMutableFixedSizePaddedArray{S,T,N,R,L}}
) where {S,P,T<:Union{Float32,Float64},N,R,L}
    rand_mutable_fixed_size_expr(L, T, P, :randn, B === T ? 0 : 1, C === T ? 0 : 1)
end
@generated function Random.randn!(
    rng::VectorizedRNG.AbstractPCG{P},
    A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L},
    B::Union{T,<:AbstractMutableFixedSizePaddedArray{S,T,N,R,L}}
) where {S,P,T<:Union{Float32,Float64},N,R,L}
    rand_mutable_fixed_size_expr(L, T, P, :randn, B === T ? 0 : 1)
end
# Specific to avoid ambiguities
Random.randn!(A::AbstractMutableFixedSizePaddedArray) = randn!(VectorizedRNG.GLOBAL_vPCG, A)
function Random.randn!(
    A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L},
    B::Union{T,<:AbstractMutableFixedSizePaddedArray{S,T,N,R,L}}
) where {S,T,N,R,L}
    randn!(VectorizedRNG.GLOBAL_vPCG, A, B)
end
function Random.randn!(
    A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L},
    B::Union{T,<:AbstractMutableFixedSizePaddedArray{S,T,N,R,L}},
    C::Union{T,<:AbstractMutableFixedSizePaddedArray{S,T,N,R,L}}
) where {S,T,N,R,L}
    randn!(VectorizedRNG.GLOBAL_vPCG, A, B, C)
end

@generated function Random.randn(rng::VectorizedRNG.AbstractPCG{P}, ::Type{<:ConstantFixedSizePaddedArray{S,T}}) where {P,S,T<:Union{Float32,Float64}}
    N,R,L = calc_NPL(S.parameters, T)
    quote
        A = MutableFixedSizePaddedArray{$S,$T,$N,$R,$L}(undef)
        $(rand_mutable_fixed_size_expr(L, T, P, :randn))
        ConstantFixedSizePaddedArray(A)
    end
end
function Random.randn(::Type{<:ConstantFixedSizePaddedArray{S,T}}) where {S,T<:Union{Float32,Float64}}
    randn(VectorizedRNG.GLOBAL_vPCG, ConstantFixedSizePaddedArray{S,T})
end
@generated function Random.randn(rng::VectorizedRNG.AbstractPCG, ::Static{S}) where {S}
    if isa(S, Integer)
        ST = Tuple{S}
    else
        ST = S
    end
    quote
        $(Expr(:meta,:inline))
        #randn(rng, ConstantFixedSizePaddedArray{$ST,Float64})
        randn(rng, MutableFixedSizePaddedArray{$ST,Float64})
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
        randn(VectorizedRNG.GLOBAL_vPCG, MutableFixedSizePaddedArray{$ST,Float64})
    end
end


@generated function Random.randexp!(rng::VectorizedRNG.AbstractPCG{P}, A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}) where {P,S,T<:Union{Float32,Float64},N,R,L}
    rand_mutable_fixed_size_expr(L, T, P, :randexp)
end
@generated function Random.randexp!(
    rng::VectorizedRNG.AbstractPCG{P},
    A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L},
    B::Union{T,<:AbstractMutableFixedSizePaddedArray{S,T,N,R,L}},
    C::Union{T,<:AbstractMutableFixedSizePaddedArray{S,T,N,R,L}}
) where {P,S,T<:Union{Float32,Float64},N,R,L}
    rand_mutable_fixed_size_expr(L, T, P, :randexp, B === T ? 0 : 1, C === T ? 0 : 1)
end
@generated function Random.randexp!(
    rng::VectorizedRNG.AbstractPCG{P},
    A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L},
    B::Union{T,<:AbstractMutableFixedSizePaddedArray{S,T,N,R,L}}
) where {P,S,T<:Union{Float32,Float64},N,R,L}
    rand_mutable_fixed_size_expr(L, T, P, :randexp, B === T ? 0 : 1)
end
Random.randexp!(A::AbstractMutableFixedSizePaddedArray) = randexp!(VectorizedRNG.GLOBAL_vPCG, A, args...)
function Random.randexp!(
    A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L},
    B::Union{T,<:AbstractMutableFixedSizePaddedArray{S,T,N,R,L}},
    C::Union{T,<:AbstractMutableFixedSizePaddedArray{S,T,N,R,L}}
) where {P,S,T<:Union{Float32,Float64},N,R,L}
    randexp!(VectorizedRNG.GLOBAL_vPCG, A, B, C)
end
function Random.randexp!(
    A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L},
    B::Union{T,<:AbstractMutableFixedSizePaddedArray{S,T,N,R,L}}
) where {P,S,T<:Union{Float32,Float64},N,R,L}
    randexp!(VectorizedRNG.GLOBAL_vPCG, A, B)
end


@generated function Random.randexp(rng::VectorizedRNG.AbstractPCG{P}, ::Type{<:ConstantFixedSizePaddedArray{S,T}}) where {P,S,T<:Union{Float32,Float64}}
    N,R,L = calc_NPL(S.parameters, T)
    quote
        A = MutableFixedSizePaddedArray{$S,$T,$N,$R,$L}(undef)
        $(rand_mutable_fixed_size_expr(L, T, P, :randexp))
        ConstantFixedSizePaddedArray(A)
    end
end
function Random.randexp(::Type{<:ConstantFixedSizePaddedArray{S,T}}) where {S,T<:Union{Float32,Float64}}
    randexp(VectorizedRNG.GLOBAL_vPCG, ConstantFixedSizePaddedArray{S,T})
end

function Random.rand(::Type{<: MutableFixedSizePaddedArray{S,T}}) where {S,T}
    rand!(MutableFixedSizePaddedArray{S,T}(undef))
end
function Random.randn(::Type{<: MutableFixedSizePaddedArray{S,T}}) where {S,T}
    randn!(MutableFixedSizePaddedArray{S,T}(undef))
end
function Random.randexp(::Type{<: MutableFixedSizePaddedArray{S,T}}) where {S,T}
    randexp!(MutableFixedSizePaddedArray{S,T}(undef))
end
function Random.rand(rng, ::Type{<: MutableFixedSizePaddedArray{S,T}}) where {S,T}
    rand!(rng, MutableFixedSizePaddedArray{S,T}(undef))
end
function Random.randn(rng, ::Type{<: MutableFixedSizePaddedArray{S,T}}) where {S,T}
    randn!(rng, MutableFixedSizePaddedArray{S,T}(undef))
end
function Random.randexp(rng, ::Type{<: MutableFixedSizePaddedArray{S,T}}) where {S,T}
    randexp!(rng, MutableFixedSizePaddedArray{S,T}(undef))
end

function Random.rand(::Type{<: MutableFixedSizePaddedArray{S,T}}, l::T, u::T) where {S,T}
    rand!(MutableFixedSizePaddedArray{S,T}(undef), l, u)
end
function Random.randn(::Type{<: MutableFixedSizePaddedArray{S,T}}, μ::T, σ::T) where {S,T}
    randn!(MutableFixedSizePaddedArray{S,T}(undef), μ, σ)
end
function Random.randexp(::Type{<: MutableFixedSizePaddedArray{S,T}}, β::T, l::T) where {S,T}
    randexp!(MutableFixedSizePaddedArray{S,T}(undef), β, l)
end
function Random.rand(rng, ::Type{<: MutableFixedSizePaddedArray{S,T}}, l::T, u::T) where {S,T}
    rand!(rng, MutableFixedSizePaddedArray{S,T}(undef), l, u)
end
function Random.randn(rng, ::Type{<: MutableFixedSizePaddedArray{S,T}}, μ::T, σ::T) where {S,T}
    randn!(rng, MutableFixedSizePaddedArray{S,T}(undef), μ, σ)
end
function Random.randexp(rng, ::Type{<: MutableFixedSizePaddedArray{S,T}}, β::T, l::T) where {S,T}
    randexp!(rng, MutableFixedSizePaddedArray{S,T}(undef), β, l)
end
function Random.randn(::Type{<: MutableFixedSizePaddedArray{S,T}}, σ::T) where {S,T}
    randn!(MutableFixedSizePaddedArray{S,T}(undef), σ)
end
function Random.randexp(::Type{<: MutableFixedSizePaddedArray{S,T}}, β::T) where {S,T}
    randexp!(MutableFixedSizePaddedArray{S,T}(undef), β)
end
function Random.randn(rng, ::Type{<: MutableFixedSizePaddedArray{S,T}}, σ::T) where {S,T}
    randn!(rng, MutableFixedSizePaddedArray{S,T}(undef), σ)
end
function Random.randexp(rng, ::Type{<: MutableFixedSizePaddedArray{S,T}}, β::T) where {S,T}
    randexp!(rng, MutableFixedSizePaddedArray{S,T}(undef), β)
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
    if expr.args[2] ∈ randtypes
        T = expr.args[2]
        n += 1
    else
        T = Float64
    end
    return :( $(expr.args[1])( $(R){Tuple{$(esc.(expr.args[n:end])...)}, $T}, $([:(PaddedMatrices.compatible($T,$arg)) for arg in args]...)  )  )
end

# macro Mutable(expr)
#     rand_expr(expr, :MutableFixedSizePaddedArray)
# end
# macro Constant(expr)
#     rand_expr(expr, :ConstantFixedSizePaddedArray)
# end

macro Mutable(expr, args...)
    rand_expr(expr, :MutableFixedSizePaddedArray, args...)
end
macro Constant(expr, args...)
    rand_expr(expr, :ConstantFixedSizePaddedArray, args...)
end


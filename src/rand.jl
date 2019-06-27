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


@generated function Random.rand!(rng::VectorizedRNG.PCG, A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}) where {S,T<:Union{Float32,Float64},N,R,L}
    size_T = sizeof(T)
    W = VectorizationBase.pick_vector_width(L, T)
    nrep, r = divrem(L, 4W)
    float_q = :(rand(rng, Vec{$(4W),$T}, VectorizedRNG.RXS_M_XS))
    store_expr = quote end
    for n ∈ 0:3
        push!(store_expr.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(:u, W, n*W)), i + $(n*W))))
    end
    if nrep > 0
        q = quote
            ptr_A = pointer(A)
            for i ∈ 1:$(4W):$(nrep * 4W)
                u = $float_q
                $store_expr
            end
        end
    else
        q = quote ptr_A = pointer(A) end
    end
    if r > 0
        rrep, rrem = divrem(r, W)
        u_sym = gensym(Symbol(:u_remw))
        rrep > 0 && push!(q.args, :($u_sym = rand(rng, Vec{$(W*rrep),$T}) ))
        for n ∈ 0:rrep - 1
            push!(q.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(u_sym, W, n*W)), $(L - r + 1 + n*W))))
        end
        if rrem > 0
            u_sym = gensym(:u_rem)
            push!(u_exprs.args, :($u_sym = rand(rng, Vec{$rrem,$T}) ))
            push!(q.args, :(vstore!(ptr_A, $u_sym, $(L - rrem + 1))))
        end
    end
    push!(q.args, :A)
    q
end
Random.rand!(A::AbstractMutableFixedSizePaddedArray) = rand!(VectorizedRNG.GLOBAL_vPCG, A)
@generated function Random.rand(rng::VectorizedRNG.PCG, ::Type{ <: ConstantFixedSizePaddedArray{S,T}}) where {S,T<:Union{Float32,Float64}}
    N = length(S.parameters)

    R, L = calculate_L_from_size(S.parameters, T)

    size_T = sizeof(T)
    W = VectorizationBase.pick_vector_width(L, T)
    nrep, r = divrem(L, 4W)
    float_q = :(rand(rng, Vec{$(4W),$T}, VectorizedRNG.RXS_M_XS))
    store_expr = quote end
    for n ∈ 0:3
        push!(store_expr.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(:u, W, n*W)), i + $(n*W))))
    end
    u_exprs = quote end
    out_exprs = Expr(:tuple,)
    for i ∈ 1:nrep
        u_sym = Symbol(:u_, i)
        push!(u_exprs.args, :($u_sym = $float_q ))
        for j ∈ 1:4W
            push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
        end
    end
    if r > 0
        rrep, rrem = divrem(r, W)
        u_sym = gensym(Symbol(:u_remw))
        rrep > 0 && push!(u_exprs.args, :($u_sym = rand(rng, Vec{$(W*rrep),$T}) ))
        for j ∈ 1:W*rrep
            push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
        end
        if rrem > 0
            u_sym = gensym(:u_rem)
            push!(u_exprs.args, :($u_sym = rand(rng, Vec{$rrem,$T}) ))
            for j ∈ 1:rrem
                push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
            end
        end
    end
    push!(u_exprs.args, :(ConstantFixedSizePaddedArray{$S,$T,$N,$R,$L}($out_exprs)))
    u_exprs
end
function Random.rand(::Type{ <: ConstantFixedSizePaddedArray{S,T}}) where {S,T<:Union{Float32,Float64}}
    rand(VectorizedRNG.GLOBAL_vPCG, ConstantFixedSizePaddedArray{S,T})
end

@generated function Random.randn!(rng::VectorizedRNG.PCG, A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}) where {S,T<:Union{Float32,Float64},N,R,L}
    size_T = sizeof(T)
    W = VectorizationBase.pick_vector_width(L, T)
    nrep, r = divrem(L, 4W)
    float_q = :(randn(rng, Vec{$(4W),$T}, VectorizedRNG.RXS_M_XS))
    store_expr = quote end
    for n ∈ 0:3
        push!(store_expr.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(:u, W, n*W)), i + $(n*W))))
    end
    if nrep > 0
        q = quote
            ptr_A = pointer(A)
            for i ∈ 1:$(4W):$(nrep * 4W)
                u = $float_q
                $store_expr
            end
        end
    else
        q = quote ptr_A = pointer(A) end
    end
    if r > 0 #&& (r & (W-1)) == 0
        rrep, rrem = divrem(r, W)
        u_sym = gensym(Symbol(:u_remw))
        rrep > 0 && push!(q.args, :($u_sym = randn(rng, Vec{$(W*rrep),$T}) ))
        for n ∈ 0:rrep - 1
            push!(q.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(u_sym, W, n*W)), $(L - r + 1 + n*W))))
        end
        if rrem > 0
            u_sym = gensym(:u_rem)
            push!(q.args, :($u_sym = randn(rng, Vec{$W,$T}) ))
            mask = (VectorizationBase.mask_type(W))(2^rrem - 1)
            push!(q.args, :(vstore!(ptr_A, $u_sym, $(L - rrem + 1), $mask )))
        end
    end
    push!(q.args, :A)
    q
end
Random.randn!(A::AbstractMutableFixedSizePaddedArray) = randn!(VectorizedRNG.GLOBAL_vPCG, A)

@generated function Random.randn(rng::VectorizedRNG.PCG, ::Type{<:ConstantFixedSizePaddedArray{S,T}}) where {S,T<:Union{Float32,Float64}}
    N = length(S.parameters)

    R, L = calculate_L_from_size(S.parameters, T)

    size_T = sizeof(T)
    W = VectorizationBase.pick_vector_width(L, T)
    nrep, r = divrem(L, 4W)
    float_q = :(randn(rng, Vec{$(4W),$T}, VectorizedRNG.RXS_M_XS))
    store_expr = quote end
    for n ∈ 0:3
        push!(store_expr.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(:u, W, n*W)), i + $(n*W))))
    end
    u_exprs = quote end
    out_exprs = Expr(:tuple,)
    for i ∈ 1:nrep
        u_sym = gensym(Symbol(:u_, i))
        push!(u_exprs.args, :($u_sym = $float_q ))
        for j ∈ 1:4W
            push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
        end
    end
    if r > 0
        rrep, rrem = divrem(r, W)
        u_sym = gensym(Symbol(:u_remw))
        rrep > 0 && push!(u_exprs.args, :($u_sym = randn(rng, Vec{$(W*rrep),$T}) ))
        for j ∈ 1:W*rrep
            push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
        end
        if rrem > 0
            u_sym = gensym(:u_rem)
            push!(u_exprs.args, :($u_sym = randn(rng, Vec{$rrem,$T}) ))
            for j ∈ 1:rrem
                push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
            end
        end
    end
    push!(u_exprs.args, :(ConstantFixedSizePaddedArray{$S,$T,$N,$R,$L}($out_exprs)))
    u_exprs
end
function Random.randn(::Type{<:ConstantFixedSizePaddedArray{S,T}}) where {S,T<:Union{Float32,Float64}}
    randn(VectorizedRNG.GLOBAL_vPCG, ConstantFixedSizePaddedArray{S,T})
end
@generated function Random.randn(rng::VectorizedRNG.PCG, ::Static{S}) where {S}
    if isa(S, Integer)
        ST = Tuple{S}
    else
        ST = S
    end
    quote
        $(Expr(:meta,:inline))
        randn(rng, ConstantFixedSizePaddedArray{$ST,Float64})
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
        randn(VectorizedRNG.GLOBAL_vPCG, ConstantFixedSizePaddedArray{$ST,Float64})
    end
end


@generated function Random.randexp!(rng::VectorizedRNG.PCG, A::AbstractMutableFixedSizePaddedArray{S,T,N,R,L}) where {S,T<:Union{Float32,Float64},N,R,L}
    size_T = sizeof(T)
    W = VectorizationBase.pick_vector_width(L, T)
    nrep, r = divrem(L, 4W)
    float_q = :(randexp(rng, Vec{$(4W),$T}, VectorizedRNG.RXS_M_XS))
    store_expr = quote end
    for n ∈ 0:3
        push!(store_expr.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(:u, W, n*W)), i + $(n*W))))
    end
    if nrep > 0
        q = quote
            ptr_A = pointer(A)
            for i ∈ 1:$(4W):$(nrep * 4W)
                u = $float_q
                $store_expr
            end
        end
    else
        q = quote ptr_A = pointer(A) end
    end
    if r > 0
        rrep, rrem = divrem(r, W)
        u_sym = gensym(Symbol(:u_remw))
        rrep > 0 && push!(q.args, :($u_sym = randexp(rng, Vec{$(W*rrep),$T}) ))
        for n ∈ 0:rrep - 1
            push!(q.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(u_sym, W, n*W)), $(L - r + 1 + n*W))))
        end
        if rrem > 0
            u_sym = gensym(:u_rem)
            push!(u_exprs.args, :($u_sym = randexp(rng, Vec{$rrem,$T}) ))
            push!(q.args, :(vstore!(ptr_A, $u_sym, $(L - rrem + 1))))
        end
    end
    push!(q.args, :A)
    q
end
Random.randexp!(A::AbstractMutableFixedSizePaddedArray) = randexp!(VectorizedRNG.GLOBAL_vPCG, A)


@generated function Random.randexp(rng::VectorizedRNG.PCG, ::Type{<:ConstantFixedSizePaddedArray{S,T}}) where {S,T<:Union{Float32,Float64}}
    N = length(S.parameters)

    R, L = calculate_L_from_size(S.parameters, T)

    size_T = sizeof(T)
    W = VectorizationBase.pick_vector_width(L, T)
    nrep, r = divrem(L, 4W)
    float_q = :(randexp(rng, Vec{$(4W),$T}, VectorizedRNG.RXS_M_XS))
    store_expr = quote end
    for n ∈ 0:3
        push!(store_expr.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(:u, W, n*W)), i + $(n*W))))
    end
    u_exprs = quote end
    out_exprs = Expr(:tuple,)
    for i ∈ 1:nrep
        u_sym = gensym(Symbol(:u_, i))
        push!(u_exprs.args, :($u_sym = $float_q ))
        for j ∈ 1:4W
            push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
        end
    end
    if r > 0
        rrep, rrem = divrem(r, W)
        u_sym = gensym(Symbol(:u_remw))
        rrep > 0 && push!(u_exprs.args, :($u_sym = randexp(rng, Vec{$(W*rrep),$T}) ))
        for j ∈ 1:W*rrep
            push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
        end
        if rrem > 0
            u_sym = gensym(:u_rem)
            push!(u_exprs.args, :($u_sym = randexp(rng, Vec{$rrem,$T}) ))
            for j ∈ 1:rrem
                push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
            end
        end
    end
    push!(u_exprs.args, :(ConstantFixedSizePaddedArray{$S,$T,$N,$R,$L}($out_exprs)))
    u_exprs
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


function rand_expr(expr, R)
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
    return :( $(expr.args[1])( $(R){Tuple{$(esc.(expr.args[n:end])...)}, $T}  )  )
end

macro Mutable(expr)
    rand_expr(expr, :MutableFixedSizePaddedArray)
end
macro Constant(expr)
    rand_expr(expr, :ConstantFixedSizePaddedArray)
end


# Not sure where these should live.
# They are not vectorized, which is why they aren't in
# VectorizedRNG - that'd be misleading.
function randgamma_g1(rng::AbstractRNG, α::T) where {T}
    # @show α
    OneThird = one(T)/T(3)
    d = α - OneThird
    @fastmath c = OneThird / sqrt(d)
    @fastmath while true
        x = randn(rng, T)
        v = one(T) + c*x
        v < zero(T) && continue
        v3 = v^3
        dv3 = d*v3
        randexp(rng, T) > T(-0.5)*x^2 - d + dv3 - d*log(v3) && return dv3
    end
end
function randgamma(rng::AbstractRNG, α::T) where {T}
    α < one(T) ? exp(-randexp(rng, T)/α) * randgamma_g1(rng, α+one(T)) : randgamma_g1(rng, α)
end
randgamma(rng::AbstractRNG, α::T, β::T) where {T} = β * randgamma(rng, α)
randgamma(α::Real, β::Real) = randgamma(Random.GLOBAL_RNG, α, β)
randgamma(α::Real) = randgamma(Random.GLOBAL_RNG, α)

randbeta(rng::AbstractRNG, α::Real, β::Real) = (x = randgamma(rng, α); x / (x + randgamma(rng, β)))
randbeta(α::Real, β::Real) = randbeta(Random.GLOBAL_RNG, α, β)

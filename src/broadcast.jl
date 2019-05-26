# First, we add templates for matrix multiplication
abstract type AbstractProdct{T,N} end
abstract type AbstractSizedProdct{M,P,K,T,N} <: AbstractProdct{T,N} end


function reduce_size(S::DataType, ::Val{N}) where {N}
    s = fill!(MutableFixedSizePaddedVector{N,Int}(undef),1)
    SV = S.parameters
    j = SV[1]::Union{Int,DataType}
    if j isa Int
        for i ∈ eachindex(SV)
            s[i] = (SV[i])::Int
        end
    elseif j isa DataType
        for i ∈ eachindex(SV)
            s2 = reduce_size(SV[i]::DataType, Val(N))
            for n ∈ 1:N
                sₙ = s[n]
                s2ₙ = s2[n]
                if sₙ == 1
                    s[n] = s2[n]
                elseif (s2ₙ == sₙ) || s2ₙ == 1 # neither are 1
                    nothing
                else
                    throw("sₙ == $sₙ != s2ₙ == $s2ₙ")
                end
            end
        end
    end
    ConstantFixedSizePaddedVector(s)
end
function reduce_size(S::DataType)
    reduce_size!(Int[1], S)
end
function pick_size(s1::Int, s2::Int)
    if s1 == 1
        s3 = s2
    elseif (s1 == s2) || (s2 == 1)
        s3 = s1
    else
        throw("s1 == $s1 != s2 == $s2; they should be equal, or == 1")
    end
    s3
end

function reduce_size!(s::Vector{Int}, S::DataType)
    SV = S.parameters
    length(SV) == 0 && return s
    j = SV[1]::Union{Int,DataType}
    if j isa Int
        N1 = length(SV)
        N2 = length(s)
        for n ∈ 1:min(N1,N2)
            s[n] = pick_size(s[n], SV[n])
        end
        for n ∈ min(N1,N2)+1:N1
            push!(s, SV[n])
        end
    elseif j isa DataType
        for i ∈ eachindex(SV)
            s2 = reduce_size!(s, (SV[i])::DataType)
        end
    end
    s
end

#reduce_size(S,Val(2))


@inline Base.ndims(::Type{<:AbstractProdct{T,N}}) where {T,N} = N
@inline Base.broadcastable(A::AbstractProdct) = A
function Base.show(io::IO, p::AbstractProdct)
    # We don't define a getindex method.
    println("Product of A, of size $(size(p.A)):\n", p.A, "\nand B, of size $(size(p.B)):\n", p.B)
end

struct SizedMatrixMatrixProduct{T,M,P,K,TA <: AbstractFixedSizePaddedMatrix{M,P,T}, TB <: AbstractFixedSizePaddedMatrix{P,K,T}} <: AbstractSizedProdct{M,P,K,T,2}
    A::TA
    B::TB
end
@inline function LinearAlgebra.:×(A::TA, B::TB) where {M,P,K,T,TA <: AbstractFixedSizePaddedMatrix{M,P,T}, TB <: AbstractFixedSizePaddedMatrix{P,K,T}}
    SizedMatrixMatrixProduct{T,M,P,K,TA,TB}(A,B)
end

@inline Base.size(::SizedMatrixMatrixProduct{T,M,P,K}) where {T,M,P,K} = (M,K)
@inline Base.axes(::SizedMatrixMatrixProduct{T,M,P,K}) where {T,M,P,K} = (Base.OneTo(M),Base.OneTo(K))

struct SizedVectorMatrixProduct{T,P,K,TA <: AbstractFixedSizePaddedVector{P,T}, TB <: AbstractFixedSizePaddedMatrix{P,K,T}} <: AbstractSizedProdct{1,P,K,T,2}
    A::TA
    B::TB
end
function LinearAlgebra.:×(A::TA, B::TB) where {P,K,T,TA <: AbstractFixedSizePaddedVector{P,T}, TB <: AbstractFixedSizePaddedMatrix{P,K,T}}
    SizedVectorMatrixProduct{T,P,K,TA,TB}(A,B)
end
@inline Base.size(::SizedVectorMatrixProduct{T,P,K}) where {T,P,K} = (1,K)
@inline Base.axes(::SizedVectorMatrixProduct{T,P,K}) where {T,P,K} = (Base.OneTo(1),Base.OneTo(K))


struct SizedMatrixVectorProduct{T,M,P,TA <: AbstractFixedSizePaddedMatrix{M,P,T}, TB <: AbstractFixedSizePaddedVector{P,T}} <: AbstractSizedProdct{M,P,1,T,1}
    A::TA
    B::TB
end
function LinearAlgebra.:×(A::TA, B::TB) where {M,P,T,TA <: AbstractFixedSizePaddedMatrix{M,P,T}, TB <: AbstractFixedSizePaddedVector{P,T}}
    SizedMatrixVectorProduct{T,M,P,TA,TB}(A,B)
end
@inline Base.size(::SizedMatrixVectorProduct{T,M}) where {T,M} = (M,)
@inline Base.axes(::SizedMatrixVectorProduct{T,M}) where {T,M} = (Base.OneTo(M),Base.OneTo(1))


struct MatrixMatrixProduct{T,TA <: AbstractPaddedMatrix{T}, TB <: AbstractPaddedMatrix{T}} <: AbstractProdct{T,2}
    A::TA
    B::TB
end
function LinearAlgebra.:×(A::TA, B::TB) where {T,TA <: AbstractPaddedMatrix{T}, TB <: AbstractPaddedMatrix{T}}
    MatrixMatrixProduct{T,TA,TB}(A,B)
end
@inline Base.size(A::MatrixMatrixProduct) = (size(A.A,1),size(A.B,2))
@inline Base.axes(A::MatrixMatrixProduct) = (Base.OneTo(size(A.A,1)),Base.OneTo(size(A.B,2)))

struct VectorMatrixProduct{T,TA <: AbstractPaddedVector{T}, TB <: AbstractPaddedMatrix{T}} <: AbstractProdct{T,2}
    A::TA
    B::TB
end
function LinearAlgebra.:×(A::TA, B::TB) where {T,TA <: AbstractPaddedVector{T}, TB <: AbstractPaddedMatrix{T}}
    VectorMatrixProduct{T,TA,TB}(A,B)
end
@inline Base.size(A::VectorMatrixProduct) = (1,size(A.B,2))
@inline Base.axes(A::VectorMatrixProduct) = (Base.OneTo(1),Base.OneTo(size(A.B,2)))
nn
struct MatrixVectorProduct{T,TA <: AbstractPaddedMatrix{T}, TB <: AbstractPaddedVector{T}} <: AbstractProdct{T,1}
    A::TA
    B::TB
end
function LinearAlgebra.:×(A::TA, B::TB) where {T,TA <: AbstractPaddedMatrix{T}, TB <: AbstractPaddedVector{T}}
    MatrixVectorProduct{T,TA,TB}(A,B)
end
@inline Base.size(A::MatrixVectorProduct) = (size(A.A,1),)
@inline Base.axes(A::MatrixVectorProduct) = (Base.OneTo(size(A.A,1)),)

@generated function load_kernel(
                MMP::Union{<:SizedMatrixMatrixProduct{T},<:MatrixMatrixProduct{T}},
                i, j, ::Val{MP}, ::Val{mp}
            ) where {T,MP,mp}
    M, P = MP
    mk, pk = mp
    out = MutableFixedSizePaddedMatrix{mk,pk,T}(undef)
    # Should write new, simpler kernel function
    # that can take N as a symbol, and accepts arbitrary strides.
    quote

    end
end

@inline function Base.convert(::Type{A}, MMP::AbstractProdct) where {A <: AbstractArray}
    MMP.A * MMP.B
end
@inline function Base.copyto!(
                C::Union{<: PaddedArray{T},<:AbstractMutableFixedSizePaddedArray{T}},
                MMP::AbstractProdct{T}
            ) where {T}
    mul!(C, MMP.A, MMP.B)
end
@inline function Base.:+(C::AbstractPaddedArray, MMP::AbstractProdct)
    gemm(C, MMP.A, MMP.B)
end
@inline function Base.:+(MMP::AbstractProdct, C::AbstractPaddedArray)
    gemm(C, MMP.A, MMP.B)
end

struct KernelView{BC <: Base.Broadcast.Broadcasted}
    bc::BC
    m::Int
    k::Int
end
@inline function KernelView(bc::BC, m, k, ::Any, ::Any) where {BC <: Base.Broadcast.Broadcasted}
    KernelView{BC}(bc, m, k)
end
@inline function KernelView(bc::SizedMatrixMatrixProduct{T,M,P,K}, m, k, ::Val{Mk}, ::Val{Kk}) where {T,M,P,K,Mk,Kk}
    
end
@inline function Base.getindex(kv::KernelView, i, j)
    @inbounds kv.bc[kv.m + i, kv.k + j]
end

# @inline function Base.copyto!(
#             C::AbstractPaddedArray,
#             AB::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{2},Nothing,typeof(+),Tuple{Array{Float64,2},Foo{Array{Float64,2},Array{Float64,2}}}}
#         )

# end

# @enum ContainerType begin # First has highest priority
#     PaddedUnsized
#     MutableFixedSize
#     ConstantFixedSize
# end
@enum AccessPattern begin # Last has highest priority
    LinearIndexing
    BatchedColumnMajor # Batched pattern down rows, to reduce reloads of (column) vectors broadcasted with matrices
    CartesianIndexing # Standard Cartesian Indexing; if there is a row vector, we save on it's reloads.
    KernelBatches # Same pattern as matrix mul functions
end

abstract type AbstractPaddedMatrixStyle{S, A <: AccessPattern} <: Broadcast.BroadcastStyle end

struct FixedSizePaddedMatrixDefaultStyle{S,A} <: AbstractPaddedMatrixStyle{S,A} end

Base.BroadcastStyle(::Type{<:AbstractFixedSizePaddedArray{S}}) where {S} = FixedSizePaddedMatrixDefaultStyle{S,Agnostic}()

function Base.BroadcastStyle(style1::FixedSizePaddedMatrixDefaultStyle{S1,A1},
                             style2::FixedSizePaddedMatrixDefaultStyle{S2,A2}) where {S1,S2,A1,A2}


end


# @generated function Base.BroadcastStyle(style1::AbstractPaddedMatrixStyle{S1,A1,C1}, style2::AbstractPaddedMatrixStyle{S2,A2,C2}) where {S1,A1,C1,S2,A2,C2}
#     A3 = max(A1, A2)
#     C3 = min(C1, C2)
#     if C3 == PaddedUnsized
#         S1N = C1 == PaddedUnsized ? S1 : length(S1.parameters)
#         S2N = C2 == PaddedUnsized ? S2 : length(S2.parameters)
#         S3 = max(S1N, S2N)
#     else
#         S1V = S1.parameters
#         S2V = S2.parameters
#         S1N = length(S1V)
#         S2N = length(S2V)
#         s3 = Int[]
#         for i ∈ 1:min(S1N,S2N)
#             if S1V[i] == S2V[i] || S2V[i] == 1
#                 push!(s3, S1V[i])
#             elseif S1V[i] == 1
#                 push!(s3, S2V[i])
#             else
#                 throw("Broadcast dimensions aren't alligned, $(S1V) vs $(S2V).")
#             end
#         end
#         sv_longer = S1N > S2N ? S1V : S2V
#         for i ∈ min(S1N,S2N)+1:max(S1N,S2N)
#             push!(s3, sv_longer[i])
#         end
#         S3 = Tuple{s3...}
#     end
#     :(PaddedMatrixStyle{$S3,$A3,$C3}())


# end
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{0}, style::AbstractPaddedMatrixStyle) = style


@generated function Base.similar(bc::Base.Broadcast.Broadcasted{PaddedMatrixStyle{S,A,C}}, ::Type{T}) where {S,A,C,T}
    Salloc = reduce_size(S)
    N, R, L = calc_NPL(Salloc, T)
    ST = Tuple{Salloc...}
    :(MutableFixedSizePaddedArray{$ST,$T,$N,$R,$L}(undef))
end
@generated function Base.similar(bc::Base.Broadcast.Broadcasted{PaddedMatrixStyle{S,A,C}}, mystack::Ptr{T}) where {S,A,C,T}
    Salloc = reduce_size(S)
    N, R, L = calc_NPL(Salloc, T)
    ST = Tuple{Salloc...}
    # allocates from mystack, and advances that pointer.
    :(PtrArray{$ST,$T,$N,$R,$L,true}(mystack), mystack+$(L*sizeof(T)))
end
@inline function Base.Broadcast.materialize(bc::Base.Broadcast.Broadcasted{PaddedMatrixStyle{S,A,C}}, mystack::Ptr{T}) where {S,A,C,T}
    out, mystack = similar(bc, mystack)
    Base.broadcast.materialize!(out, bc), mystack
end
@inline function Base.Broadcast.materialize(bc::Base.Broadcast.Broadcasted{PaddedMatrixStyle{S,A,C}}) where {S,A,C}
    Base.broadcast.materialize!(similar(bc), bc)
end

@inline extract(x::Base.RefValue) = x[]
@inline extract(x::Number) = x

## flattens a broadcast
function broadcast_index_expression(SB, N)
    inds = [gensym(Symbol(:n_,n)) for n ∈ 1:N]
    q = quote end
    preq = quote end
    assign_to = gensym(:assign)
    broadcast_index_expression!(q, preq, (SB.parameters)::DataType, inds, :bc, assign_to)
    push!(q.args, Expr(:call, :setindex!, :out, assign_to, inds...))
    inds, q, preq 
end
function broadcast_index_expression!(q, preq, SBV, inds, bcsym, assign)
    callexpr = Expr(:call, Expr(:., bcsym, :(:f)))
    for l ∈ eachindex(SBV)
        SBVₗ = ((SBV[l])::DataType).parameters
        argsym = gensym(:arg)
        if length(SBVₗ) == 0 # scalar argument
            push!(preq.args, :($argsym = @inbounds PaddedMatrices.extract($bcsym.args[1])))
            push!(callexpr.args, argsym)
            continue
        else
            push!(preq.args, :($argsym = @inbounds $bcsym.args[1]))
        end
        SBVₗ₁ = (SBVₗ[1])::Union{Int,DataType}
        if SBVₗ₁ isa Int # array argument
            ind = Expr(:call, :getindex, argsym)
            for n ∈ 1:length(SBVₗ)
                indₙ = (SBVₗ[n])::Int
                if indₙ == 1
                    push!(ind.args, 1)
                else
                    push!(ind.args, inds[n])
                end
            end
            push!(callexpr.args, ind)
        elseif SBVₗ₁ isa DataType
            assign_to = gensym(:assign)
            broadcast_index_expression!(q, preq, SBVₗ, inds, argsym, assign_to)
            push!(callexpr.args, assign_to)
        end
    end
    push!(q.args, Expr(:(=), assign, callexpr))
    nothing
end
function broadcast_linearindex_expression(SB, N)
    ind = gensym(:ind)
    q = quote end
    preq = quote end
    assign_to = gensym(:assign)
    broadcast_linearindex_expression!(q, preq, (SB.parameters)::DataType, ind, :bc, assign_to)
    push!(q.args, Expr(:call, :setindex!, :out, assign_to, ind))
    ind, q, preq
end
function broadcast_linearindex_expression!(q, preq, SBV, ind, bcsym, assign)
    callexpr = Expr(:call, Expr(:., bcsym, :(:f)))
    for l ∈ eachindex(SBV)
        SBVₗ = ((SBV[l])::DataType).parameters
        argsym = gensym(:arg)
        if length(SBVₗ) == 0 # scalar argument
            push!(preq.args, :($argsym = @inbounds PaddedMatrices.extract($bcsym.args[1])))
            push!(callexpr.args, argsym)
            continue
        else
            push!(preq.args, :($argsym = @inbounds $bcsym.args[1]))
        end
        SBVₗ₁ = (SBVₗ[1])::Union{Int,DataType}
        if SBVₗ₁ isa Int # array argument
            push!(callexpr.args, Expr(:call, :getindex, argsym, ind))
        elseif SBVₗ₁ isa DataType
            assign_to = gensym(:assign)
            broadcast_linearindex_expression!(q, preq, SBVₗ, ind, argsym, assign_to)
            push!(callexpr.args, assign_to)
        end                           
    end
    push!(q.args, Expr(:(=), assign, callexpr))
    nothing
end

function materialize_quote(S, A, C, T, SB, N, P)
    if A == LinearIndexing
        assigned_to, ind, loop_body, pre_loop = broadcast_index_expression(SB, N)
        return quote
            $pre_loop
            LoopVectorization.@vectorize $T for $ind ∈ 1:$(prod(S))
                $loop_body
            end
        end
    elseif A == BatchedColumnMajor
        @assert N == 2
        # Hvaven't actually implemented this as batched yet.
        # For now, will default to regular CartesianIndexing fallback.
        # A proper implementation would be able to estimate how many registers are left over,
        # and make sure those that shouldn't be reloaded are in fact not.
        # Ie, which arrays are column vectors vs matrices,
        # and are special functions like exp getting called, which will consume a lot of registers?
        assigned_to, inds, loop_body, pre_loop = broadcast_index_expression(SB, N)
#        M = S[1]
#        L = S[2]
#        Md, Mr = divrem(M, 
        loop = quote
            LoopVectorization.@vectorize $T for $(inds[1]) ∈ 1:$(S[1])
                $loop_body
            end
        end
        for n ∈ 2:N
            loop = quote
                for $(inds[n]) ∈ 1:$(S[n])
                    $loop
                end
            end
        end
        return quote
            $pre_loop
            $loop
        end
    elseif A == CartesianIndexing
        assigned_to, inds, loop_body, pre_loop = broadcast_index_expression(SB, N)
        loop = quote
            LoopVectorization.@vectorize $T for $(inds[1]) ∈ 1:$(S[1])
                $loop_body
            end
        end
        for n ∈ 2:N
            loop = quote
                for $(inds[n]) ∈ 1:$(S[n])
                    $loop
                end
            end
        end
        return quote
            $pre_loop
            $loop
        end
    elseif A == KernelBatches
        
    end
    

end


@generated function Base.Broadcast.materialize!(out::AbstractMutableFixedSizePaddedMatrix{S,T,N,P}, bc::Base.Broadcast.Broadcasted{PaddedMatrixStyle{SB,A,C}}) where {S,A,C,T,SB,N,P}

    materialize_quote(S, A, C, T, SB, N, P)
    
    

end



function batched_column_major_quote(S, A, C, T)

end

function kernel_batch_quote(S, A, C, T)
    N = length(S)
    # for now
    @assert N == 2
    M, P = S
    epr, m_k, p_k = pick_kernel_size(T, M, P)
    mrep, mrem = divrem(M, m_k)
    prep, prem = divrem(P, p_k)
    quote
        for p ∈ 0:prep-1
            for m ∈ 0:mrep-1

            end
        end
    end
end

@generated function Base.copyto!(
                        dest::AbstractMutableFixedSizePaddedArray{S1,T},
                        bc::Base.Broadcast.Broadcasted{PaddedMatrixStyle{S2,A,C}}
                    ) where {S1,S2,A,C,T}
    SV = S1.parameters
    if A == Agnostic || A == StandardColumnMajor
        return quote
            @inbounds @simd for I ∈ CartesianIndices($(Expr(:tuple, SV...)))
                dest[I] = bc[I]
            end
        end
    elseif A == BatchedColumnMajor
        return batched_column_major_quote(SV, A, C, T)
    elseif A == KernelBatches
        return kernel_bactch_quote(SV, A, C, T)
    end
end

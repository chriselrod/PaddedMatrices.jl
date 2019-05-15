# First, we add templates for matrix multiplication
abstract type AbstractProdct{T,N} end
abstract type AbstractSizedProdct{M,P,K,T,N} <: AbstractProdct{T,N} end

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

@enum ContainerType begin # First has highest priority
    PaddedUnsized
    MutableFixedSize
    ConstantFixedSize
end
@enum AccessPattern begin # Last has highest priority
    Agnostic  # When we don't have a preference: Standard Cartesian Indexing
    BatchedColumnMajor # Batched pattern down rows, to reduce reloads of (column) vectors broadcasted with matrices
    StandardColumnMajor # Standard Cartesian Indexing; if there is a row vector, we save on it's reloads.
    KernelBatches # Same pattern as matrix mul functions
end

abstract type PaddedMatrixStyle{S,A <: AccessPattern,C <: ContainerType} <: Broadcast.BroadcastStyle end

@generated function Base.BroadcastStyle(style1::PaddedMatrixStyle{S1,A1,C1}, style2::PaddedMatrixStyle{S2,A2,C2}) where {S1,A1,C1,S2,A2,C2}
    A3 = max(A1, A2)
    C3 = min(C1, C2)
    if C3 == PaddedUnsized
        S1N = C1 == PaddedUnsized ? S1 : length(S1.parameters)
        S2N = C2 == PaddedUnsized ? S2 : length(S2.parameters)
        S3 = max(S1N, S2N)
    else
        S1V = S1.parameters
        S2V = S2.parameters
        S1N = length(S1V)
        S2N = length(S2V)
        s3 = Int[]
        for i ∈ 1:min(S1N,S2N)
            if S1V[i] == S2V[i] || S2V[i] == 1
                push!(s3, S1V[i])
            elseif S1V[i] == 1
                push!(s3, S2V[i])
            else
                throw("Broadcast dimensions aren't alligned, $(S1V) vs $(S2V).")
            end
        end
        sv_longer = S1N > S2N ? S1V : S2V
        for i ∈ min(S1N,S2N)+1:max(S1N,S2N)
            push!(s3, sv_longer[i])
        end
        S3 = Tuple{s3...}
    end
    :(PaddedMatrixStyle{$S3,$A3,$C3}())
end
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{0}, style::PaddedMatrixStyle) = style


@generated function Base.similar(bc::Base.Broadcast.Broadcasted{PaddedMatrixStyle{S,A,C}}, ::Type{T}) where {S,A,C,T}
    if S isa Int
        return :(PaddedArray{$T}(undef, size(bc)))
    end
    N, R, L = calc_NPL(S, T)
    :(MutableFixedSizePaddedArray{$S,$T,$N,$R,$L}(undef))
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

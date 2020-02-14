

function LinearAlgebra.mul!(
    c::AbstractVector{T}, A::AbstractStrideMatrix{M,N,T}, b::AbstractVector{T}
) where {M,N,T}
    @assert size(c,1) == size(A,1)
    @assert size(b,1) == size(A,2)
    @avx for m ∈ 1:size(A,1)
        cₘ = zero(T)
        for n ∈ 1:size(A,2)
            cₘ += A[m,n] * b[n]
        end
        c[m] = cₘ
    end
    C
end
function nmul!(
    c::AbstractVector{T}, A::AbstractStrideMatrix{M,N,T}, b::AbstractVector{T}
) where {M,N,T}
    @assert size(c,1) == size(A,1)
    @assert size(b,1) == size(A,2)
    @avx for m ∈ 1:size(A,1)
        cₘ = zero(T)
        for n ∈ 1:size(A,2)
            cₘ -= A[m,n] * b[n]
        end
        c[m] = cₘ
    end
    C
end

function matmul_sizes(C, A, B)
    MC, NC = maybestaticsize(C, Val{1:2}())
    MA, KA = maybestaticsize(A, Val{1:2}())
    KB, NB = maybestaticsize(B, Val{1:2}())
    M = VectorizationBase.static_promote(MC, MA)
    K = VectorizationBase.static_promote(KA, KB)
    N = VectorizationBase.static_promote(NC, NB)
    M, K, N
end

function LinearAlgebra.mul!(
    C::AbstractStrideMatrix{<:Any,<:Any,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {T}
    M, K, N = matmul_sizes(C, A, B)
    @avx for m ∈ 1:M
        for n ∈ 1:N
            Cₘₙ = zero(T)
            for k ∈ 1:K
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    C
end
function nmul!(
    C::AbstractStrideMatrix{<:Any,<:Any,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {T}
    M, K, N = matmul_sizes(C, A, B)
    @avx for m ∈ 1:M
        for n ∈ 1:N
            Cₘₙ = zero(T)
            for k ∈ 1:K
                Cₘₙ -= A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    C
end

@inline function Base.:*(
    sp::StackPointer,
    A::AbstractStrideMatrix{M,<:Any,T},
    B::AbstractStrideMatrix{<:Any,N,T}
) where {M,K,N,T}
    sp, D = PtrMatrix{M,N,T}(sp)
    sp, mul!(D, A, B)
end
function Base.:*(
    A::AbstractStrideMatrix{M,K,T},
    B::AbstractStrideMatrix{K,N,T}
) where {M,K,N,T}
    mul!(FixedSizeMatrix{M,N,T}(undef), A, B)
end

function muladd!(
    C::AbstractStrideMatrix{<:Any,<:Any,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {T}
    M, K, N = matmul_sizes(C, A, B)
    @avx for m ∈ 1:M
        for n ∈ 1:N
            Cₘₙ = zero(T)
            for k ∈ 1:K
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] += Cₘₙ
        end
    end
    C
end
function mulsub!(
    C::AbstractStrideMatrix{<:Any,<:Any,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {T}
    M, K, N = matmul_sizes(C, A, B)
    @avx for m ∈ 1:M
        for n ∈ 1:N
            Cₘₙ = zero(T)
            for k ∈ 1:K
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ - C[m,n]
        end
    end
    C
end
function nmuladd!(
    C::AbstractStrideMatrix{<:Any,<:Any,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {T}
    M, K, N = matmul_sizes(C, A, B)
    @avx for m ∈ 1:M
        for n ∈ 1:N
            Cₘₙ = zero(T)
            for k ∈ 1:K
                Cₘₙ -= A[m,k] * B[k,n]
            end
            C[m,n] += Cₘₙ
        end
    end
    C
end
function nmulsub!(
    C::AbstractStrideMatrix{<:Any,<:Any,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {T}
    M, K, N = matmul_sizes(C, A, B)
    @avx for m ∈ 1:M
        for n ∈ 1:N
            Cₘₙ = zero(T)
            for k ∈ 1:K
                Cₘₙ -= A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ - C[m,n]
        end
    end
    C
end

@inline extract_λ(a) = a
@inline extract_λ(a::UniformScaling) = a.λ
@inline function Base.:*(A::AbstractFixedSizeArray{S,T,N,X,L}, bλ::Union{T,UniformScaling{T}}) where {S,T<:Real,N,X,L}
    mv = FixedSizeArray{S,T,N,X,L}(undef)
    b = extract_λ(bλ)
    @avx for i ∈ eachindex(A)
        mv[i] = A[i] * b
    end
    mv
end
@inline function Base.:*(Aadj::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,X,L}}, bλ::Union{T,UniformScaling{T}}) where {S,T<:Real,N,X,L}
    mv = FixedSizeArray{S,T,N,X,L}(undef)
    A = Aadj.parent
    b = extract_λ(bλ)
    @avx for i ∈ eachindex(A)
        mv[i] = A[i] * b
    end
    mv'
end
function Base.:*(
    sp::StackPointer,
    A::AbstractFixedSizeArray{S,T,N,X,L},
    bλ::Union{T,UniformScaling{T}}
) where {S,T<:Real,N,X,L}
    mv = PtrArray{S,T,N,X,L}(pointer(sp,T))
    b = extract_λ(bλ)
    @avx for i ∈ eachindex(A)
        mv[i] = A[i] * b
    end
    sp + VectorizationBase.align(sizeof(T)*L), mv
end
function Base.:*(
    sp::StackPointer,
    Aadj::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,X,L}},
    bλ::Union{T,UniformScaling{T}}
) where {S,T<:Real,N,X,L}
    mv = PtrArray{S,T,N,X,L}(pointer(sp,T))
    A = Aadj.parent
    b = extract_λ(bλ)
    @avx for i ∈ eachindex(A)
        mv[i] = A[i] * b
    end
    sp + VectorizationBase.align(sizeof(T)*L), mv'
end
@inline function Base.:*(a::T, B::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
    mv = FixedSizeArray{S,T,N,X,L}(undef)
    @avx for i ∈ eachindex(B)
        mv[i] = a * B[i]
    end
        # mv
    ConstantFixedSizeArray(mv)
end

@inline function nmul!(
    D::AbstractMatrix{T},
    A′::LinearAlgebra.Adjoint{T,<:AbstractMatrix{T}},
    X::AbstractMatrix{T}
) where {T <: BLAS.BlasFloat}
    BLAS.gemm!('T','N',-one(T),A′.parent,X,zero(T),D)
end


function LinearAlgebra.mul!(
    C::AbstractStrideMatrix{<:Any,<:Any,T},
    A::LinearAlgebra.Diagonal{T,<:AbstractStrideVector{<:Any,T}},
    B::AbstractMatrix{T}
) where {T}
    M, K, N = matmul_sizes(C, A, B)
    MandK = VectorizationBase.static_promote(M, K)
    vA = parent(A)
    @avx for n ∈ 1:N, m ∈ 1:MandK
        C[m,n] = vA[m] * B[m,n]
    end
    C
end
function Base.:*(
    A::LinearAlgebra.Diagonal{T,<:AbstractStrideVector{M,T}},
    B::AbstractStrideMatrix{M,N,T}
) where {M,N,T}
    mul!(similar(B), A, B)
end
function Base.:*(
    sp::StackPointer,
    A::LinearAlgebra.Diagonal{T,<:AbstractStrideVector{M,T}},
    B::AbstractStrideMatrix{M,N,T}
) where {M,N,T}
    sp, C = similar(sp, B)
    sp, mul!(C, A, B)
end


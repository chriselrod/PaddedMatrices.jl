

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

check_matmul_sizes(::AbstractStrideMatrix{M,N}, ::AbstractStrideMatrix{M,K}, ::AbstractStrideMatrix{K,N}) where {M,K,N} = nothing
check_matmul_sizes(C::AbstractStrideMatrix{-1,N}, A::AbstractStrideMatrix{-1,K}, B::AbstractStrideMatrix{K,N}) where {K,N} = check_matmul_sizes_dynamic(C, A, B)
check_matmul_sizes(C::AbstractStrideMatrix{M,-1}, A::AbstractStrideMatrix{M,K}, B::AbstractStrideMatrix{K,-1}) where {M,K} = check_matmul_sizes_dynamic(C, A, B)
check_matmul_sizes(C::AbstractStrideMatrix{M,N}, A::AbstractStrideMatrix{M,-1}, B::AbstractStrideMatrix{-1,N}) where {M,N} = check_matmul_sizes_dynamic(C, A, B)
check_matmul_sizes(C::AbstractStrideMatrix{-1,-1}, A::AbstractStrideMatrix{-1,K}, B::AbstractStrideMatrix{K,-1}) where {K} = check_matmul_sizes_dynamic(C, A, B)
check_matmul_sizes(C::AbstractStrideMatrix{-1,N}, A::AbstractStrideMatrix{-1,-1}, B::AbstractStrideMatrix{-1,N}) where {N} = check_matmul_sizes_dynamic(C, A, B)
check_matmul_sizes(C::AbstractStrideMatrix{M,-1}, A::AbstractStrideMatrix{M,-1}, B::AbstractStrideMatrix{-1,-1}) where {M} = check_matmul_sizes_dynamic(C, A, B)
check_matmul_sizes(C::AbstractStrideMatrix{-1,-1}, A::AbstractStrideMatrix{-1,-1}, B::AbstractStrideMatrix{-1,-1}) = check_matmul_sizes_dynamic(C, A, B)

check_matmul_sizes(::AbstractStrideMatrix{M,N}, ::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeArray{Tuple{M}}}, ::AbstractStrideMatrix{M,N}) where {T,M,N} = nothing
function check_matmul_sizes(C::AbstractStrideMatrix{M,-1}, A::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeArray{Tuple{M}}}, B::AbstractStrideMatrix{M,-1}) where {T,M}
    check_matmul_sizes_dynamic(C, A, B)
end
function check_matmul_sizes(C::AbstractStrideMatrix{-1,N}, A::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeArray{Tuple{-1}}}, B::AbstractStrideMatrix{-1,N}) where {T,N}
    check_matmul_sizes_dynamic(C, A, B)
end
function check_matmul_sizes(C::AbstractStrideMatrix{-1,-1}, A::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeArray{Tuple{-1}}}, B::AbstractStrideMatrix{-1,-1}) where {T}
    check_matmul_sizes_dynamic(C, A, B)
end
check_matmul_sizes(C, A, B) = check_matmul_sizes_dynamic(C, A, B)
# check_matmul_sizes(C::AbstractStrideMatrix, A::AbstractStrideMatrix, B::AbstractStrideMatrix) = throw("Sizes are incompatible!; C: $(size(C)); A: $(size(A)); B: $(size(B))")

function check_matmul_sizes_dynamic(C, A, B)
    Mc, Nc = size(C)
    Ma, Ka = size(A)
    Kb, Nb = size(B)
    Mc == Ma || throw("C and A must have the same number of rows, but found $Mc and $Ma.")
    Nc == Nb || throw("C and B must have the same number of columns, but found $Nc and $Nb.")
    Ka == Kb || throw("A must have the same number of columns as B has rows, but found $Ka and $Kb.")
    nothing
end

function LinearAlgebra.mul!(
    C::AbstractStrideMatrix{M,N,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {M,N,T}
    check_matmul_sizes(C, A, B)
    @avx for m ∈ 1:size(C,1)
        for n ∈ 1:size(C,2)
            Cₘₙ = zero(T)
            for k ∈ 1:size(B,1)
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    C
end
function nmul!(
    C::AbstractStrideMatrix{M,N,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {M,N,T}
    check_matmul_sizes(C, A, B)
    @avx for m ∈ 1:size(C,1)
        for n ∈ 1:size(C,2)
            Cₘₙ = zero(T)
            for k ∈ 1:size(B,1)
                Cₘₙ -= A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    C
end

@inline function Base.:*(
    sp::StackPointer,
    A::AbstractStrideMatrix{M,K,T},
    B::AbstractStrideMatrix{K,N,T}
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
    C::AbstractStrideMatrix{M,N,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {M,N,T}
    check_matmul_sizes(C, A, B)
    @avx for m ∈ 1:size(C,1)
        for n ∈ 1:size(C,2)
            Cₘₙ = zero(T)
            for k ∈ 1:size(B,1)
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] += Cₘₙ
        end
    end
    C
end
function mulsub!(
    C::AbstractStrideMatrix{M,N,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {M,N,T}
    check_matmul_sizes(C, A, B)
    @avx for m ∈ 1:size(C,1)
        for n ∈ 1:size(C,2)
            Cₘₙ = zero(T)
            for k ∈ 1:size(B,1)
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ - C[m,n]
        end
    end
    C
end
function nmuladd!(
    C::AbstractStrideMatrix{M,N,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {M,N,T}
    check_matmul_sizes(C, A, B)
    @avx for m ∈ 1:size(C,1)
        for n ∈ 1:size(C,2)
            Cₘₙ = zero(T)
            for k ∈ 1:size(B,1)
                Cₘₙ -= A[m,k] * B[k,n]
            end
            C[m,n] += Cₘₙ
        end
    end
    C
end
function nmulsub!(
    C::AbstractStrideMatrix{M,N,T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {M,N,T}
    check_matmul_sizes(C, A, B)
    @avx for m ∈ 1:size(C,1)
        for n ∈ 1:size(C,2)
            Cₘₙ = zero(T)
            for k ∈ 1:size(B,1)
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
    C::AbstractStrideMatrix{M,N,T},
    A::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeVector{M,T}},
    B::AbstractMatrix{T}
) where {M,N,T}
    check_matmul_sizes(C,A,B)
    vA = parent(A)
    @avx for n ∈ 1:size(B,2), m ∈ 1:size(B,1)
        C[m,n] = vA[m] * B[m,n]
    end
    C
end
function Base.:*(
    A::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeVector{M,T}},
    B::AbstractStrideMatrix{M,N,T}
) where {M,N,T}
    mul!(similar(B), A, B)
end
function Base.:*(
    sp::StackPointer,
    A::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeVector{M,T}},
    B::AbstractStrideMatrix{M,N,T}
) where {M,N,T}
    sp, C = similar(sp, B)
    sp, mul!(C, A, B)
end


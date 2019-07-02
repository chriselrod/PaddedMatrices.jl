

# @generated function Base.:*(A::AbstractConstantFixedSizePaddedMatrix{M,N,T,R1,L1}, B::AbstractConstantFixedSizePaddedMatrix{N,P,T,R2,L2}) where {M,N,P,T,R1,R2,L1,L2}
@generated function Base.:*(A::AbstractConstantFixedSizePaddedMatrix{M,N,T,R1,L1}, B::AbstractConstantFixedSizePaddedMatrix{N,P,T,R2,L2}) where {M,N,P,T,R1,R2,L2,L1}
    q = static_mul_quote(M,N,P,T,R1,R2)
    # push!(q.args,  :(ConstantFixedSizePaddedMatrix( out )) )
    push!(q.args,  :(ConstantFixedSizePaddedMatrix{$M,$P,$T,$R1,$(R1*P)}( output_data )) )
    q
end
@generated function Base.:*(
            A::AbstractConstantFixedSizePaddedMatrix{M,N,T,R1,L1},
            B::LinearAlgebra.Adjoint{T,<:AbstractConstantFixedSizePaddedMatrix{P,N,T,R2,L2}}
        ) where {M,N,P,T,R1,R2,L1,L2}
        # ) where {M,N,P,T,R1,R2,L2,L1}
    q = static_mul_nt_quote(M,N,P,T,R1,R2)
    # push!(q.args,  :(ConstantFixedSizePaddedMatrix( out )) )
    push!(q.args,  :(ConstantFixedSizePaddedMatrix{$M,$P,$T,$R1,$(R1*P)}( output_data )) )
    q
end
@generated function Base.:*(
            A::AbstractConstantFixedSizePaddedVector{M,T,R1,L1},
            B::LinearAlgebra.Adjoint{T,<:AbstractConstantFixedSizePaddedVector{P,T,R2,L2}}
        ) where {M,P,T,R1,R2,L1,L2}
        # ) where {M,N,P,T,R1,R2,L2,L1}
    q = static_mul_nt_quote(M,1,P,T,R1,1)
    # push!(q.args,  :(ConstantFixedSizePaddedMatrix( out )) )
    push!(q.args,  :(ConstantFixedSizePaddedMatrix{$M,$P,$T,$R1,$(R1*P)}( output_data )) )
    q
end
# @generated function Base.:*(A::LinearAlgebra.Adjoint{T,StaticSIMDVector{N,T,R1,L1}}, B::StaticSIMDMatrix{N,P,T,R2}) where {N,P,T,R1,R2,L1}
#     static_mul_quote(1,N,P,T,R1,R2)
# end
@generated function Base.:*(A::AbstractConstantFixedSizePaddedMatrix{M,N,T,R1}, B::AbstractConstantFixedSizePaddedVector{N,T,R2}) where {M,N,T,R1,R2}
    q = static_mul_quote(M,N,1,T,R1,R2)
    push!(q.args,  :(ConstantFixedSizePaddedVector{$M,$T,$R1,$R1}( output_data )) )
    q
end


@generated function Base.:*(
                A::AbstractMutableFixedSizePaddedMatrix{M,N,T,ADR},
                X::AbstractMutableFixedSizePaddedMatrix{N,P,T,XR}
            ) where {M,N,P,T,ADR,XR}
    q = quote
        $(Expr(:meta,:inline))
        D = MutableFixedSizePaddedMatrix{$M,$P,$T,$ADR,$(ADR*P)}(undef)
        pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        $(mulquote(ADR,N,P,ADR,XR,T))
        # ConstantFixedSizePaddedMatrix(D)
    end
    # if M * P > 224
    if M * P > 64
        push!(q.args, :D)
    else
        push!(q.args, :(ConstantFixedSizePaddedMatrix(D)))
    end
    q
end
@generated function Base.:*(
    sp::StackPointer,
    A::AbstractMutableFixedSizePaddedMatrix{M,N,T,ADR},
    X::AbstractMutableFixedSizePaddedMatrix{N,P,T,XR}
) where {M,N,P,T,ADR,XR}
    quote
        $(Expr(:meta,:inline))
        sp, D = PtrMatrix{$M,$P,$T,$ADR}(sp)
        pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        $(mulquote(ADR,N,P,ADR,XR,T))
        sp, D
    end
end


@inline function LinearAlgebra.mul!(C, A, B::DynamicPaddedMatrix)
    Bdata = B.data
    @uviews Bdata mul!(C, A, @view(B.data[1:B.nrow,:]))
end

@inline function LinearAlgebra.mul!(C, A::DynamicPaddedMatrix, B::DynamicPaddedMatrix)
    Bdata = B.data
    @uviews Bdata mul!(C, A.data, @view(B.data[1:B.nrow,:]))
end
@inline function LinearAlgebra.mul!(C::DynamicPaddedMatrix, A, B::DynamicPaddedMatrix)
    Bdata = B.data
    @uviews Bdata mul!(C.data, A, @view(B.data[1:B.nrow,:]))
end
@inline function LinearAlgebra.mul!(C::DynamicPaddedMatrix, A::DynamicPaddedMatrix, B::DynamicPaddedMatrix)
    Bdata = B.data
    @uviews Bdata mul!(C.data, A.data, @view(B.data[1:B.nrow,:]))
end

@generated function LinearAlgebra.mul!(D::AbstractMutableFixedSizePaddedMatrix{M,P,T,DR},
                            A::AbstractMutableFixedSizePaddedMatrix{M,N,T,AR},
                            X::AbstractMutableFixedSizePaddedMatrix{N,P,T,XR}) where {M,N,P,T,AR,XR,DR}
    quote
        $(Expr(:meta,:inline))
        pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        $(mulquote(AR,N,P,AR,XR,T,:initkernel!,nothing,DR))
    end
end


@generated function LinearAlgebra.mul!(D::AbstractMutableFixedSizePaddedVector{M,T,DR},
                            A::AbstractMutableFixedSizePaddedMatrix{M,N,T,AR},
                            X::AbstractMutableFixedSizePaddedVector{N,T,XR}) where {M,N,T,AR,XR,DR}
    quote
        $(Expr(:meta,:inline))
        pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        $(mulquote(AR,N,1,AR,XR,T,:initkernel!,nothing,DR))
    end
end
@generated function LinearAlgebra.mul!(D::PtrMatrix{M,P,T,DR,LD,PD},
                            A::PtrMatrix{M,N,T,AR,LA,PA},
                            X::PtrMatrix{N,P,T,XR}) where {M,N,P,T,AR,XR,DR,LA,PA,LD,PD}
    if PD && PA && (DR == AR)
        Meffective = DR
    else
        Meffective = M
    end
    quote
        $(Expr(:meta,:inline))
        pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        $(mulquote(Meffective,N,P,AR,XR,T,:initkernel!,nothing,DR))
    end
end
@generated function LinearAlgebra.mul!(D::PtrVector{M,T,DR,LD,PD},
                            A::PtrMatrix{M,N,T,AR,LA,PA},
                                       X::PtrVector{N,T,XR}) where {M,N,T,AR,XR,DR,LD,PD,LA,PA}
    if PD && PA && (DR == AR)
        Meffective = DR
    else
        Meffective = M
    end
    
    quote
        $(Expr(:meta,:inline))
        pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        $(mulquote(Meffective,N,1,AR,XR,T,:initkernel!,nothing,DR))
    end
end


function elementwise_product_quote(N,T,P)
    quote
        $(Expr(:meta,:inline)) # do we really want to force inline this?
        mv = MutableFixedSizePaddedVector{$N,$T,$P,$P}(undef)
        @vectorize $T for i ∈ 1:$P
            mv[i] = A[i] * B[i]
        end
    end
end

@generated function Base.:*(Adiagonal::Diagonal{T,<:AbstractFixedSizePaddedVector{N,T,P}}, B::AbstractFixedSizePaddedVector{N,T,P}) where {N,T,P}
    quote
        A = Adiagonal.diag
        $(elementwise_product_quote(N,T,P))
        ConstantFixedSizePaddedArray(mv)
    end
end
@generated function Base.:*(
            Aadjoint::LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedVector{N,T,P}},
            Bdiagonal::Diagonal{T,<:AbstractFixedSizePaddedVector{N,T,P}}
        ) where {N,T,P}
    quote
        A = Aadjoint.parent
        B = Bdiagonal.diag
        $(elementwise_product_quote(N,T,P))
        ConstantFixedSizePaddedArray(mv)'
    end
end
@generated function Base.:*(
            Adiagonal::Diagonal{T,<:AbstractMutableFixedSizePaddedVector{N,T,P}},
            B::AbstractMutableFixedSizePaddedVector{N,T,P}) where {N,T,P}
    quote
        A = Adiagonal.diag
        $(elementwise_product_quote(N,T,P))
        mv
    end
end
@generated function Base.:*(
    Aadjoint::LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizePaddedVector{N,T,P1}},
    Bdiagonal::Diagonal{T,<:AbstractMutableFixedSizePaddedVector{N,T,P2}}
) where {N,T,P1,P2}
    P = min(P1,P2)
    quote
        A = Aadjoint.parent
        B = Bdiagonal.diag
        $(elementwise_product_quote(N,T,P))
        mv'
    end
end
@generated function Base.:*(
    sp::StackPointer,
    Adiagonal::Diagonal{T,<:AbstractMutableFixedSizePaddedVector{N,T,P1}},
    B::AbstractMutableFixedSizePaddedVector{N,T,P2}
) where {N,T,P1,P2}
    P = min(P1,P2)
    quote
        sp, mv = PtrVector{$N,$T,$P}(sp)
        A = Adiagonal.diag
        @vvectorize $T for p ∈ 1:$P
            mv[p] = A[p] * B[p]
        end
        sp, mv
    end
end
@generated function Base.:*(
    sp::StackPointer,
    Aadjoint::LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizePaddedVector{N,T,P1}},
    Bdiagonal::Diagonal{T,<:AbstractMutableFixedSizePaddedVector{N,T,P2}}
) where {N,T,P1,P2}
    P = min(P1,P2)
    quote
#        $(Expr(:meta,:inline))
        sp, mv = PtrVector{$N,$T,$P}(sp)
        A = Aadjoint.parent
        B = Bdiagonal.diag
        @vvectorize $T for p ∈ 1:$P
            mv[p] = A[p] * B[p]
        end
        sp, mv'
    end
end


@generated function Base.:+(
    A::AbstractFixedSizePaddedArray{S,T,N,P,L},
    B::AbstractFixedSizePaddedArray{S,T,N,P,L}
) where {S,T<:Number,N,P,L}
    quote
        mv = MutableFixedSizePaddedArray{$S,$T,$N,$P,$L}(undef)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] + B[i]
        end
        ConstantFixedSizePaddedArray(mv)
    end
end
@inline function Base.:+(A::AbstractMutableFixedSizePaddedArray{S,T,N,P,L}, B::AbstractMutableFixedSizePaddedArray{S,T,N,P,L}) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = A[i] + B[i]
    end
    mv
end
@generated function Base.:+(
    sp::StackPointer,
    A::AbstractFixedSizePaddedArray{S,T,N,P,L},
    B::AbstractFixedSizePaddedArray{S,T,N,P,L}
) where {S,T<:Number,N,P,L}
    P = min(PA,PB)
    L = min(LA,LB)
    quote
        mv = PtrArray{$S,$T,$N,$P,$L}(pointer(sp,$T))
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] + B[i]
        end
        sp + $(sizeof(T)*L), mv
    end
end
@generated function Base.:+(
    sp::StackPointer,
    A::AbstractFixedSizePaddedVector{N,T,PA,PA},
    B::AbstractFixedSizePaddedVector{N,T,PB,PB}
) where {N,T<:Number,PA,PB}
    P = min(PA,PB)
    quote
        mv = PtrVector{$N,$T,$P,$P}(pointer(sp,$T))
        @vvectorize $T for i ∈ 1:$P
            mv[i] = A[i] + B[i]
        end
        sp + $(sizeof(T)*P), mv
    end
end
@inline function Base.:+(A::LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedArray{S,T,N,P,L}},
                        B::LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedArray{S,T,N,P,L}}) where {S,T,N,P,L}
    (A' + B')'
end
@inline function Base.:+(
    sp::StackPointer,
    A::LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedArray{S,T,N,PA,LA}},
    B::LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedArray{S,T,N,PB,LB}}
) where {S,T,N,PA,PB,LA,LB}
    sp2, C = (+(sp, A', B'))
    sp2, C'
end
@inline function Base.:+(a::T, B::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = a + B[i]
    end
    ConstantFixedSizePaddedArray(mv)
end
@inline function Base.:+(A::AbstractFixedSizePaddedArray{S,T,N,P,L}, b::T) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = A[i] + b
    end
    ConstantFixedSizePaddedArray(mv)
end
@inline function Base.:+(a::T, Badj::LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedArray{S,T,N,P,L}}) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    B = Badj.parent
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = a + B[i]
    end
    ConstantFixedSizePaddedArray(mv)'
end
@inline function Base.:+(Aadj::LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedArray{S,T,N,P,L}}, b::T) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    A = Aadj.parent
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = A[i] + b
    end
    ConstantFixedSizePaddedArray(mv)'
end
@inline function Base.:-(A::AbstractFixedSizePaddedArray{S,T,N,P,L}, B::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = A[i] - B[i]
    end
    ConstantFixedSizePaddedArray(mv)
end
@inline function diff!(C::MutableFixedSizePaddedArray{S,T,N,P,L}, A::AbstractFixedSizePaddedArray{S,T,N,P,L}, B::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T<:Number,N,P,L}
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        C[i] = A[i] - B[i]
    end
    C
end
@inline function Base.:-(A::AbstractFixedSizePaddedArray{S,T,N,P,L}, b::T) where {S,T<:Number,N,L,P}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = A[i] - b
    end
    ConstantFixedSizePaddedArray(mv)
end
@inline function Base.:-(a::T, B::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = a - B[i]
    end
    ConstantFixedSizePaddedArray(mv)
end


@generated function Base.sum!(s::AbstractMutableFixedSizePaddedVector{P,T,L,L}, A::AbstractPaddedMatrix{T}) where {T,P,L}
    stride_bytes = L*sizeof(T)
    sample_mat_stride = L

    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)

    WT = W * sizeof(T)
    V = Vec{W,T}

    # +2, to divide by an additional 4
    iterations = L >> (Wshift + 2)
    r = L & ((W << 2) - 1)
    riter = r >> Wshift
    remainder_quote = quote
        Base.Cartesian.@nexprs $riter j -> begin
            offset_j = $(4WT*iterations) + $WT*(j-1)
            x_j = SIMDPirates.vbroadcast($V, zero($T))
        end
        for n ∈ 0:N-1
            offset_n = n * $stride_bytes
            Base.Cartesian.@nexprs $riter j -> begin
                x_j = SIMDPirates.vadd(x_j, SIMDPirates.vload($V, ptrA + offset_n + offset_j))
            end
        end
        Base.Cartesian.@nexprs $riter j -> SIMDPirates.vstore!(ptrs + offset_j, x_j)
    end

    quote
        # x̄ = zero(PaddedMatrices.MutableFixedSizePaddedVector{P,T})
        # s = PaddedMatrices.MutableFixedSizePaddedVector{$P,$T}(undef)
        ptrs = pointer(s)
        ptrA = pointer(A)
        N = size(A,2)
        GC.@preserve s A begin
            for i ∈ 0:$(iterations-1)
                Base.Cartesian.@nexprs 4 j -> begin
                    offset_j = $(4WT)*i + $WT*(j-1)
                    x_j = SIMDPirates.vbroadcast($V, zero($T))
                end
                for n ∈ 0:N-1
                    offset_n = n * $stride_bytes
                    Base.Cartesian.@nexprs 4 j -> begin
                        x_j = SIMDPirates.vadd(x_j, SIMDPirates.vload($V, ptrA + offset_n + offset_j))
                    end
                end
                Base.Cartesian.@nexprs 4 j -> SIMDPirates.vstore!(ptrs + offset_j, x_j)
            end
            $(riter == 0 ? nothing : remainder_quote)
        end
        s
    end
end
@generated function negative_sum!(s::AbstractMutableFixedSizePaddedVector{P,T,L,L}, A::AbstractPaddedMatrix{T}) where {T,P,L}
    stride_bytes = L*sizeof(T)
    sample_mat_stride = L

    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)

    WT = W * sizeof(T)
    V = Vec{W,T}

    # +2, to divide by an additional 4
    iterations = L >> (Wshift + 2)
    r = L & ((W << 2) - 1)
    riter = r >> Wshift
    remainder_quote = quote
        Base.Cartesian.@nexprs $riter j -> begin
            offset_j = $(4WT*iterations) + $WT*(j-1)
            x_j = SIMDPirates.vbroadcast($V, zero($T))
        end
        for n ∈ 0:N-1
            offset_n = n * $stride_bytes
            Base.Cartesian.@nexprs $riter j -> begin
                x_j = SIMDPirates.vsub(x_j, SIMDPirates.vload($V, ptrA + offset_n + offset_j))
            end
        end
        Base.Cartesian.@nexprs $riter j -> SIMDPirates.vstore!(ptrs + offset_j, x_j)
    end

    quote
        # x̄ = zero(PaddedMatrices.MutableFixedSizePaddedVector{P,T})
        # s = PaddedMatrices.MutableFixedSizePaddedVector{$P,$T}(undef)
        ptrs = pointer(s)
        ptrA = pointer(A)
        N = size(A,2)
        GC.@preserve s A begin
            for i ∈ 0:$(iterations-1)
                Base.Cartesian.@nexprs 4 j -> begin
                    offset_j = $(4WT)*i + $WT*(j-1)
                    x_j = SIMDPirates.vbroadcast($V, zero($T))
                end
                for n ∈ 0:N-1
                    offset_n = n * $stride_bytes
                    Base.Cartesian.@nexprs 4 j -> begin
                        x_j = SIMDPirates.vsub(x_j, SIMDPirates.vload($V, ptrA + offset_n + offset_j))
                    end
                end
                Base.Cartesian.@nexprs 4 j -> SIMDPirates.vstore!(ptrs + offset_j, x_j)
            end
            $(riter == 0 ? nothing : remainder_quote)
        end
        s
    end
end
# @generated function prodmuls(coefficients::NTuple{N,T}, As::NTuple{N,<:AbstractFixedSizePaddedArray{S,T,N,P,L}}) where {S,T,N,P,L}


# end

@inline extract_λ(a) = a
@inline extract_λ(a::UniformScaling) = a.λ
@generated function Base.:*(A::AbstractFixedSizePaddedArray{S,T,N,P,L}, bλ::Union{T,UniformScaling{T}}) where {S,T<:Real,N,P,L}
    quote
        mv = MutableFixedSizePaddedArray{$S,$T,$N,$P,$L}(undef)
        b = extract_λ(bλ)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] * b
        end
        ConstantFixedSizePaddedArray(mv)
    end
end
@generated function Base.:*(Aadj::LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedArray{S,T,N,P,L}}, bλ::Union{T,UniformScaling{T}}) where {S,T<:Real,N,P,L}
    quote
        mv = MutableFixedSizePaddedArray{$S,$T,$N,$P,$L}(undef)
        A = Aadj.parent
        b = extract_λ(bλ)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] * b
        end
        ConstantFixedSizePaddedArray(mv)'
    end
end
@generated function Base.:*(
    sp::StackPointer,
    A::AbstractFixedSizePaddedArray{S,T,N,P,L},
    bλ::Union{T,UniformScaling{T}}
) where {S,T<:Real,N,P,L}
    quote
        mv = PtrArray{$S,$T,$N,$P,$L}(pointer(sp,$T))
        b = extract_λ(bλ)
        @vvectorize for i ∈ 1:$L
            mv[i] = A[i] * b
        end
        sp + $(sizeof(T)*L), mv
    end
end
@generated function Base.:*(
    sp::StackPointer,
    Aadj::LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedArray{S,T,N,P,L}},
    bλ::Union{T,UniformScaling{T}}
) where {S,T<:Real,N,P,L}
    quote
        mv = PtrArray{$S,$T,$N,$P,$L}(pointer(sp,$T))
        A = Aadj.parent
        b = extract_λ(bλ)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] * b
        end
        sp + $(sizeof(T)*L), mv'
    end
end
@generated function Base.:*(a::T, B::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T<:Number,N,P,L}
    quote
        mv = MutableFixedSizePaddedArray{$S,$T,$N,$P,$L}(undef)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = a * B[i]
        end
        ConstantFixedSizePaddedArray(mv)
    end
end
@inline function SIMDPirates.vmuladd(a::T, x::AbstractFixedSizePaddedArray{S,T,N,P,L}, y::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = a * x[i] + y[i]
    end
    ConstantFixedSizePaddedArray(mv)
end
@inline function SIMDPirates.vmuladd(a::T,
                x::LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedVector{N,T,L,L}},
                y::LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedVector{N,T,L,L}}
            ) where {N,T,L}
    mv = MutableFixedSizePaddedVector{N,T,L,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = a * x[i] + y[i]
    end
    ConstantFixedSizePaddedArray(mv)
end

# @generated function SIMDPirates.vmuladd(a::T,
#                 x::Union{<:AbstractFixedSizePaddedVector{P1,T,L1},LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedVector{P1,T,L1}}}, y::Union{<:AbstractFixedSizePaddedVector{P2,T,L2},LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedVector{P2,T,L2}}}
#                 ) where {T,P1,L1,P2,L2}
#     if x <: LinearAlgebra.Adjoint
#         @assert y <: LinearAlgebra.Adjoint
#     else
#         @assert !( y <: LinearAlgebra.Adjoint )
#     end
#     if L1 < L2
#         return quote
#             mv = MutableFixedSizePaddedVector{$P2,$T,$L2}(undef)
#             @fastmath @inbounds @simd ivdep for i ∈ 1:$L1
#                 mv[i] = a * x[i] + y[i]
#             end
#             @inbounds @simd ivdep for i ∈ $(L1+1):$L2
#                 mv[i] = y[i]
#             end
#             $( x <: LinearAlgebra.Adjoint ? :(ConstantFixedSizePaddedArray(mv)') : :(ConstantFixedSizePaddedArray(mv)) )
#         end
#     elseif L1 > L2
#         return quote
#             mv = MutableFixedSizePaddedVector{$P1,$T,$L1}(undef)
#             @fastmath @inbounds @simd ivdep for i ∈ 1:$L2
#                 mv[i] = a * x[i] + y[i]
#             end
#             @inbounds @simd ivdep for i ∈ $(L2+1):$L1
#                 mv[i] = a * x[i]
#             end
#             $( x <: LinearAlgebra.Adjoint ? :(ConstantFixedSizePaddedArray(mv)') : :(ConstantFixedSizePaddedArray(mv)) )
#         end
#     else
#         return quote
#             mv = MutableFixedSizePaddedVector{$P1,$T,$L1}(undef)
#             @fastmath @inbounds @simd ivdep for i ∈ 1:$L1
#                 mv[i] = a * x[i] + y[i]
#             end
#             $( x <: LinearAlgebra.Adjoint ? :(ConstantFixedSizePaddedArray(mv)') : :(ConstantFixedSizePaddedArray(mv)) )
#         end
#     end
# end
@inline function SIMDPirates.vfnmadd(a::T, x::AbstractFixedSizePaddedArray{S,T,N,P,L}, y::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = y[i] - a * x[i]
    end
    ConstantFixedSizePaddedArray(mv)
end

@generated function LinearAlgebra.dot(A::AbstractFixedSizePaddedArray{S,T,N,P,L}, B::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L}
    if N > 1
        ST = ntuple(n -> S.parameters[n], Val(N))
        return quote
            out = zero(T)
            ind = 0
            @nloops $(N-1) i j -> 1:$ST[j+1] begin
                @vectorize $T for i_0 ∈ 1:$(ST[1])
                    out += A[ind + i_0] * B[ind + i_0]
                end
                ind += P
            end
            out
        end
    else #if N == 1
        return quote
            out = zero(T)
            @vectorize $T  for i ∈ 1:$(S.parameters[1])
                out += A[i] * B[i]
            end
            out
        end
    end
end
@generated function dot_self(A::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T,N,P,L}
    if N > 1
        ST = ntuple(n -> S.parameters[n], Val(N))
        return quote
            out = zero(T)
            ind = 0
            @nloops $(N-1) i j -> 1:$ST[j+1] begin
                @vectorize $T for i_0 ∈ 1:$(ST[1])
                    Aᵢ = A[ind + i_0]
                    out += Aᵢ * Aᵢ
                end
                ind += P
            end
            out
        end
    else #if N == 1
        return quote
            out = zero(T)
            @vectorize $T  for i ∈ 1:$(S.parameters[1])
                out += A[i] * A[i]
            end
            out
        end
    end
end

"""
Not within BLAS module, because we aren't supporting the full gemm API at the moment.
This simply calculates:
D += A*X
"""
@generated function gemm!(D::AbstractMutableFixedSizePaddedMatrix{M,P,T,ADR},
                            A::AbstractMutableFixedSizePaddedMatrix{M,N,T,ADR},
                            X::AbstractMutableFixedSizePaddedMatrix{N,P,T,XR}) where {M,N,P,T,ADR,XR}

    mulquote(ADR,N,P,ADR,XR,T,:kernel!)
end

function gemm(
            D::AbstractMutableFixedSizePaddedMatrix{M,P,T,ADR},
            A::AbstractMutableFixedSizePaddedMatrix{M,N,T,ADR},
            X::AbstractMutableFixedSizePaddedMatrix{N,P,T,XR}
        ) where {M,N,P,T,ADR,XR}
    C = copy(D)
    gemm!(C, A, X)
    C
end
# @generated function gemm!(D::PtrMatrix{M,P,T,ADR},
#                             A::PtrMatrix{M,N,T,ADR},
#                             X::PtrMatrix{N,P,T,XR},
#                             prefetchAX = nothing) where {M,N,P,T,ADR,XR}
#
#     mulquote(M, N, P, ADR, XR, T, :kernel!, prefetchAX)
# end

function mulquote(D::AbstractMutableFixedSizePaddedMatrix{M,P,T,ADR},
                A::AbstractMutableFixedSizePaddedMatrix{M,N,T,ADR},
                X::AbstractMutableFixedSizePaddedMatrix{N,P,T,XR},init = :initkernel!) where {M,N,P,T,ADR,XR}
    quote
        $(Expr(:meta,:inline))
        pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        $(mulquote(AR,N,P,AR,XR,T,init,nothing,DR))
    end
end
function mulquote(M,N,P,AR,XR,T,init=:initkernel!,prefetchAX=nothing,DR=AR)
    (L1S, L2S, L3S), num = blocking_structure(M, N, P, T)
    if num == 0 || (M*N*P < 104^3)
        # if init == :kernel! || M*P > 14*16
            return cache_mulquote(M,N,P,AR,XR,L1S,T,init,DR)
        # else
            # M, P, strideA, strideX, N, T
            # return kernel_quote(M,P,ADR,XR,N,T,true,true,CDR)
        # end
        # return base_mulquote(M,N,P,ADR,XR,T)
    # elseif num == 1
    #     # Loop over L1 cache blocks
    #     return cache_mulquote(M,N,P,ADR,XR,L1S,T,init)#,prefetchAX)
    elseif T <: LinearAlgebra.BlasFloat
        # blas_gemm_quote(M,N,P,DR,AR,XR,T,init)
        :(BLAS.gemm!('N','N',one($T),A,X,$(init==:initkernel! ? zero(T) : one(T)),D))
    elseif num == 1
        return cache_mulquote(M,N,P,AR,XR,L1S,T,init,DR)
    elseif num == 2
        # Loop over L2 cache blocks
        return cache_mulquote(M,N,P,AR,XR,L1S,L2S,T,init,prefetchAX)
    else #num == 3
        # Loop over L3 cache blocks.
        # return cache_mulquote(ADR,N,P,ADR,XR,L1S,L2S,L3S,T,init)
        # Except that they are fused, so we aren't doing anything special.
        return cache_mulquote(M,N,P,AR,XR,L1S,L3S,T,init,prefetchAX) # should recurse calling mul and gemm!
    end
end

# function blas_gemm_quote(M,N,P,DR,AR,XR,T,init)
#     quote
#         ccall((@blasfunc))
#     end
# end

function initkernel_quote(D::AbstractMutableFixedSizePaddedMatrix{M,Pₖ,T,stride_AD},
                            A::AbstractMutableFixedSizePaddedMatrix{M,N,T,stride_AD},
                            X::AbstractMutableFixedSizePaddedMatrix{N,Pₖ,T,stride_X}) where {M,Pₖ,stride_AD,stride_X,N,T}
    kernel_quote(M,Pₖ,stride_AD,stride_X,N,T,true,true)
end

function block_loop_quote(L1M,L1N,L1P,stride_AD,stride_X,M_iter,M_remain,P_iter,P_remain,T_size,kernel=:kernel!,pA=:pAₙ,pX=:pXₙ,pD=:pD,X_transposed=false)

    if M_remain == 0
        D = :($pD + $(T_size*L1M)*mᵢ + $(T_size*L1P*stride_AD)*pᵢ)
        A = :($pA + $(T_size*L1M)*mᵢ)
        X = X_transposed ? pX : :($pX + $(T_size*L1P*stride_X)*pᵢ)
        if P_iter > M_iter + 1 # Excess of 1 is okay.
            PM_ratio, PM_remainder = divrem(P_iter, M_iter)
            q = quote
                    for pmᵣ ∈ 1:$PM_ratio, pᵢ ∈ (pmᵣ-1)*$M_iter:$M_iter*pmᵣ - 1
                        for mᵢ ∈ $M_iter*pmᵣ - pᵢ:$(M_iter-1)
                            $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_AD,$stride_X,$L1N}())
                        end
                        for mᵢ ∈ 0:$M_iter*pmᵣ - pᵢ - 1
                            $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_AD,$stride_X,$L1N}())
                        end
                    end
                    for pᵢ ∈ $(M_iter*PM_ratio):$(P_iter-1)
                        for mᵢ ∈ $(M_iter*(PM_ratio+1))-pᵢ:$(M_iter-1)
                            $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_AD,$stride_X,$L1N}())
                        end
                        for mᵢ ∈ 0:$(M_iter*(PM_ratio+1)-1)-pᵢ
                            $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_AD,$stride_X,$L1N}())
                        end
                    end
            end
            MP_terminal = PM_remainder == 0 ? 0 : M_iter - PM_remainder # ==  (M_iter*(PM_ratio+1)) - P_iter
        else
            q = quote
                # $prefetch_quote
                for pᵢ ∈ 0:$(P_iter-1)
                    for mᵢ ∈ $(M_iter)-pᵢ:$(M_iter-1)
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_AD,$stride_X,$L1N}())
                    end
                    for mᵢ ∈ 0:$(M_iter-1)-pᵢ
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_AD,$stride_X,$L1N}())
                    end
                end # for pᵢ ∈ 0:$(P_iter-1)
            end # quote
            MP_terminal = P_iter == M_iter + 1 ? M_iter - 1 : M_iter - P_iter
        end
    else
        ### Here
        D = :($pD + $(T_size*L1M)*mᵢ + $(T_size*L1P*stride_AD)*pᵢ)
        A = :($pA + $(T_size*L1M)*mᵢ)
        X = X_transposed ? pX : :($pX + $(T_size*L1P*stride_X)*pᵢ)
        D_r = :($pD + $(T_size*L1M*M_iter) + $(T_size*L1P*stride_AD)*pᵢ)
        A_r = :($pA + $(T_size*L1M*M_iter))
        if P_iter > M_iter + 2
            # Here, we insert a kernel call to "M_remain" that is of an abridged size.
            # period for rotation over M is one longer.
            PM_ratio, PM_remainder = divrem(P_iter, M_iter + 1)
            q = quote
                # $prefetch_quote
                for mᵢ ∈ 0:$(M_iter-1)
                    $(kernel)($pD + $(T_size*L1M)*mᵢ, $A, $pX, Kernel{$L1M,$L1P,$stride_AD,$stride_X,$L1N}())
                end
                $(kernel)($pD + $(T_size*L1M*M_iter), $A_r, $pX, Kernel{$M_remain,$L1P,$stride_AD,$stride_X,$L1N}())
                for pᵢ ∈ 1:$(M_iter+1)
                    for mᵢ ∈ $(M_iter+1) - pᵢ:$(M_iter-1)
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_AD,$stride_X,$L1N}())
                    end
                    $(kernel)($D_r, $A_r, $X, Kernel{$M_remain,$L1P,$stride_AD,$stride_X,$L1N}())
                    for mᵢ ∈ 0:$M_iter - pᵢ
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_AD,$stride_X,$L1N}())
                    end
                end
                for pmᵣ ∈ 2:$(PM_ratio+1), pᵢ ∈ (pmᵣ-1)*$(M_iter+1)+1:min($(M_iter+1)*pmᵣ,$(P_iter-1))
                    for mᵢ ∈ $(M_iter+1)*pmᵣ - pᵢ:$(M_iter-1)
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_AD,$stride_X,$L1N}())
                    end
                    $(kernel)($D_r, $A_r, $X, Kernel{$M_remain,$L1P,$stride_AD,$stride_X,$L1N}())
                    for mᵢ ∈ 0:$(M_iter+1)*pmᵣ - pᵢ - 1
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_AD,$stride_X,$L1N}())
                    end
                end
            end
            MP_terminal = PM_remainder == 0 ? M_iter : (M_iter+1) - PM_remainder

        else
            q = quote
                for mᵢ ∈ 0:$(M_iter-1)
                    $(kernel)($pD + $(T_size*L1M)*mᵢ, $A, $pX, Kernel{$L1M,$L1P,$stride_AD,$stride_X,$L1N}())
                end
                $(kernel)($pD + $(T_size*L1M*M_iter), $A_r, $pX, Kernel{$M_remain,$L1P,$stride_AD,$stride_X,$L1N}())
                for pᵢ ∈ 1:$(P_iter-1)
                    for mᵢ ∈ $(M_iter+1)-pᵢ:$(M_iter-1)
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_AD,$stride_X,$L1N}())
                    end
                    $(kernel)($D_r, $A_r, $X, Kernel{$M_remain,$L1P,$stride_AD,$stride_X,$L1N}())
                    for mᵢ ∈ 0:$M_iter-pᵢ
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_AD,$stride_X,$L1N}())
                    end
                end
            end
            MP_terminal = P_iter == M_iter + 2 ? M_iter : M_iter + 1 - P_iter
        end
    end

    if P_remain != 0
        if M_remain == 0
            D = :($pD + $(T_size*L1M)*mᵢ + $(T_size*L1P*stride_AD*P_iter))
            A = :($pA + $(T_size*L1M)*mᵢ)
            X = X_transposed ? pX : :($pX + $(T_size*L1P*stride_X*P_iter))
            push!(q.args,
            quote
                for mᵢ ∈ $(MP_terminal):$(M_iter-1)
                    $(kernel)($D, $A, $X, Kernel{$L1M,$P_remain,$stride_AD,$stride_X,$L1N}())
                end
                for mᵢ ∈ 0:$(MP_terminal-1)
                    $(kernel)($D, $A, $X, Kernel{$L1M,$P_remain,$stride_AD,$stride_X,$L1N}())
                end
            end
            )
        else
            D = :($pD + $(T_size*L1M)*mᵢ + $(T_size*L1P*stride_AD*P_iter))
            A = :($pA + $(T_size*L1M)*mᵢ)
            X = X_transposed ? pX : :($pX + $(T_size*L1P*stride_X*P_iter))
            D_r = :($pD + $(T_size*L1M*M_iter) + $(T_size*L1P*stride_AD*P_iter))
            A_r = :($pA + $(T_size*L1M*M_iter))
            push!(q.args,
            quote
                for mᵢ ∈ $(MP_terminal):$(M_iter-1)
                    $(kernel)($D, $A, $X, Kernel{$L1M,$P_remain,$stride_AD,$stride_X,$L1N}())
                end
                $(kernel)($D_r, $A_r, $X, Kernel{$M_remain,$P_remain,$stride_AD,$stride_X,$L1N}())
                for mᵢ ∈ 0:$(MP_terminal-1)
                    $(kernel)($D, $A, $X, Kernel{$L1M,$P_remain,$stride_AD,$stride_X,$L1N}())
                end
            end
            )
        end

    end
    q
end

function cache_mulquote(M,N,P,stride_A,stride_X,(L1M,L1N,L1P),::Type{T}, init = :initkernel!, stride_D::Integer = stride_A) where {T}

    primary = :kernel!
    M_iter, M_remain = divrem(M, L1M)
    N_iter, N_remain = divrem(N, L1N)
    P_iter, P_remain = divrem(P, L1P)
    T_size = sizeof(T)
    if (M_iter == 0 || ((M_iter == 1) && (M_remain == 0))) && (P_iter == 0 || ((P_iter == 1) && (P_remain == 0)))
        return kernel_quote(M,P,stride_A,stride_X,N,T,true,true,stride_D)
    end
    if stride_A == stride_D
        stride_AD = stride_A
    else
        throw("Stride of A == $stride_A != stride of D == $stride_D")
    end

    q = quote
        pD, pA, pX = pointer(D), pointer(A), pointer(X)

        $(block_loop_quote(L1M,L1N,L1P,stride_AD,stride_X,M_iter,M_remain,P_iter,P_remain,T_size,init,:pA,:pX,:pD))
    end

    AN_stride = stride_AD * L1N * T_size
    XN_stride = stride_X  * L1N * T_size

    if N_iter > 2
        push!(q.args,
        quote
            for n ∈ 1:$(N_iter-1)
                pAₙ = pA + n*$(L1N * T_size * stride_AD)
                pXₙ = pX + n*$(L1N * T_size)
                $(block_loop_quote(L1M,L1N,L1P,stride_AD,stride_X,M_iter,M_remain,P_iter,P_remain,T_size,primary,:pAₙ,:pXₙ,:pD))
            end
        end
        )

    elseif N_iter == 2
        push!(q.args,
        quote
            pAₙ = pA + $(L1N * T_size * stride_AD)
            pXₙ = pX + $(L1N * T_size)
            $(block_loop_quote(L1M,L1N,L1P,stride_AD,stride_X,M_iter,M_remain,P_iter,P_remain,T_size,primary,:pAₙ,:pXₙ,:pD))
        end
        )
    end
    if N_remain > 0 # we need two goes
        push!(q.args,
        quote
            pAₙ = pA + $(L1N*N_iter * T_size * stride_AD)
            pXₙ = pX + $(L1N*N_iter * T_size)
            $(block_loop_quote(L1M,N_remain,L1P,stride_AD,stride_X,M_iter,M_remain,P_iter,P_remain,T_size,primary,:pAₙ,:pXₙ,:pD))
        end
        )
    end
    q
end



function cache_mulquote(M,N,P,stride_AD,stride_X,(L1M,L1N,L1P),(L2M,L2N,L2P),::Type{T}, init = :initkernel!, prefetchAX = nothing) where T
    M_iter, M_remain = divrem(M, L2M)
    N_iter, N_remain = divrem(N, L2N)
    P_iter, P_remain = divrem(P, L2P)
    T_size = sizeof(T)
    prefetch_ = prefetchAX != nothing
    initmul = ifelse(init == :initkernel!, :mul!, :gemm!)

    q = quote
        pD, pA, pX = pointer(D), pointer(A), pointer(X)


        for pᵢ ∈ 0:$(P_iter-1)
            pd_off = pᵢ*$(L2P*stride_AD*T_size)
            px_off = pᵢ*$(L2P*stride_X*T_size)
            for mᵢ ∈ 0:$(M_iter-1)
                m_off = mᵢ*$(L2M*T_size)
                pD_temp = PtrMatrix{$L2M,$L2P,$T,$stride_AD,$(stride_AD*L2P),false}(pD + m_off + pd_off)
                pA_temp = PtrMatrix{$L2M,$L2N,$T,$stride_AD,$(stride_AD*L2N),false}(pA + m_off)
                pX_temp = PtrMatrix{$L2N,$L2P,$T,$stride_X,$(stride_X*L2P),false}(pX + px_off)
                $(prefetch_ ? quote
                    prefetch(pD_temp, Val(1))
                    prefetch(pA_temp, Val(0))
                    prefetch(pX_temp, Val(0))
                end : nothing )
                $initmul(pD_temp,
                     pA_temp,
                     pX_temp)
                for nᵢ ∈ 1:$(N_iter-1)
                    pA_temp = PtrMatrix{$L2M,$L2N,$T,$stride_AD,$(stride_AD*L2N),false}(pA + m_off + nᵢ*$(L2N*stride_AD*T_size))
                    pX_temp = PtrMatrix{$L2N,$L2P,$T,$stride_X,$(stride_X*L2p),false}(pX + nᵢ*$(L2N*T_size) + px_off)
                    $(prefetch_ ? quote
                        prefetch(pA_temp, Val(0))
                        prefetch(pX_temp, Val(0))
                    end : nothing )
                    gemm!(pD_temp,
                         pA_temp,
                         pX_temp)
                end
                $(N_remain == 0 ? nothing : quote
                    pA_temp = PtrMatrix{$L2M,$N_remain,$T,$stride_AD,$(stride_AD*N_remain),false}(pA + m_off + $(N_iter*L2N*stride_AD*T_size))
                    pX_temp = PtrMatrix{$N_remain,$L2P,$T,$stride_X,$(stride_X*L2P),false}(pX + $(N_iter*L2N*T_size) + px_off)
                    $(prefetch_ ? quote
                        prefetch(pA_temp, Val(0))
                        prefetch(pX_temp, Val(0))
                    end : nothing )
                    gemm!(pD_temp, pA_temp, pX_temp)
                end)
            end
            ### Check if we need to add an expression for a remainder of M.
            $(M_remain == 0 ? nothing : quote
                pD_temp = PtrMatrix{$M_remain,$L2P,$T,$stride_AD,$(stride_AD*L2P),false}(pD + $(M_iter*L2M*T_size) + pd_off)
                pA_temp = PtrMatrix{$M_remain,$L2N,$T,$stride_AD,$(stride_AD*L2N),false}(pA + $(M_iter*L2M*T_size))
                pX_temp = PtrMatrix{$L2N,$L2P,$T,$stride_X,$(stride_X*L2P),false}(pX + px_off)
                $(prefetch_ ? quote
                    prefetch(pD_temp, Val(1))
                    prefetch(pA_temp, Val(0))
                    prefetch(pX_temp, Val(0))
                end : nothing )
                $initmul(pD_temp,
                     pA_temp,
                     pX_temp)
                for nᵢ ∈ 1:$(N_iter-1)
                    pA_temp = PtrMatrix{$M_remain,$L2N,$T,$stride_AD,$(stride_AD*L2N),false}(pA + $(M_iter*L2M*T_size) + nᵢ*$(L2N*stride_AD*T_size))
                    pX_temp = PtrMatrix{$L2N,$L2P,$T,$stride_X,$(stride_X*L2P),false}(pX + nᵢ*$(L2N*T_size) + px_off)
                    $(prefetch_ ? quote
                        prefetch(pA_temp, Val(0))
                        prefetch(pX_temp, Val(0))
                    end : nothing )
                    gemm!(pD_temp,
                         pA_temp,
                         pX_temp)
                end
                $(N_remain == 0 ? nothing : quote
                    pA_temp = PtrMatrix{$M_remain,$N_remain,$T,$stride_AD,$(stride_AD*N_remain),false}(pA + $(M_iter*L2M*T_size + N_iter*L2N*stride_AD*T_size))
                    pX_temp = PtrMatrix{$N_remain,$L2P,$T,$stride_X,$(stride_X*L2P),false}(pX + $(N_iter*L2N*T_size) + px_off)
                    $(prefetch_ ? quote
                        prefetch(pA_temp, Val(0))
                        prefetch(pX_temp, Val(0))
                    end : nothing )
                    gemm!(pD_temp,
                     pA_temp,
                     pX_temp)
                end )
            end) # $(M_remain == 0 ? nothing
        end # for pᵢ ∈ 0:$(P_iter-1)
    end # quote

    if P_remain > 0
        push!(q.args,
            quote
                for mᵢ ∈ 0:$(M_iter-1)
                    m_off = mᵢ*$(L2M*T_size)
                    pD_temp = PtrMatrix{$L2M,$P_remain,$T,$stride_AD,$(stride_AD*P_remain),false}(pD + m_off + $(P_iter*L2P*stride_AD*T_size))
                    pA_temp = PtrMatrix{$L2M,$L2N,$T,$stride_AD,$(stride_AD*L2N),false}(pA + m_off)
                    pX_temp = PtrMatrix{$L2N,$P_remain,$T,$stride_X,$(stride_X*P_remain),false}(pX + $(P_iter*L2P*stride_X*T_size))
                    $(prefetch_ ? quote
                        prefetch(pD_temp, Val(1))
                        prefetch(pA_temp, Val(0))
                        prefetch(pA_temp, Val(0))
                    end : nothing )
                    $initmul(pD_temp,
                         pA_temp,
                         pX_temp)
                    for nᵢ ∈ 1:$(N_iter-1)
                        pA_temp = PtrMatrix{$L2M,$L2N,$T,$stride_AD,$(stride_AD*L2N),false}(pA + m_off + nᵢ*$(L2N*stride_AD*T_size))
                        pX_temp = PtrMatrix{$L2N,$P_remain,$T,$stride_X,$(stride_X*P_remain),false}(pX + nᵢ*$(L2N*T_size) + $(P_iter*L2P*stride_X*T_size))
                        $(prefetch_ ? quote
                            prefetch(pA_temp, Val(0))
                            prefetch(pX_temp, Val(0))
                        end : nothing )
                        gemm!(pD_temp,
                             pA_temp,
                             pX_temp)
                    end
                    $(N_remain == 0 ? nothing : quote
                        pA_temp = PtrMatrix{$L2M,$N_remain,$T,$stride_AD,$(stride_AD*N_remain),false}(pA + m_off + $(N_iter*L2N*stride_AD*T_size))
                        pX_temp = PtrMatrix{$N_remain,$P_remain,$T,$stride_X,$(stride_X*P_remain),false}(pX + $(N_iter*L2N*T_size + P_iter*L2P*stride_X*T_size))
                        $(prefetch_ ? quote
                            prefetch(pA_temp, Val(0))
                            prefetch(pX_temp, Val(0))
                        end : nothing )
                        gemm!(pD_temp,
                         pA_temp,
                         pX_temp)
                    end )
                end
                ### Check if we need to add an expression for a remainder of M.
                $(M_remain == 0 ? nothing : quote
                    pD_temp = PtrMatrix{$M_remain,$P_remain,$T,$stride_AD,$(stride_AD*P_remain),false}(pD + $(M_iter*L2M*T_size + P_iter*L2P*stride_AD*T_size))
                    pA_temp = PtrMatrix{$M_remain,$L2N,$T,$stride_AD,$(stride_AD*L2N),false}(pA + $(M_iter*L2M*T_size))
                    pX_temp = PtrMatrix{$L2N,$P_remain,$T,$stride_X,$(stride_X*P_remain),false}(pX + $(P_iter*L2P*stride_X*T_size))
                    $(prefetch_ ? quote
                        prefetch(pD_temp, Val(1))
                        prefetch(pA_temp, Val(0))
                        prefetch(pX_temp, Val(0))
                    end : nothing )
                    $initmul(pD_temp,
                         pA_temp,
                         pX_temp)
                    for nᵢ ∈ 1:$(N_iter-1)
                        pA_temp = PtrMatrix{$M_remain,$L2N,$T,$stride_AD,$(stride_AD*L2N),false}(pA + $(M_iter*L2M*T_size) + nᵢ*$(L2N*stride_AD*T_size))
                        pX_temp = PtrMatrix{$L2N,$P_remain,$T,$stride_X,$(stride_X*P_remain),false}(pX + nᵢ*$(L2N*T_size) + $(P_iter*L2P*stride_X*T_size))
                        $(prefetch_ ? quote
                            prefetch(pA_temp, Val(0))
                            prefetch(pX_temp, Val(0))
                        end : nothing )
                        gemm!(pD_temp,
                             pA_temp,
                             pX_temp)
                    end
                    $(N_remain == 0 ? nothing : quote
                        pA_temp = PtrMatrix{$M_remain,$N_remain,$T,$stride_AD,$(stride_AD*N_remain),false}(pA + $(M_iter*L2M*T_size + N_iter*L2N*stride_AD*T_size))
                        pX_temp = PtrMatrix{$N_remain,$P_remain,$T,$stride_X,$(stride_X*P_remain),false}(pX + $(N_iter*L2N*T_size + P_iter*L2P*stride_X*T_size))
                        $(prefetch_ ? quote
                            prefetch(pA_temp, Val(0))
                            prefetch(pX_temp, Val(0))
                        end : nothing )
                        gemm!(pD_temp,
                         pA_temp,
                         pX_temp)
                    end )
                end) # $(M_remain == 0 ? nothing
            end) # end quote
    end # if P_remain > 0
    q
end


@generated function Base.:*(A::LinearAlgebra.Diagonal{T,<:AbstractFixedSizePaddedVector{M,T,P}}, B::AbstractFixedSizePaddedMatrix{M,N,T,P}) where {M,N,T,P}
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
    reps = P >> Wshiftp
    V = Vec{W,T}
    q = quote
        C = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)
        va = VectorizationBase.vectorizable(A.diag)
        vB = VectorizationBase.vectorizable(B)
        vC = VectorizationBase.vectorizable(C)
        Base.Cartesian.@nexprs $reps r -> va_r = SIMDPirates.vload($V, va + $W*(r-1))
        for n ∈ 0:$(N-1)
            Base.Cartesian.@nexprs $reps r -> begin
                prod_r = SIMDPirates.vmul(va_r, SIMDPirates.vload($V, vB + $P*n + $W*(r-1) ))
                SIMDPirates.vstore!(vC + $P*n + $W*(r-1), prod_r)
            end
        end
        #C
    end
    if B <: AbstractConstantFixedSizePaddedMatrix
        push!(q.args, :(ConstantFixedSizePaddedMatrix(C)))
    else
        push!(q.args, :(C))
    end
    q
end
@generated function Base.:*(
    sp::StackPointer,
    A::LinearAlgebra.Diagonal{T,<:AbstractFixedSizePaddedVector{M,T,P}},
    B::AbstractFixedSizePaddedMatrix{M,N,T,P}
) where {M,N,T,P}
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
    reps = P >> Wshiftp
    V = Vec{W,T}
    q = quote
        C = PtrMatrix{$M,$N,$T,$P}(pointer(sp,$T))
        va = VectorizationBase.vectorizable(A.diag)
        vB = VectorizationBase.vectorizable(B)
        vC = VectorizationBase.vectorizable(C)
        Base.Cartesian.@nexprs $reps r -> va_r = SIMDPirates.vload($V, va + $W*(r-1))
        for n ∈ 0:$(N-1)
            Base.Cartesian.@nexprs $reps r -> begin
                prod_r = SIMDPirates.vmul(va_r, SIMDPirates.vload($V, vB + $P*n + $W*(r-1) ))
                SIMDPirates.vstore!(vC + $P*n + $W*(r-1), prod_r)
            end
        end
        sp + $(sizeof(T)*P*N), C
    end
    q
end

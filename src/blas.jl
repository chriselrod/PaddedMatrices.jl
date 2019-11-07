

@generated function Base.:*(
    A::AbstractConstantFixedSizeMatrix{M,N,T,R1,L1},
    B::AbstractConstantFixedSizeMatrix{N,P,T,R2,L2}
) where {M,N,P,T,R1,R2,L2,L1}
    q = static_mul_quote(M,N,P,T,R1,R2)
    push!(q.args,  :(ConstantFixedSizeMatrix{$M,$P,$T,$R1,$(R1*P)}( output_data )) )
    q
end
@generated function Base.:*(
    A::AbstractConstantFixedSizeMatrix{M,N,T,R1,L1},
    B::AbstractFixedSizeVector{N,T,R2}
) where {M,N,T,R1,R2,L1}
    q = static_mul_quote(M,N,1,T,R1,R2)
    push!(q.args,  :(ConstantFixedSizeVector{$M,$T,$R1}( output_data )) )
    q
end
@generated function Base.:*(
    A::AbstractConstantFixedSizeMatrix{M,N,T,R1,L1},
    B::LinearAlgebra.Adjoint{T,<:AbstractConstantFixedSizeMatrix{P,N,T,R2,L2}}
) where {M,N,P,T,R1,R2,L1,L2}
# ) where {M,N,P,T,R1,R2,L2,L1}
    q = static_mul_nt_quote(M,N,P,T,R1,R2)
    # push!(q.args,  :(ConstantFixedSizeMatrix( out )) )
    push!(q.args,  :(ConstantFixedSizeMatrix{$M,$P,$T,$R1,$(R1*P)}( output_data )) )
    q
end
@generated function Base.:*(
    A::AbstractConstantFixedSizeVector{M,T,L1},
    B::LinearAlgebra.Adjoint{T,<:AbstractConstantFixedSizeVector{P,T,L2}}
) where {M,P,T,L1,L2}
# ) where {M,N,P,T,R1,R2,L2,L1}
    q = static_mul_nt_quote(M,1,P,T,L1,1)
    # push!(q.args,  :(ConstantFixedSizeMatrix( out )) )
    push!(q.args,  :(ConstantFixedSizeMatrix{$M,$P,$T,$L1,$(L1*P)}( output_data )) )
    q
end

@generated function Base.:*(
    A::AbstractConstantFixedSizeMatrix{M,N,T,R1},
    B::AbstractConstantFixedSizeVector{N,T,R2}
) where {M,N,T,R1,R2}
    q = static_mul_quote(M,N,1,T,R1,R2)
    push!(q.args,  :(ConstantFixedSizeVector{$M,$T,$R1,$R1}( output_data )) )
    q
end

@generated function Base.:*(
    A::AbstractMutableFixedSizeMatrix{M,N,T,ADR},
    X::AbstractMutableFixedSizeMatrix{N,P,T,XR}
) where {M,N,P,T,ADR,XR}
    q = quote
        $(Expr(:meta,:inline))
        D = FixedSizeMatrix{$M,$P,$T,$ADR,$(ADR*P)}(undef)
        pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        $(mulquote(ADR,N,P,ADR,XR,T))
        # ConstantFixedSizeMatrix(D)
    end
    # if M * P > 224
    if M * P > 64
        push!(q.args, :D)
    else
        push!(q.args, :(ConstantFixedSizeMatrix(D)))
    end
    q
end
@generated function Base.:*(
    sp::StackPointer,
    A::AbstractMutableFixedSizeMatrix{M,N,T,ADR},
    X::AbstractMutableFixedSizeMatrix{N,P,T,XR}
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


@generated function LinearAlgebra.mul!(
    D::AbstractMutableFixedSizeMatrix{M,P,T,DR},
    A::AbstractMutableFixedSizeMatrix{M,N,T,AR},
    X::AbstractMutableFixedSizeMatrix{N,P,T,XR}
) where {M,N,P,T,AR,XR,DR}
# ) where {M,P,N,T,AR,XR,DR}
    quote
        $(Expr(:meta,:inline))
        pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        $(evaluate_integers(mulquote(AR,N,P,AR,XR,T,:initkernel!,nothing,DR)))
        D
    end
end


@generated function LinearAlgebra.mul!(
    D::AbstractMutableFixedSizeVector{M,T,DR},
    A::AbstractMutableFixedSizeMatrix{M,N,T,AR},
    X::AbstractMutableFixedSizeVector{N,T,XR}
) where {M,N,T,AR,XR,DR}
    quote
        $(Expr(:meta,:inline))
        pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        $(mulquote(AR,N,1,AR,XR,T,:initkernel!,nothing,DR))
        D
    end
end
@generated function LinearAlgebra.mul!(
    D::PtrMatrix{M,P,T,DR,LD,VD},
    A::PtrMatrix{M,N,T,AR,LA},
    X::PtrMatrix{N,P,T,XR}
) where {M,N,P,T,AR,XR,DR,LA,LD,VD}
    Meffective = VD ? M : DR
    quote
        $(Expr(:meta,:inline))
        pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        $(mulquote(Meffective,N,P,AR,XR,T,:initkernel!,nothing,DR))
        D
    end
end
@generated function LinearAlgebra.mul!(
    D::PtrVector{M,T,DR,VD},
    A::PtrMatrix{M,N,T,AR,LA},
    X::PtrVector{N,T,XR}
) where {M,N,T,AR,XR,DR,LA,VD}
    Meffective = VD ? M : DR
    quote
        $(Expr(:meta,:inline))
        pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        $(mulquote(Meffective,N,1,AR,XR,T,:initkernel!,nothing,DR))
        D
    end
end

@generated function gemm!(
    D::AbstractMutableFixedSizeMatrix{M,P,T,DR},
    A::AbstractMutableFixedSizeMatrix{M,N,T,AR},
    X::AbstractMutableFixedSizeMatrix{N,P,T,XR}
) where {M,P,N,T,DR,AR,XR}
    quote
        $(Expr(:meta,:inline))
        pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        $(mulquote(M,N,P,AR,XR,T,:kernel!,nothing,DR))
        D
    end
end


function elementwise_product_quote(N,T,P)
    quote
        $(Expr(:meta,:inline)) # do we really want to force inline this?
        mv = FixedSizeVector{$N,$T,$P}(undef)
        @vvectorize $T for i ∈ 1:$P
            mv[i] = A[i] * B[i]
        end
    end
end

@generated function Base.:*(Adiagonal::Diagonal{T,<:AbstractFixedSizeVector{N,T,P}}, B::AbstractFixedSizeVector{N,T,P}) where {N,T,P}
    quote
        A = Adiagonal.diag
        $(elementwise_product_quote(N,T,P))
        ConstantFixedSizeArray(mv)
    end
end
@generated function Base.:*(
            Aadjoint::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeVector{N,T,P}},
            Bdiagonal::Diagonal{T,<:AbstractFixedSizeVector{N,T,P}}
        ) where {N,T,P}
    quote
        A = Aadjoint.parent
        B = Bdiagonal.diag
        $(elementwise_product_quote(N,T,P))
        ConstantFixedSizeArray(mv)'
    end
end
@generated function Base.:*(
            Adiagonal::Diagonal{T,<:AbstractMutableFixedSizeVector{N,T,P}},
            B::AbstractMutableFixedSizeVector{N,T,P}) where {N,T,P}
    quote
        A = Adiagonal.diag
        $(elementwise_product_quote(N,T,P))
        mv
    end
end
@generated function Base.:*(
    Aadjoint::LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizeVector{N,T,P1}},
    Bdiagonal::Diagonal{T,<:AbstractMutableFixedSizeVector{N,T,P2}}
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
    Adiagonal::Diagonal{T,<:AbstractMutableFixedSizeVector{N,T,P1}},
    B::AbstractMutableFixedSizeVector{N,T,P2}
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
    Aadjoint::LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizeVector{N,T,P1}},
    Bdiagonal::Diagonal{T,<:AbstractMutableFixedSizeVector{N,T,P2}}
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
    A::AbstractFixedSizeArray{S,T,N,X,L},
    B::AbstractFixedSizeArray{S,T,N,X,L}
) where {S,T<:Number,N,X,L}
    quote
        mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] + B[i]
        end
        ConstantFixedSizeArray(mv)
    end
end
@generated function Base.:+(A::AbstractMutableFixedSizeArray{S,T,N,X,L}, B::AbstractMutableFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
    if isview(A) | isview(B) | (first(X.parameters)::Int > 1)
        return quote
            $(Expr(:meta,:inline))
            mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
            @inbounds for l ∈ 1:$L
                mv[l] = A[l] + B[l]
            end
            mv
        end
    end
    quote
        $(Expr(:meta,:inline))
        mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
        @vvectorize $T for l ∈ 1:$L
            mv[l] = A[l] + B[l]
        end
        mv
    end
end
@generated function Base.:+(
    sp::StackPointer,
    A::AbstractFixedSizeArray{S,T,N,X,LA},
    B::AbstractFixedSizeArray{S,T,N,X,LB}
) where {S,T<:Number,N,X,LA,LB}
    L = min(LA,LB)
    @assert first(X.parameters)::Int == 1
    quote
        mv = PtrArray{$S,$T,$N,$X,$L}(pointer(sp,$T))
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] + B[i]
        end
        sp + $(VectorizationBase.align(sizeof(T)*L)), mv
    end
end
@generated function Base.:+(
    sp::StackPointer,
    A::AbstractFixedSizeVector{N,T,LA},
    B::AbstractFixedSizeVector{N,T,LB}
) where {N,T<:Number,LA,LB}
    L = min(LA,LB)
    quote
        mv = PtrVector{$N,$T,$L}(pointer(sp,$T))
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] + B[i]
        end
        sp + $(VectorizationBase.align(sizeof(T)*L)), mv
    end
end
@inline function Base.:+(A::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,P,L}},
                        B::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,P,L}}) where {S,T,N,P,L}
    (A' + B')'
end
@inline function Base.:+(
    sp::StackPointer,
    A::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,XA,LA}},
    B::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,XB,LB}}
) where {S,T,N,XA,XB,LA,LB}
    sp2, C = (+(sp, A', B'))
    sp2, C'
end
@generated function Base.:+(a::T, B::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
    quote
        # $(Expr(:meta,:inline))
        mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = a + B[i]
        end
        ConstantFixedSizeArray(mv)
    end
end
@generated function Base.:+(A::AbstractFixedSizeArray{S,T,N,X,L}, b::T) where {S,T<:Number,N,X,L}
    quote
        # $(Expr(:meta,:inline))
        mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] + b
        end
        ConstantFixedSizeArray(mv)
    end
end
@generated function Base.:+(a::T, Badj::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,X,L}}) where {S,T<:Number,N,X,L}
    quote
        $(Expr(:meta,:inline))
        mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = a + B[i]
        end
        ConstantFixedSizeArray(mv)'
    end
end
@generated function Base.:+(Aadj::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,X,L}}, b::T) where {S,T<:Number,N,X,L}
    quote
        $(Expr(:meta,:inline))
        mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] + b
        end
        ConstantFixedSizeArray(mv)'
    end
end
@generated function Base.:-(A::AbstractFixedSizeArray{S,T,N,X,L}, B::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
    quote
        # $(Expr(:meta,:inline))
        mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] - B[i]
        end
        ConstantFixedSizeArray(mv)
    end
end
@generated function diff!(C::AbstractMutableFixedSizeArray{S,T,N,X,L}, A::AbstractFixedSizeArray{S,T,N,X,L}, B::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
    quote
        @vvectorize $T for i ∈ 1:$L
            C[i] = A[i] - B[i]
        end
        C
    end
end
@generated function Base.:-(A::AbstractFixedSizeArray{S,T,N,X,L}, b::T) where {S,T<:Number,N,L,X}
    quote
        # $(Expr(:meta,:inline))
        mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] - b
        end
        ConstantFixedSizeArray(mv)
    end
end
@generated function Base.:-(a::T, B::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
    quote
        # $(Expr(:meta,:inline))
        mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = a - B[i]
        end
        ConstantFixedSizeArray(mv)
    end
end


#### This function problematically assumes padding.
@generated function Base.sum!(s::AbstractMutableFixedSizeVector{P,T,Ls}, A::AbstractMutableFixedSizeMatrix{P,N,T,LA}) where {P,T,Ls,N,LA}
    stride_bytes = LA*sizeof(T)
#    L = max(Ls, LA)
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
    WT = W * sizeof(T)
    V = Vec{W,T}
    # +2, to divide by an additional 4
    iterations = Ls >>> (Wshift + 2)
    r = Ls & ((W << 2) - 1)
    riter = r >>> Wshift
    remainder_quote = quote
            Base.Cartesian.@nexprs $riter j -> begin
            offset_j = $(4WT*iterations) + $WT*(j-1)
            x_j = SIMDPirates.vbroadcast($V, zero($T))
        end
        for n ∈ 0:$(N-1)
            offset_n = n * $stride_bytes
            Base.Cartesian.@nexprs $riter j -> begin
                x_j = SIMDPirates.vadd(x_j, SIMDPirates.vload($V, ptrA + offset_n + offset_j))
            end
        end
        Base.Cartesian.@nexprs $riter j -> SIMDPirates.vstore!(ptrs + offset_j, x_j)
    end
    quote
        # $(Expr(:meta,:inline))
        # x̄ = zero(PaddedMatrices.FixedSizeVector{P,T})
        # s = PaddedMatrices.FixedSizeVector{$P,$T}(undef)
        ptrs = pointer(s)
        ptrA = pointer(A)
        GC.@preserve s A begin
            for i ∈ 0:$(iterations-1)
                Base.Cartesian.@nexprs 4 j -> begin
                    offset_j = $(4WT)*i + $WT*(j-1)
                    x_j = SIMDPirates.vbroadcast($V, zero($T))
                end
                for n ∈ 0:$(N-1)
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
@generated function Base.sum!(s::AbstractMutableFixedSizeVector{P,T,L}, A::AbstractPaddedMatrix{T}) where {T,P,L}
    stride_bytes = L*sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
    WT = W * sizeof(T)
    V = Vec{W,T}
    # +2, to divide by an additional 4
    iterations = L >>> (Wshift + 2)
    r = L & ((W << 2) - 1)
    riter = r >>> Wshift
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
        # $(Expr(:meta,:inline))
        # x̄ = zero(PaddedMatrices.FixedSizeVector{P,T})
        # s = PaddedMatrices.FixedSizeVector{$P,$T}(undef)
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
@generated function negative_sum!(s::AbstractMutableFixedSizeVector{P,T,L}, A::AbstractPaddedMatrix{T}) where {T,P,L}
    stride_bytes = L*sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)

    WT = W * sizeof(T)
    V = Vec{W,T}

    # +2, to divide by an additional 4
    iterations = L >>> (Wshift + 2)
    r = L & ((W << 2) - 1)
    riter = r >>> Wshift
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
        # $(Expr(:meta,:inline))
        # x̄ = zero(PaddedMatrices.FixedSizeVector{P,T})
        # s = PaddedMatrices.FixedSizeVector{$P,$T}(undef)
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
# @generated function prodmuls(coefficients::NTuple{N,T}, As::NTuple{N,<:AbstractFixedSizeArray{S,T,N,P,L}}) where {S,T,N,P,L}


# end

@inline extract_λ(a) = a
@inline extract_λ(a::UniformScaling) = a.λ
@generated function Base.:*(A::AbstractFixedSizeArray{S,T,N,X,L}, bλ::Union{T,UniformScaling{T}}) where {S,T<:Real,N,X,L}
    quote
        mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
        b = extract_λ(bλ)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] * b
        end
        ConstantFixedSizeArray(mv)
    end
end
@generated function Base.:*(Aadj::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,X,L}}, bλ::Union{T,UniformScaling{T}}) where {S,T<:Real,N,X,L}
    quote
        $(Expr(:meta,:inlne))
        mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
        A = Aadj.parent
        b = extract_λ(bλ)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] * b
        end
        ConstantFixedSizeArray(mv)'
    end
end
@generated function Base.:*(
    sp::StackPointer,
    A::AbstractFixedSizeArray{S,T,N,X,L},
    bλ::Union{T,UniformScaling{T}}
) where {S,T<:Real,N,X,L}
    quote
        mv = PtrArray{$S,$T,$N,$X,$L}(pointer(sp,$T))
        b = extract_λ(bλ)
        @vvectorize for i ∈ 1:$L
            mv[i] = A[i] * b
        end
        sp + $(VectorizationBase.align(sizeof(T)*L)), mv
    end
end
@generated function Base.:*(
    sp::StackPointer,
    Aadj::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeArray{S,T,N,X,L}},
    bλ::Union{T,UniformScaling{T}}
) where {S,T<:Real,N,X,L}
    quote
        $(Expr(:meta,:inline))
        mv = PtrArray{$S,$T,$N,$X,$L}(pointer(sp,$T))
        A = Aadj.parent
        b = extract_λ(bλ)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = A[i] * b
        end
        sp + $(VectorizationBase.align(sizeof(T)*L)), mv'
    end
end
@generated function Base.:*(a::T, B::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
    quote
        mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = a * B[i]
        end
        ConstantFixedSizeArray(mv)
    end
end
@generated function SIMDPirates.vmuladd(a::T, x::AbstractFixedSizeArray{S,T,N,X,L}, y::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
    quote
        # $(Expr(:meta,:inline))
        mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = a * x[i] + y[i]
        end
        ConstantFixedSizeArray(mv)
    end
end
@generated function SIMDPirates.vmuladd(
    a::T,
    x::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeVector{N,T,L}},
    y::LinearAlgebra.Adjoint{T,<:AbstractFixedSizeVector{N,T,L}}
) where {N,T,L}
    quote
        $(Expr(:meta,:inline))
        mv = FixedSizeVector{$N,T,$L,$L}(undef)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = a * x[i] + y[i]
        end
        ConstantFixedSizeArray(mv)
    end
end

# @generated function SIMDPirates.vmuladd(a::T,
#                 x::Union{<:AbstractFixedSizeVector{P1,T,L1},LinearAlgebra.Adjoint{T,<:AbstractFixedSizeVector{P1,T,L1}}}, y::Union{<:AbstractFixedSizeVector{P2,T,L2},LinearAlgebra.Adjoint{T,<:AbstractFixedSizeVector{P2,T,L2}}}
#                 ) where {T,P1,L1,P2,L2}
#     if x <: LinearAlgebra.Adjoint
#         @assert y <: LinearAlgebra.Adjoint
#     else
#         @assert !( y <: LinearAlgebra.Adjoint )
#     end
#     if L1 < L2
#         return quote
#             mv = FixedSizeVector{$P2,$T,$L2}(undef)
#             @fastmath @inbounds @simd ivdep for i ∈ 1:$L1
#                 mv[i] = a * x[i] + y[i]
#             end
#             @inbounds @simd ivdep for i ∈ $(L1+1):$L2
#                 mv[i] = y[i]
#             end
#             $( x <: LinearAlgebra.Adjoint ? :(ConstantFixedSizeArray(mv)') : :(ConstantFixedSizeArray(mv)) )
#         end
#     elseif L1 > L2
#         return quote
#             mv = FixedSizeVector{$P1,$T,$L1}(undef)
#             @fastmath @inbounds @simd ivdep for i ∈ 1:$L2
#                 mv[i] = a * x[i] + y[i]
#             end
#             @inbounds @simd ivdep for i ∈ $(L2+1):$L1
#                 mv[i] = a * x[i]
#             end
#             $( x <: LinearAlgebra.Adjoint ? :(ConstantFixedSizeArray(mv)') : :(ConstantFixedSizeArray(mv)) )
#         end
#     else
#         return quote
#             mv = FixedSizeVector{$P1,$T,$L1}(undef)
#             @fastmath @inbounds @simd ivdep for i ∈ 1:$L1
#                 mv[i] = a * x[i] + y[i]
#             end
#             $( x <: LinearAlgebra.Adjoint ? :(ConstantFixedSizeArray(mv)') : :(ConstantFixedSizeArray(mv)) )
#         end
#     end
# end
@generated function SIMDPirates.vfnmadd(a::T, x::AbstractFixedSizeArray{S,T,N,X,L}, y::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T<:Number,N,X,L}
    quote
        # $(Expr(:meta,:inline))
        mv = FixedSizeArray{$S,$T,$N,$X,$L}(undef)
        @vvectorize $T for i ∈ 1:$L
            mv[i] = y[i] - a * x[i]
        end
        ConstantFixedSizeArray(mv)
    end
end

@generated function LinearAlgebra.dot(A::AbstractFixedSizeArray{S,T,N,X,L}, B::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    if N > 1
        ST = tuple(S.parameters...)
        P = (X.parameters[2])::Int
        return quote
            out = zero(T)
            ind = 0
            @nloops $(N-1) i j -> 1:$ST[j+1] begin
                @vectorize $T for i_0 ∈ 1:$(ST[1])
                    out += A[ind + i_0] * B[ind + i_0]
                end
                ind += $P
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
@generated function dot_self(A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    nrows = first(S.parameters)::Int
    if N > 1
        P = (X.parameters[2])::Int
        return quote
            out = zero(T)
            ind = 0
            Base.Cartesian.@nloops $(N-1) i j -> 1:size(A,j+1) begin
                @vvectorize $T for i_0 ∈ 1:$nrows
                    Aᵢ = A[ind + i_0]
                    out += Aᵢ * Aᵢ
                end
                ind += $P
            end
            out
        end
    else #if N == 1
        return quote
            out = zero(T)
            @vectorize $T  for i ∈ 1:$nrows
                out += A[i] * A[i]
            end
            out
        end
    end
end

# @generated function gemm!(D::PtrMatrix{M,P,T,ADR},
#                             A::PtrMatrix{M,N,T,ADR},
#                             X::PtrMatrix{N,P,T,XR},
#                             prefetchAX = nothing) where {M,N,P,T,ADR,XR}
#
#     mulquote(M, N, P, ADR, XR, T, :kernel!, prefetchAX)
# end

function mulquote(
    D::AbstractMutableFixedSizeMatrix{M,P,T,DR},
    A::AbstractMutableFixedSizeMatrix{M,N,T,AR},
    X::AbstractMutableFixedSizeMatrix{N,P,T,XR},
    init = :initkernel!
) where {M,N,P,T,AR,XR,DR}
    quote
        $(Expr(:meta,:inline))
        pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        $(mulquote(M,N,P,AR,XR,T,init,nothing,DR))
    end
end
function mulquote(M,N,P,AR,XR,T,init=:initkernel!,prefetchAX=nothing,DR=AR)
    (L1S, L2S, L3S), num = blocking_structure(M, N, P, T)
    if num == 0 || (M*N*P < 64^3)#104^3)
        return cache_mulquote(M,N,P,AR,XR,L1S,T,init,DR)
    elseif T <: LinearAlgebra.BlasFloat
        :(BLAS.gemm!('N','N',one($T),A,X,$(init==:initkernel! ? zero(T) : one(T)),D))
    elseif num == 1
        return cache_mulquote(M,N,P,AR,XR,L1S,T,init,DR)
    else#if num == 2
        return cache_mulquote(M,N,P,AR,XR,L1S,L2S,T,init)#,true)#prefetchAX)
    end
end

function block_loop_quote_minbandwidth(
#function block_loop_quote(
    L1M::Int,L1N::Int,L1P::Int,stride_A::Int,stride_X::Int,M_iter::Int,M_remain::Int,P_iter::Int,P_remain::Int,
    T_size::Int,kernel::Symbol=:kernel!,pA::Symbol=:pAₙ,pX::Symbol=:pXₙ,pD::Symbol=:pD,X_transposed::Bool=false,prefetch::Bool=false,
    stride_D::Int = stride_A,negative::Bool=false
)
    if M_remain == 0
        D = :($pD + $(T_size*L1M)*mᵢ + $(T_size*L1P*stride_D)*pᵢ)
        A = :($pA + $(T_size*L1M)*mᵢ)
        X = X_transposed ? pX : :($pX + $(T_size*L1P*stride_X)*pᵢ)
        if P_iter > M_iter + 1 # Excess of 1 is okay.
            PM_ratio, PM_remainder = divrem(P_iter, M_iter)
            q = quote
                    for pmᵣ ∈ 1:$PM_ratio, pᵢ ∈ (pmᵣ-1)*$M_iter:$M_iter*pmᵣ - 1
                        for mᵢ ∈ $M_iter*pmᵣ - pᵢ:$(M_iter-1)
                            $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                        end
                        for mᵢ ∈ 0:$M_iter*pmᵣ - pᵢ - 1
                            $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                        end
                    end
                    for pᵢ ∈ $(M_iter*PM_ratio):$(P_iter-1)
                        for mᵢ ∈ $(M_iter*(PM_ratio+1))-pᵢ:$(M_iter-1)
                            $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                        end
                        for mᵢ ∈ 0:$(M_iter*(PM_ratio+1)-1)-pᵢ
                            $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                        end
                    end
            end
            MP_terminal = PM_remainder == 0 ? 0 : M_iter - PM_remainder # ==  (M_iter*(PM_ratio+1)) - P_iter
        else
            q = quote
                # $prefetch_quote
                for pᵢ ∈ 0:$(P_iter-1)
                    for mᵢ ∈ $(M_iter)-pᵢ:$(M_iter-1)
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                    end
                    for mᵢ ∈ 0:$(M_iter-1)-pᵢ
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                    end
                end # for pᵢ ∈ 0:$(P_iter-1)
            end # quote
            MP_terminal = P_iter == M_iter + 1 ? M_iter - 1 : M_iter - P_iter
        end
    else
        ### Here
        D = :($pD + $(T_size*L1M)*mᵢ + $(T_size*L1P*stride_D)*pᵢ)
        A = :($pA + $(T_size*L1M)*mᵢ)
        X = X_transposed ? pX : :($pX + $(T_size*L1P*stride_X)*pᵢ)
        D_r = :($pD + $(T_size*L1M*M_iter) + $(T_size*L1P*stride_D)*pᵢ)
        A_r = :($pA + $(T_size*L1M*M_iter))
        if P_iter > M_iter + 2
            # Here, we insert a kernel call to "M_remain" that is of an abridged size.
            # period for rotation over M is one longer.
            PM_ratio, PM_remainder = divrem(P_iter, M_iter + 1)
            q = quote
                # $prefetch_quote
                for mᵢ ∈ 0:$(M_iter-1)
                    $(kernel)($pD + $(T_size*L1M)*mᵢ, $A, $pX, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                end
                $(kernel)($pD + $(T_size*L1M*M_iter), $A_r, $pX, Kernel{$M_remain,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                for pᵢ ∈ 1:$(M_iter+1)
                    for mᵢ ∈ $(M_iter+1) - pᵢ:$(M_iter-1)
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                    end
                    $(kernel)($D_r, $A_r, $X, Kernel{$M_remain,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                    for mᵢ ∈ 0:$M_iter - pᵢ
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                    end
                end
                for pmᵣ ∈ 2:$(PM_ratio+1), pᵢ ∈ (pmᵣ-1)*$(M_iter+1)+1:min($(M_iter+1)*pmᵣ,$(P_iter-1))
                    for mᵢ ∈ $(M_iter+1)*pmᵣ - pᵢ:$(M_iter-1)
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                    end
                    $(kernel)($D_r, $A_r, $X, Kernel{$M_remain,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                    for mᵢ ∈ 0:$(M_iter+1)*pmᵣ - pᵢ - 1
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                    end
                end
            end
            MP_terminal = PM_remainder == 0 ? M_iter : (M_iter+1) - PM_remainder

        else
            q = quote
                for mᵢ ∈ 0:$(M_iter-1)
                    $(kernel)($pD + $(T_size*L1M)*mᵢ, $A, $pX, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                end
                $(kernel)($pD + $(T_size*L1M*M_iter), $A_r, $pX, Kernel{$M_remain,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                for pᵢ ∈ 1:$(P_iter-1)
                    for mᵢ ∈ $(M_iter+1)-pᵢ:$(M_iter-1)
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                    end
                    $(kernel)($D_r, $A_r, $X, Kernel{$M_remain,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                    for mᵢ ∈ 0:$M_iter-pᵢ
                        $(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                    end
                end
            end
            MP_terminal = P_iter == M_iter + 2 ? M_iter : M_iter + 1 - P_iter
        end
    end

    if P_remain != 0
        if M_remain == 0
            D = :($pD + $(T_size*L1M)*mᵢ + $(T_size*L1P*stride_D*P_iter))
            A = :($pA + $(T_size*L1M)*mᵢ)
            X = X_transposed ? pX : :($pX + $(T_size*L1P*stride_X*P_iter))
            push!(q.args,
            quote
                for mᵢ ∈ $(MP_terminal):$(M_iter-1)
                    $(kernel)($D, $A, $X, Kernel{$L1M,$P_remain,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                end
                for mᵢ ∈ 0:$(MP_terminal-1)
                    $(kernel)($D, $A, $X, Kernel{$L1M,$P_remain,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                end
            end
            )
        else
            D = :($pD + $(T_size*L1M)*mᵢ + $(T_size*L1P*stride_D*P_iter))
            A = :($pA + $(T_size*L1M)*mᵢ)
            X = X_transposed ? pX : :($pX + $(T_size*L1P*stride_X*P_iter))
            D_r = :($pD + $(T_size*L1M*M_iter) + $(T_size*L1P*stride_D*P_iter))
            A_r = :($pA + $(T_size*L1M*M_iter))
            push!(q.args,
            quote
                for mᵢ ∈ $(MP_terminal):$(M_iter-1)
                    $(kernel)($D, $A, $X, Kernel{$L1M,$P_remain,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                end
                $(kernel)($D_r, $A_r, $X, Kernel{$M_remain,$P_remain,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                for mᵢ ∈ 0:$(MP_terminal-1)
                    $(kernel)($D, $A, $X, Kernel{$L1M,$P_remain,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}())
                end
            end
            )
        end

    end
    q
end

#function block_loop_quote_test_simple_prefetch(
function block_loop_quote(
    L1M::Int,L1N::Int,L1P::Int,stride_A::Int,stride_X::Int,M_iter::Int,M_remain::Int,P_iter::Int,P_remain::Int,
    T_size::Int,kernel::Symbol=:kernel!,pA::Symbol=:pAₙ,pX::Symbol=:pXₙ,pD::Symbol=:pD,X_transposed::Bool=false,prefetch::Bool = false,
    stride_D::Int = stride_A, negative::Bool = false
)
    #    prefetch = true
    #    @show T_size, L1P, stride_D
    prefetch = false
    # kernel = Symbol(:inline_, kernel)
    D = :($pD + $(T_size*L1M)*mᵢ + $(T_size*L1P*stride_D)*pᵢ)
    A = :($pA + $(T_size*L1M)*mᵢ)
    X = X_transposed ? pX : :($pX + $(T_size*L1P*stride_X)*pᵢ)
    if prefetch && M_remain == 0
        M_remain = L1M
        M_iter -= 1
    end
    # Mₖ,Pₖ,stride_A,stride_X,stride_D,N
    kernel_call = if prefetch
        :($(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$(PrefetchA(L1M))}()))
    else
        :($(kernel)($D, $A, $X, Kernel{$L1M,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}()))
    end
    kernel_call_Mremain = if prefetch
        :($(kernel)($D, $A, $X, Kernel{$M_remain,$L1P,$stride_A,$stride_X,$stride_D,$L1N}()),Val{$(PrefetchAX(-L1M*M_iter,L1P*stride_X))}())
    else
        :($(kernel)($D, $A, $X, Kernel{$M_remain,$L1P,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$negative}()))
    end
    ploopbody = if M_iter isa Integer
        if M_iter == 1
            quote
                mᵢ = 0
                $kernel_call
            end
        else
            quote
                for mᵢ ∈ 0:$(M_iter - 1)
                    $kernel_call
                end
            end
        end
    else
        quote
            for mᵢ ∈ 0:$M_iter - 1
                $kernel_call
            end
        end
    end
    if M_remain > 0
        remain_expr = quote
            mᵢ = $M_iter
            $kernel_call_Mremain
        end
        push!(ploopbody.args, remain_expr)
    end

    q = if P_iter isa Integer
        if P_iter == 1
            quote
                pᵢ = 0
                $ploopbody
            end
        else
            quote
                for pᵢ ∈ 0:$(P_iter - 1)
                    $ploopbody
                end
            end
        end
    else
        quote
            for pᵢ ∈ 0:$P_iter - 1
                $ploopbody
            end
        end
    end
    if P_remain > 0
        kernel_call = if prefetch
            :($(kernel)($D, $A, $X, Kernel{$L1M,$P_remain,$stride_A,$stride_X,$stride_D,$L1N}(), Val{$(PrefetchA(L1M))}()))
        else
            :($(kernel)($D, $A, $X, Kernel{$L1M,$P_remain,$stride_A,$stride_X,$stride_D,$L1N}()))
        end
        kernel_call_Mremain = :($(kernel)($D, $A, $X, Kernel{$M_remain,$P_remain,$stride_A,$stride_X,$stride_D,$L1N}()))
        p_remain_quote = if M_iter isa Integer
            if M_iter == 1
                quote
                    pᵢ = $P_iter
                    mᵢ = 0
                    $kernel_call
                end
            else
                quote
                    pᵢ = $P_iter
                    for mᵢ ∈ 0:$(M_iter - 1)
                        $kernel_call
                    end
                end
            end
        else
            quote
                pᵢ = $P_iter
                for mᵢ ∈ 0:$M_iter - 1
                    $kernel_call
                end
            end
        end
        if M_remain > 0
            push!(p_remain_quote.args, :(mᵢ = $M_iter))
            push!(p_remain_quote.args, kernel_call_Mremain)
        end
        push!(q.args, p_remain_quote)
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
        kernel = DynamicKernel(
            M, P, N, stride_D, stride_A, stride_X, T, X_transposed = false
        )
        return kernel_quote(kernel, init = true, force_inline = true, mask_ops = true, runtime_mask = false)
    end
    
    q = quote
        pD, pA, pX = pointer(D), pointer(A), pointer(X)

        $(block_loop_quote(L1M,L1N,L1P,stride_A,stride_X,M_iter,M_remain,P_iter,P_remain,T_size,init,:pA,:pX,:pD,false,false,stride_D))
    end

    AN_stride = stride_A * L1N * T_size
    XN_stride = stride_X  * L1N * T_size

    if N_iter > 2
        push!(q.args,
        quote
            for n ∈ 1:$(N_iter-1)
                pAₙ = pA + n*$(L1N * T_size * stride_A)
                pXₙ = pX + n*$(L1N * T_size)
                $(block_loop_quote(L1M,L1N,L1P,stride_A,stride_X,M_iter,M_remain,P_iter,P_remain,T_size,primary,:pAₙ,:pXₙ,:pD,false,false,stride_D))
            end
        end
        )

    elseif N_iter == 2
        push!(q.args,
        quote
            pAₙ = pA + $(L1N * T_size * stride_A)
            pXₙ = pX + $(L1N * T_size)
            $(block_loop_quote(L1M,L1N,L1P,stride_A,stride_X,M_iter,M_remain,P_iter,P_remain,T_size,primary,:pAₙ,:pXₙ,:pD,false,false,stride_D))
        end
        )
    end
    if N_remain > 0 # we need two goes
        push!(q.args,
        quote
            pAₙ = pA + $(L1N*N_iter * T_size * stride_A)
            pXₙ = pX + $(L1N*N_iter * T_size)
            $(block_loop_quote(L1M,N_remain,L1P,stride_A,stride_X,M_iter,M_remain,P_iter,P_remain,T_size,primary,:pAₙ,:pXₙ,:pD,false,false,stride_D))
        end
        )
    end
    q
end



function cache_mulquote(M,N,P,stride_A,stride_X,(L1M,L1N,L1P),(L2M,L2N,L2P),::Type{T}, init = :initkernel!, prefetchAX = nothing, stride_D = stride_A) where T
    M_iter, M_remain = divrem(M, L2M)
    N_iter, N_remain = divrem(N, L2N)
    P_iter, P_remain = divrem(P, L2P)
    T_size = sizeof(T)
    prefetch_ = prefetchAX != nothing
    initmul = ifelse(init == :initkernel!, :mul!, :gemm!)

    q = quote
        pD, pA, pX = pointer(D), pointer(A), pointer(X)


        for pᵢ ∈ 0:$(P_iter-1)
            pd_off = pᵢ*$(L2P*stride_D*T_size)
            px_off = pᵢ*$(L2P*stride_X*T_size)
            for mᵢ ∈ 0:$(M_iter-1)
                m_off = mᵢ*$(L2M*T_size)
                pD_temp1 = PtrMatrix{$L2M,$L2P,$T,$stride_D,$(stride_D*L2P),false}(pD + m_off + pd_off)
                pA_temp1 = PtrMatrix{$L2M,$L2N,$T,$stride_A,$(stride_A*L2N),false}(pA + m_off)
                pX_temp1 = PtrMatrix{$L2N,$L2P,$T,$stride_X,$(stride_X*L2P),false}(pX + px_off)
                $(prefetch_ ? quote
                    prefetch(pD_temp1, Val(1))
                    prefetch(pA_temp1, Val(0))
                    prefetch(pX_temp1, Val(0))
                end : nothing )
                $initmul(pD_temp1,
                     pA_temp1,
                     pX_temp1)
                for nᵢ ∈ 1:$(N_iter-1)
                    pA_temp1 = PtrMatrix{$L2M,$L2N,$T,$stride_A,$(stride_A*L2N),false}(pA + m_off + nᵢ*$(L2N*stride_A*T_size))
                    pX_temp1 = PtrMatrix{$L2N,$L2P,$T,$stride_X,$(stride_X*L2P),false}(pX + nᵢ*$(L2N*T_size) + px_off)
                    $(prefetch_ ? quote
                        prefetch(pA_temp1, Val(0))
                        prefetch(pX_temp1, Val(0))
                    end : nothing )
                    gemm!(pD_temp1,
                         pA_temp1,
                         pX_temp1)
                end
                $(N_remain == 0 ? nothing : quote
                    pA_temp2 = PtrMatrix{$L2M,$N_remain,$T,$stride_A,$(stride_A*N_remain),false}(pA + m_off + $(N_iter*L2N*stride_A*T_size))
                    pX_temp2 = PtrMatrix{$N_remain,$L2P,$T,$stride_X,$(stride_X*L2P),false}(pX + $(N_iter*L2N*T_size) + px_off)
                    $(prefetch_ ? quote
                        prefetch(pA_temp2, Val(0))
                        prefetch(pX_temp2, Val(0))
                    end : nothing )
                    gemm!(pD_temp1, pA_temp2, pX_temp2)
                end)
            end
            ### Check if we need to add an expression for a remainder of M.
            $(M_remain == 0 ? nothing : quote
                pD_temp3 = PtrMatrix{$M_remain,$L2P,$T,$stride_D,$(stride_D*L2P),false}(pD + $(M_iter*L2M*T_size) + pd_off)
                pA_temp3 = PtrMatrix{$M_remain,$L2N,$T,$stride_A,$(stride_A*L2N),false}(pA + $(M_iter*L2M*T_size))
                pX_temp3 = PtrMatrix{$L2N,$L2P,$T,$stride_X,$(stride_X*L2P),false}(pX + px_off)
                $(prefetch_ ? quote
                    prefetch(pD_temp3, Val(1))
                    prefetch(pA_temp3, Val(0))
                    prefetch(pX_temp3, Val(0))
                end : nothing )
                $initmul(pD_temp3,
                     pA_temp3,
                     pX_temp3)
                for nᵢ ∈ 1:$(N_iter-1)
                    pA_temp3 = PtrMatrix{$M_remain,$L2N,$T,$stride_A,$(stride_A*L2N),false}(pA + $(M_iter*L2M*T_size) + nᵢ*$(L2N*stride_A*T_size))
                    pX_temp3 = PtrMatrix{$L2N,$L2P,$T,$stride_X,$(stride_X*L2P),false}(pX + nᵢ*$(L2N*T_size) + px_off)
                    $(prefetch_ ? quote
                        prefetch(pA_temp3, Val(0))
                        prefetch(pX_temp3, Val(0))
                    end : nothing )
                    gemm!(pD_temp3,
                         pA_temp3,
                         pX_temp3)
                end
                $(N_remain == 0 ? nothing : quote
                    pA_temp4 = PtrMatrix{$M_remain,$N_remain,$T,$stride_A,$(stride_A*N_remain),false}(pA + $(M_iter*L2M*T_size + N_iter*L2N*stride_A*T_size))
                    pX_temp4 = PtrMatrix{$N_remain,$L2P,$T,$stride_X,$(stride_X*L2P),false}(pX + $(N_iter*L2N*T_size) + px_off)
                    $(prefetch_ ? quote
                        prefetch(pA_temp4, Val(0))
                        prefetch(pX_temp4, Val(0))
                    end : nothing )
                    gemm!(pD_temp3,
                     pA_temp4,
                     pX_temp4)
                end )
            end) # $(M_remain == 0 ? nothing
        end # for pᵢ ∈ 0:$(P_iter-1)
    end # quote

    if P_remain > 0
        push!(q.args,
            quote
                for mᵢ ∈ 0:$(M_iter-1)
                    m_off = mᵢ*$(L2M*T_size)
                    pD_temp5 = PtrMatrix{$L2M,$P_remain,$T,$stride_D,$(stride_D*P_remain),false}(pD + m_off + $(P_iter*L2P*stride_D*T_size))
                    pA_temp5 = PtrMatrix{$L2M,$L2N,$T,$stride_A,$(stride_A*L2N),false}(pA + m_off)
                    pX_temp5 = PtrMatrix{$L2N,$P_remain,$T,$stride_X,$(stride_X*P_remain),false}(pX + $(P_iter*L2P*stride_X*T_size))
                    $(prefetch_ ? quote
                        prefetch(pD_temp5, Val(1))
                        prefetch(pA_temp5, Val(0))
                        prefetch(pA_temp5, Val(0))
                    end : nothing )
                    $initmul(pD_temp5,
                         pA_temp5,
                         pX_temp5)
                    for nᵢ ∈ 1:$(N_iter-1)
                        pA_temp5 = PtrMatrix{$L2M,$L2N,$T,$stride_A,$(stride_A*L2N),false}(pA + m_off + nᵢ*$(L2N*stride_A*T_size))
                        pX_temp5 = PtrMatrix{$L2N,$P_remain,$T,$stride_X,$(stride_X*P_remain),false}(pX + nᵢ*$(L2N*T_size) + $(P_iter*L2P*stride_X*T_size))
                        $(prefetch_ ? quote
                            prefetch(pA_temp5, Val(0))
                            prefetch(pX_temp5, Val(0))
                        end : nothing )
                        gemm!(pD_temp5,
                             pA_temp5,
                             pX_temp5)
                    end
                    $(N_remain == 0 ? nothing : quote
                        pA_temp6 = PtrMatrix{$L2M,$N_remain,$T,$stride_A,$(stride_A*N_remain),false}(pA + m_off + $(N_iter*L2N*stride_A*T_size))
                        pX_temp6 = PtrMatrix{$N_remain,$P_remain,$T,$stride_X,$(stride_X*P_remain),false}(pX + $(N_iter*L2N*T_size + P_iter*L2P*stride_X*T_size))
                        $(prefetch_ ? quote
                            prefetch(pA_temp5, Val(0))
                            prefetch(pX_temp5, Val(0))
                        end : nothing )
                        gemm!(pD_temp5,
                         pA_temp6,
                         pX_temp6)
                    end )
                end
                ### Check if we need to add an expression for a remainder of M.
                $(M_remain == 0 ? nothing : quote
                    pD_temp7 = PtrMatrix{$M_remain,$P_remain,$T,$stride_D,$(stride_D*P_remain),false}(pD + $(M_iter*L2M*T_size + P_iter*L2P*stride_D*T_size))
                    pA_temp7 = PtrMatrix{$M_remain,$L2N,$T,$stride_A,$(stride_A*L2N),false}(pA + $(M_iter*L2M*T_size))
                    pX_temp7 = PtrMatrix{$L2N,$P_remain,$T,$stride_X,$(stride_X*P_remain),false}(pX + $(P_iter*L2P*stride_X*T_size))
                    $(prefetch_ ? quote
                        prefetch(pD_temp7, Val(1))
                        prefetch(pA_temp7, Val(0))
                        prefetch(pX_temp7, Val(0))
                    end : nothing )
                    $initmul(pD_temp7,
                         pA_temp7,
                         pX_temp7)
                    for nᵢ ∈ 1:$(N_iter-1)
                        pA_temp7 = PtrMatrix{$M_remain,$L2N,$T,$stride_A,$(stride_A*L2N),false}(pA + $(M_iter*L2M*T_size) + nᵢ*$(L2N*stride_A*T_size))
                        pX_temp7 = PtrMatrix{$L2N,$P_remain,$T,$stride_X,$(stride_X*P_remain),false}(pX + nᵢ*$(L2N*T_size) + $(P_iter*L2P*stride_X*T_size))
                        $(prefetch_ ? quote
                            prefetch(pA_temp7, Val(0))
                            prefetch(pX_temp7, Val(0))
                        end : nothing )
                        gemm!(pD_temp7,
                             pA_temp7,
                             pX_temp7)
                    end
                    $(N_remain == 0 ? nothing : quote
                        pA_temp8 = PtrMatrix{$M_remain,$N_remain,$T,$stride_A,$(stride_A*N_remain),false}(pA + $(M_iter*L2M*T_size + N_iter*L2N*stride_A*T_size))
                        pX_temp8 = PtrMatrix{$N_remain,$P_remain,$T,$stride_X,$(stride_X*P_remain),false}(pX + $(N_iter*L2N*T_size + P_iter*L2P*stride_X*T_size))
                        $(prefetch_ ? quote
                            prefetch(pA_temp8, Val(0))
                            prefetch(pX_temp8, Val(0))
                        end : nothing )
                        gemm!(pD_temp7,
                         pA_temp8,
                         pX_temp8)
                    end )
                end) # $(M_remain == 0 ? nothing
            end) # end quote
    end # if P_remain > 0
    q
end


function add_row_rem_expression!(q::Expr, row_rem, kernel::DynamicKernel; mask_ops::Bool = true, runtime_mask::Bool = false, init::Bool = true)
    @unpack R, C, N, stride_D, stride_A, stride_X, T, X_transposed, negative = kernel
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    size_T = sizeof(T)
    if row_rem isa Integer
        if row_rem > 0
            kernel = DynamicKernel(
                row_rem, C, N, stride_D, stride_A, stride_X, T, X_transposed = true, negative = negative
            )
            # only mask if there is no column remainder
            push!(q.args, kernel_quote(kernel, init = init, force_inline = false, mask_ops = mask_ops, runtime_mask = runtime_mask))
        end
    else
        row_rem_quote = quote initkernel_nt!(pD, pA, pX, DKernel{$R, $C}($N, $stride_D, $stride_A, $stride_X), mask, Val{$negative}()) end
        for r ∈ 1:(R >>> Wshift)
            new_nrows = R - (r << Wshift)
            row_rem_quote = quote
                if $row_rem > $new_nrows
                    $row_rem_quote
                else
                    $(new_nrows == 0 ? nothing : :(initkernel_nt!(pD, pA, pX, DKernel{$new_nrows, $C}($N, $stride_D, $stride_A, $stride_X), mask, Val{$negative}())))
                end
            end
        end
        push!(q.args, row_rem_quote)
    end
    push!(q.args, :(pD += $size_T*$row_rem; pA += $size_T*$row_rem))
    q
end

function mul_nt_quote(
    M::Union{Int,Symbol}, K::Union{Int,Symbol}, N::Union{Int,Symbol}, T, init::Bool,
    stride_A::Union{Int,Symbol} = M, stride_X::Union{Int,Symbol} = N, stride_D::Union{Int,Symbol} = M;
    negative::Bool = false, force_inline::Bool = false
)
    W, rows, cols, aloads = pick_kernel_size(T, M isa Int ? M : typemax(Int), N isa Int ? N : typemax(Int))
    q = force_inline ? quote $(Expr(:meta,:inline)) end : quote end
    if M isa Int
        row_reps, row_rem = divrem(M, rows)
        row_reps_total = row_reps + (row_rem > 0)
    else
        row_reps, row_rem = :blocked_row_repetitions, :row_remainder
        row_reps_total = typemax(Int)
        push!(q.args, :(($row_reps,$row_rem) = divrem($M,$rows)))
        push!(q.args, :(mask = VectorizationBase.mask($T, $row_rem & $(W-1))))
    end
    if K isa Int
        col_reps, col_rem = divrem(K, cols)
        col_reps_total = col_reps + (col_rem > 0)
    else
        col_reps, col_rem = :blocked_col_repetitons, :col_remainder
        col_reps_total = typemax(Int)
        push!(q.args, :(($col_reps,$col_rem) = divrem($K,$col)))
    end
    if row_reps_total == col_reps_total == 1 # must be fixed size arrays
        kernel = DynamicKernel( # To the kernel, we calculate an M*K block of N-length dot products
            M, K, N, stride_D, stride_A, stride_X, T, X_transposed = true, negative = negative
        )
        push!(q.args, :(pA = pointer(A); pD = pointer(D); pX = pointer(X)))
        #kernel_nt_quote(M, N, stride_A, stride_X, stride_D, K, T, init, false, nothing, negative = negative)
        push!(q.args, kernel_quote(kernel, init = init, force_inline = false, mask_ops = true, runtime_mask = false) ) 
        push!(q.args, :D)
        return q
    end
    size_T = sizeof(T)
    if row_reps_total == 1
        push!(q.args, :(pA = pointer(A); pD = pointer(D); pX = pointer(X)))
        kernel = DynamicKernel(
            M, cols, N, stride_D, stride_A, stride_X, T, X_transposed = true, negative = negative
        )
        kql = kernel_quote(kernel, init = init, force_inline = false, mask_ops = true, runtime_mask = false)
#        kql = kernel_nt_quote(M, cols, stride_A, stride_X, stride_D, K, T, init, false, nothing, negative = negative)
        if col_reps == 1
            push!(q.args, kql)
            push!(q.args, :(pD += $size_T*$stride_D*$cols; pX += $size_T*$cols))
        else
            loop = quote
                for c ∈ 0:$(col_reps isa Int ? col_reps -1 : :($col_reps-1))
                    $kql
                    pD += $size_T*$stride_D*$cols
                    pX += $size_T*$cols
                end
            end
            push!(q.args, loop)
        end
        if K isa Int
            if col_rem > 0
                kernel = DynamicKernel(
                    M, col_rem, N, stride_D, stride_A, stride_X, T, X_transposed = true, negative = negative
                )
                push!(q.args, kernel_quote(kernel, init = init, force_inline = false, mask_ops = true, runtime_mask = false))
            end
        else
            col_rem_quote = quote nothing end
            for c ∈ 1:cols - 1
                col_rem_quote = quote
                    if col_rem == $c
                        initkernel_nt!(pD, pA, pX, DKernel{$M,$c}($N, $stride_D, $stride_A, $stride_X), Val{$negatuve}())
                        nothing
                    else
                        $col_rem_quote
                    end
                end
            end
            push!(q.args, col_rem_quote)
        end
        push!(q.args, :D)
        return q
    elseif col_reps_total == 1
        push!(q.args, :(pA = pointer(A); pD = pointer(D); pX = pointer(X)))
        kernel = DynamicKernel(
            rows, K, N, stride_D, stride_A, stride_X, T, X_transposed = true, negative = negative
        )
        kql = kernel_quote(kernel, init = init, force_inline = false, mask_ops = false, runtime_mask = false)
        if row_reps == 1
            push!(q.args, kql)
            push!(q.args, :(pA += $size_T*$rows; pD += $size_T*$rows))
        else
            loop = quote
                for $(gensym(:r)) ∈ 0:$(row_reps isa Int ? row_reps -1 : :($row_reps - 1))
                    $kql
                    pA += $size_T*$rows
                    pD += $size_T*$rows
                end
            end
            push!(q.args, loop)
        end
        kernel = DynamicKernel(
            rows, K, N, stride_D, stride_A, stride_X, T, X_transposed = true, negative = negative
        )
        add_row_rem_expression!(q, row_rem, kernel, init = init)
        push!(q.args, :D)
        return q
    end
    push!(q.args, :(pX = pointer(X)))
    # We want to go ahead and define the mask, hoisting it out of the loop
    kernel = DynamicKernel(
        rows, cols, N, stride_D, stride_A, stride_X, T, X_transposed = true, negative = negative
    )
    kql = kernel_quote(kernel, init = init, force_inline = false, mask_ops = false, runtime_mask = false)
    inner = quote pA = pointer(A) end
    if col_reps > 1
        push!(inner.args, :(pD = pointer(D) + c*$size_T*$stride_D*$cols))
    else
        push!(inner.args, :(pD = pointer(D)))
    end
    if row_reps == 1
        push!(inner.args, kql)
        push!(inner.args, :(pA += $size_T*$rows; pD += $size_T*$rows))
    else
        loop = quote
            for $(gensym(:r)) ∈ 0:$(row_reps isa Int ? row_reps - 1 : :($row_reps - 1))
                $kql
                pA += $size_T*$rows
                pD += $size_T*$rows
            end
        end
        push!(inner.args, loop)
    end
    kernel = DynamicKernel(
        rows, cols, N, stride_D, stride_A, stride_X, T, X_transposed = true, negative = negative
    )
    add_row_rem_expression!(inner, row_rem, kernel, init = init)
    push!(inner.args, :(pX += $size_T*$(stride_X isa Int ? max(1,stride_X) : stride_X)*$cols))
    if col_reps == 1
        push!(q.args, inner)
    else
        loop = quote
            for c ∈ 0:$(col_reps isa Int ? col_reps - 1 : :($col_reps - 1))
                $inner
            end
        end
        push!(q.args, loop)
    end
    if K isa Int
        if col_rem > 0
            kernel = DynamicKernel(
                rows, col_rem, N, stride_D, stride_A, stride_X, T, X_transposed = true, negative = negative
            )
            kql = kernel_quote(kernel, init = init, force_inline = false, mask_ops = false, runtime_mask = false)
            inner = quote pA = pointer(A) end
            if col_reps > 1
                push!(inner.args, :(pD = pointer(D) + $size_T*$stride_D*$(cols*col_reps)))
            end
            if row_reps == 1
                push!(inner.args, kql)
                push!(inner.args, :(pA += $size_T*$rows; pD += $size_T*$rows))
            else
                loop = quote
                    for r ∈ 0:$(row_reps isa Int ? row_reps - 1 : :($row_reps - 1))
                        $kql
                        pA += $size_T*$rows
                        pD += $size_T*$rows
                    end
                end
                push!(inner.args, loop)
            end
            kernel = DynamicKernel(
            rows, col_rem, N, stride_D, stride_A, stride_X, T, X_transposed = true, negative = negative
            )
            add_row_rem_expression!(inner, row_rem, kernel, init = init)
            push!(q.args, inner)
        end
    else
        set_pointer_quote = quote
            pD = pointer(D) + $size_T*($row_reps*$rows + $stride_D*$col_reps*$cols)
            pX = pointer(X) + $size_T*($col_reps*$cols)
        end
        push!(q.args, set_pointer_quote)
        push!(q.args, column_remainder_xt_quote(M, cols, N, stride_D, stride_A, stride_X, negative))
    end
    push!(q.args, :D)

    q
end

function column_remainder_xt_quote(
    M::Union{Int,Symbol}, cols::Int, N::Union{Int,Symbol}, stride_D::Union{Int,Symbol}, stride_A::Union{Int,Symbol}, stride_X::Union{Int,Symbol}, negative::Bool
)
    mulfunc = negative ? :A_nmul_Xt! : :A_mul_Xt!
    q = quote nothing end
    for c ∈ 1:cols-1
        Dmat = if (M isa Integer) && (stride_D isa Integer)
            :(PtrMatrix{$M,$c,$T,$stride_D}(pD))
        else
            :(DynamicPtrMatrix{$T}(pD, ($M,$c), $stride_D))
        end
        Amat = if (N isa Integer) && (M isa Integer) && (stride_A isa Integer)
            :(PtrMatrix{$M,$N,$T,$stride_A}(pA))
        else
            :(DynamicPtrMatrix{$T}(pA,($M,$N),$stride_A))
        end
        Xmat = if (N isa Intger) && (stride_X isa Integer)
            :(PtrMatrix{$c,$N,$T,$stride_X}(pX))
        else
            :(DynamicPtrMatrix{$T}(pX, ($c,$N), $stride_X))
        end
        mulexpr = if (M isa Integer) && (stride_D isa Integer)
            :($mulfunc( $Dmat, $Amat, $Xmat ))
        else
            :($mulfunc( $Dmat, $Amat, $Xmat, Val{$c}() ))
        end
        q = quote
            if $c == $col_rem
                $mulexpr
            else
                $q
            end
        end
    end
    q
end

# Adjoints determine dispatch.
# Inlining the wrapper avoids allocating the "Adjoint" type when the arrays are heap allocated
# without requiring that we have to actually inline the entire matrix multiplication.
# quotenode_if_missing(symset, sym) = sym ∈ symset ? sym : QuoteNode(sym)
for init ∈ (true,false)
    for negative ∈ (true,false)
        namecore = Symbol(negative ? :nmul : :mul, init ? :add : Symbol())
        basefunc = Symbol(:A_,namecore,:Xt!)
        if init && !negative
            wrapfunc = :(LinearAlgebra.mul!)
        else
            wrapfunc = Symbol(namecore, :!)
        end
        known_params = Set{Symbol}()
        for Xismat ∈ (true,false)
            for Xissized ∈ (true,false)
                Xtype = if Xissized
                    if Xismat
                        Xp = union(known_params, Set([:N,:K,:PX]))
                        :(AbstractFixedSizeMatrix{N,K,T,PX})
                    else
                        Xp = union(known_params, Set([:K,:PX]))
                        :(AbstractFixedSizeVector{K,T,PX})
                    end
                else
                    Xp = known_params#copy(known_params)
                    Xismat ? :(StridedMatrix{T}) : :(StridedVector{T})
                end
                for Dissized ∈ (true,false)
                    if Dissized
                        Dtype = :(AbstractMutableFixedSizeMatrix{M,K,T,PD})
                        Dp = union(Xp, Set([:M,:K,:PD]))
                    else
                        Dtype = :(StridedMatrix{T})
                        Dp = Xp#copy(Xp)
                    end
                    for Aissized ∈ (true,false)
                        if !Xissized && !Dissized && !Aissized
                            continue
                        end
                        if Aissized
                            Atype = :(AbstractFixedSizeMatrix{M,N,T,PA})
                            Ap = union(Dp, Set([:M,:N,:PA]))
                        else
                            Atype = :(StridedMatrix{T})
                            Ap = Dp
                        end
                        body = quote end
                        if :PA ∈ Ap
                            PAS = :PA
                        else
                            PAS = QuoteNode(:PA)
                            push!(body.args, :(PA = stride(A,2)))
                        end
                        if Xismat
                            if :PX ∈ Ap
                                PXS = :PX
                            else
                                PXS = QuoteNode(:PX)
                                push!(body.args, :(PX = stride(X,2)))
                            end
                        else
                            PXS = 0
                        end
                        if :PD ∈ Ap
                            PDS = :PD
                        else
                            PDS = QuoteNode(:PD)
                            push!(body.args, :(PD = stride(D,2)))
                        end
                        if :M ∉ Ap && :N ∉ Ap
                            push!(body.args, :((M,N) = size(A)))
                        elseif :M ∉ Ap
                            push!(body.args, :(M = size(A,1)))
                        elseif :N ∉ Ap
                            push!(body.args, :(N = size(A,2)))
                        end
                        MS = :M ∈ Ap ? :M : QuoteNode(:M)
                        NS = :N ∈ Ap ? :N : QuoteNode(:N)
                        args = [:(D::$Dtype), :(A::$Atype), :(X::$Xtype)]
                        wheretypes = [Ap...]
                        push!(wheretypes, :T)
                        if :K ∉ Ap
                            push!(args, :(::Val{K}))
                            push!(wheretypes, :K)
                        end
                        push!(body.args, :(mul_nt_quote($MS,K,$NS,T,$init,$PAS,$PXS,$PDS,negative=$negative,force_inline=false)))
                        @eval @generated function $basefunc($(args...)) where {$(wheretypes...)}
                            $body
                        end
                        args[3] = :(X′::LinearAlgebra.Adjoint{T,<: $Xtype})
                        @eval @inline function $wrapfunc($(args...)) where {$(wheretypes...)}
                            $basefunc(D, A, X′.parent)
                        end
                    end
                end
            end
        end
        if negative
            @eval @inline function $wrapfunc( # Fall back
                D::AbstractMatrix{T},
                A::AbstractMatrix{T},
                X′::LinearAlgebra.Adjoint{T,<:AbstractMatrix{T}}
            ) where {T <: BLAS.BlasFloat}
                BLAS.gemm!('N','T',$(negative ? :(-one(T)) : :(one(T))),A,X′.parent,$(init ? :(zero(T)) : :(one(T))),D)
            end 
        end
    end
end



function mul_tn_quote(
    M,K,N,T,init::Bool, stride_A = M, stride_X = N, stride_D = M;
    negative::Bool = false, force_inline = false,
    Dsym::Symbol = :D, Asym::Symbol = :A, Xsym::Symbol = :X
)
    # aloads x cols is the effective kernel size here.
    W, Wshift = VectorizationBase.pick_vector_width_shift(K, T)
    # @show M, W, N
    W2, rows_times_W, cols, rows = pick_kernel_size(T, M*W, N)
    row_reps, row_rem = divrem(M, rows)
    col_reps, col_rem = divrem(N, cols)
    row_reps_total = row_reps + (row_rem > 0)
    col_reps_total = col_reps + (col_rem > 0)
    # @show rows, cols, row_reps, col_reps, row_rem, col_rem
    q = force_inline ? quote $(Expr(:meta,:inline)) end : quote end
    if row_reps_total == col_reps_total == 1
        kernel = DynamicKernel(M,N,K,stride_D,stride_A,stride_X,T,negative = negative)
        push!(q.args, :(pA = pointer($Asym); pD = pointer($Dsym); pX = pointer($Xsym)))
        push!(q.args, kernel_tn_quote(kernel, init, false))#M, N, stride_A, stride_X, stride_D, K, T, init, false, negative = negative))
        push!(q.args, :D)
        return q
    end
    size_T = sizeof(T)
    if row_reps_total == 1
        push!(q.args, :(pA = pointer($Asym); pD = pointer($Dsym); pX = pointer($Xsym)))
        kernel = DynamicKernel(M,cols,K, stride_D, stride_A, stride_X, T, negative=negative)
        kql = kernel_tn_quote(kernel, init, false)
        if col_reps == 1
            push!(q.args, kql)
            push!(q.args, :(pD += $size_T*$stride_D*$cols; pX += $size_T*$stride_X*$cols))
        else
            loop = quote
                for c ∈ 0:$(col_reps-1)
                    $kql
                    pD += $size_T*$stride_D*$cols
                    pX += $size_T*$stride_X*$cols
                end
            end
            push!(q.args, loop)
        end
        if col_rem > 0
            kernel = DynamicKernel(M,col_rem,K,stride_D,stride_A,stride_X,T,negative=negative)
            push!(q.args, kernel_tn_quote(kernel,init,false))
        end
        push!(q.args, :D)
        return q
    end
    if col_reps_total == 1
        push!(q.args, :(pA = pointer($Asym); pD = pointer($Dsym); pX = pointer($Xsym)))
        kernel = DynamicKernel(rows,N,K,stride_D,stride_A,stride_X,T,negative=negative)
        kql = kernel_tn_quote(kernel,init,false)
        if row_reps == 1
            push!(q.args, kql)
            push!(q.args, :(pA += $size_T*$stride_A*$rows; pD += $size_T*$rows))
        else
            loop = quote
                for r ∈ 0:$(row_reps-1)
                    $kql
                    pA += $size_T*$stride_A*$rows
                    pD += $size_T*$rows
                end
            end
            push!(q.args, loop)
        end
        if row_rem > 0
            kernel = DynamicKernel(row_rem,N,K,stride_D,stride_A,stride_X,T,negative=negative)
            push!(q.args, kernel_tn_quote(kernel, init, false))
        end
        push!(q.args, :D)
        return q
    end
    push!(q.args, :(pD = pointer($Dsym); pX = pointer($Xsym)))
    kernel = DynamicKernel(rows,cols,K,stride_D,stride_A,stride_X,T,negative=negative)
    kql = kernel_tn_quote(kernel, init, false)
    inner = quote pA = pointer($Asym) end
    # @show col_reps
    if col_reps > 1
        push!(inner.args, :(pD = pointer($Dsym) + c*$size_T*$stride_D*$cols))
    end
    # @show inner
    if row_reps == 1
        push!(inner.args, kql)
        push!(inner.args, :(pA += $size_T*$stride_A*$rows; pD += $size_T*$rows))
    else
        loop = quote
            for r ∈ 0:$(row_reps - 1)
                $kql
                pA += $size_T*$stride_A*$rows
                pD += $size_T*$rows
            end
        end
        push!(inner.args, loop)
    end
    if row_rem > 0
        kernel = DynamicKernel(row_rem,cols,K,stride_D,stride_A,stride_X,T,negative=negative)
        push!(inner.args, kernel_tn_quote(kernel, init, false))
    end
    push!(inner.args, :(pX += $size_T*$stride_X*$cols))
    if col_reps == 1
        # @show col_reps
        push!(q.args, inner)
    else
        # @show col_reps
        loop = quote
            for c ∈ 0:$(col_reps-1)
                $inner
            end
        end
        push!(q.args, loop)
    end
    if col_rem > 0
        kernel = DynamicKernel(rows,col_rem,K,stride_D,stride_A,stride_X,T,negative=negative)
        kql = kernel_tn_quote(kernel, init, false)
        inner = quote
            pA = pointer($Asym)
            pD = pointer($Dsym) + $size_T*$stride_D*$(cols*col_reps)
        end
        if row_reps == 1
            push!(inner.args, kql)
        else
            loop = quote
                for r ∈ 0:$(row_reps - 1)
                    $kql
                    pA += $size_T*$stride_A*$rows
                    pD += $size_T*$rows
                end
            end
            push!(inner.args, loop)
        end
        if row_rem > 0
            kernel = DynamicKernel(row_rem,col_rem,K,stride_D,stride_A,stride_X,T,negative=negative)
            push!(inner.args, kernel_tn_quote(kernel, init, false))
        end
        push!(q.args, inner)
    end
    push!(q.args, :D)
    q
end


function K_unknown_At_mul_X_quote(A, X, M, N, T, PD, negative, Dsym, Asym, Xsym)
    if A <: AbstractMutableFixedSizeMatrix
        K = A.parameters[1].parameters[1]
        stride_A = A.parameters[4]
        stride_X = gensym(:PX)
        def_quote = :($stride_X = stride(X,2))
    elseif X <: AbstractMutableFixedSizeMatrix
        K = X.parameters[1].parameters[1]
        stride_A = gensym(:PA)
        stride_X = X.parameters[4]
        def_quote = :($stride_A = stride(A,2))
    else
        stride_A = gensym(:PA)
        stride_X = gensym(:PX)
        K = gensym(:K)
        def_quote = quote
            $K = size(A,1)
            $stride_A = stride(A,2)
            $stride_X = stride(X,2)
        end
    end
    quote
        $def_quote
        $(mul_tn_quote(M,K,N,T,true,stride_A,stride_X,PD,negative = negative,force_inline=false))
    end
end

for init ∈ (true,false)
    basename = init ? :mul : :muladd
    for negative ∈ (true,false)
        core = negative ? Symbol(:n,basename) : basename
        if !negative && init
            wrapfunc = :(LinearAlgebra.mul!)
        else
            wrapfunc = Symbol(core, :!)
        end
        basefunc = Symbol(:At_,core,:_X!)
        @eval @generated function $basefunc(
            D::AbstractMutableFixedSizeMatrix{M,N,T,PD},
            A::AbstractMatrix{T},
            X::AbstractMatrix{T}
        ) where {M,N,T,PD}
            K_unknown_At_mul_X_quote(A, X, M, N, T, PD, $negative, :D, :A, :X)
        end
        @eval @generated function $basefunc(
            D::AbstractMutableFixedSizeMatrix{M,N,T,PD},
            A::AbstractMutableFixedSizeMatrix{K,M,T,PA},
            X::AbstractMutableFixedSizeMatrix{K,N,T,PX}
        ) where {M,K,N,T,PD,PA,PX}
            mul_tn_quote(M,K,N,T,$init,PA,PX,PD,negative = $negative,force_inline = false)
        end
        @eval @generated function $basefunc(
            D::StridedMatrix{T},
            A::AbstractMutableFixedSizeMatrix{K,M,T,PA},
            X::AbstractMutableFixedSizeMatrix{K,N,T,PX}
        ) where {M,K,N,T,PA,PX}
            q = mul_tn_quote(M,K,N,T,$init,PA,PX,:PD,negative = $negative,force_inline=false)
            pushfirst!(q.args, :(PD = stride(D,2)))
            q
        end
        @eval @generated function $basefunc(
            D::AbstractMutableFixedSizeMatrix{M,N,T,PD},
            A::StridedMatrix{T},
            X::AbstractMutableFixedSizeMatrix{K,N,T,PX}
        ) where {M,K,N,T,PD,PX}
            q = mul_tn_quote(M,K,N,T,$init,:PA,PX,PD,negative = $negative,force_inline=false)
            pushfirst!(q.args, :(PA = stride(A,2)))
            q
        end
        @eval @generated function $basefunc(
            D::AbstractMutableFixedSizeMatrix{M,N,T,PD},
            A::AbstractMutableFixedSizeMatrix{K,M,T,PA},
            X::StridedMatrix{T}
        ) where {M,K,N,T,PD,PA}
            q = mul_tn_quote(M,K,N,T,$init,PA,:PX,PD,negative = $negative,force_inline=false)
            pushfirst!(q.args, :(PX = stride(X,2)))
            q
        end
        @eval @inline function $wrapfunc(
            D::AbstractMutableFixedSizeMatrix{M,N,T,PD},
            A′::LinearAlgebra.Adjoint{T,<:StridedMatrix{T}},
            X::StridedMatrix{T}
        ) where {M,N,T <: Union{Float32,Float64},PD}
            $basefunc(D, A′.parent, X)
        end
    end
end

@inline function nmul!(
    D::AbstractMatrix{T},
    A′::LinearAlgebra.Adjoint{T,<:AbstractMatrix{T}},
    X::AbstractMatrix{T}
) where {T <: BLAS.BlasFloat}
    BLAS.gemm!('T','N',-one(T),A′.parent,X,zero(T),D)
end


@generated function Base.:*(A::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeVector{M,T,P}}, B::AbstractFixedSizeMatrix{M,N,T,P}) where {M,N,T,P}
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
    reps = P >>> Wshift
    V = Vec{W,T}
    q = quote
        C = MutableFixedSizeMatrix{$M,$N,$T}(undef)
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
    if B <: AbstractConstantFixedSizeMatrix
        push!(q.args, :(ConstantFixedSizeMatrix(C)))
    else
        push!(q.args, :(C))
    end
    q
end
@generated function Base.:*(
    sp::StackPointer,
    A::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeVector{M,T,P}},
    B::AbstractFixedSizeMatrix{M,N,T,P}
) where {M,N,T,P}
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
    reps = P >>> Wshift
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
        sp + $(VectorizationBase.align(sizeof(T)*P*N)), C
    end
    q
end

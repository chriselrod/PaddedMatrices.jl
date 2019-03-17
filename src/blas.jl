

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


@generated function Base.:*(A::AbstractMutableFixedSizePaddedMatrix{M,N,T,ADR}, X::AbstractMutableFixedSizePaddedMatrix{N,P,T,XR}) where {M,N,P,T,ADR,XR}
    quote
        D = AbstractMutableFixedSizePaddedMatrix{$(Tuple{M,P}),$T,2,$ADR,$(ADR*P)}(undef)
        $(mulquote(ADR,N,P,ADR,XR,T))
        ConstantFixedSizePaddedMatrix(D)
    end
end


@inline function LinearAlgebra.mul!(C, A, B::PaddedMatrix)
    Bdata = B.data
    @uviews Bdata mul!(C, A, @view(B.data[1:B.nrow,:]))
end

@inline function LinearAlgebra.mul!(C, A::PaddedMatrix, B::PaddedMatrix)
    Bdata = B.data
    @uviews Bdata mul!(C, A.data, @view(B.data[1:B.nrow,:]))
end
@inline function LinearAlgebra.mul!(C::PaddedMatrix, A, B::PaddedMatrix)
    Bdata = B.data
    @uviews Bdata mul!(C.data, A, @view(B.data[1:B.nrow,:]))
end
@inline function LinearAlgebra.mul!(C::PaddedMatrix, A::PaddedMatrix, B::PaddedMatrix)
    Bdata = B.data
    @uviews Bdata mul!(C.data, A.data, @view(B.data[1:B.nrow,:]))
end

@generated function LinearAlgebra.mul!(D::AbstractMutableFixedSizePaddedMatrix{M,P,T,ADR},
                            A::AbstractMutableFixedSizePaddedMatrix{M,N,T,ADR},
                            X::AbstractMutableFixedSizePaddedMatrix{N,P,T,XR}) where {M,N,P,T,ADR,XR}
    quote
        $(Expr(:meta,:inline))
        $(mulquote(ADR,N,P,ADR,XR,T))
    end
end

@generated function LinearAlgebra.mul!(D::AbstractMutableFixedSizePaddedVector{M,T,ADR},
                            A::AbstractMutableFixedSizePaddedMatrix{M,N,T,ADR},
                            X::AbstractMutableFixedSizePaddedVector{N,T,XR}) where {M,N,P,T,ADR,XR}

    mulquote(ADR,N,1,ADR,XR,T)
end
# @generated function LinearAlgebra.mul!(D::PtrMatrix{M,P,T,ADR},
#                             A::PtrMatrix{M,N,T,ADR},
#                             X::PtrMatrix{N,P,T,XR},
#                             prefetchAX = nothing) where {M,N,P,T,ADR,XR}
#     # The hack here for ADR vs M in the first slot is this:
#     # When called on actual matrices, M gives the "pretend size"
#     # While ADR gives this actual number of rows, so we pass on that.
#     # While for PtrMatrices, M reflects the true size of the submatrix.
#     # That is because it is partitioned along the blocks created by
#     # jBLAS.blocking_structure(M, N, P, T)
#     # which are always given an appropriate size, because it's only
#     # called after going through out filtering here. For PtrMatrices
#     # the stride can be far larger than the size of these submatrices,
#     # so we pass along M as the correct number of rows.
#     mulquote(M,N,P,ADR,XR,T,prefetchAX)
# end


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

@inline function Base.:+(A::AbstractFixedSizePaddedArray{S,T,N,P,L}, B::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = A[i] + B[i]
    end
    ConstantFixedSizePaddedArray(mv)
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
@inline function Base.:-(A::AbstractFixedSizePaddedArray{S,T,N,P,L}, B::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = A[i] - B[i]
    end
    ConstantFixedSizePaddedArray(mv)
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
@inline function Base.:*(A::AbstractFixedSizePaddedArray{S,T,N,P,L}, b::T) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = A[i] * b
    end
    ConstantFixedSizePaddedArray(mv)
end
@inline function Base.:*(a::T, B::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = a * B[i]
    end
    ConstantFixedSizePaddedArray(mv)
end
@inline function SIMDPirates.vfma(a::T, x::AbstractFixedSizePaddedArray{S,T,N,P,L}, y::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = a * x[i] + y[i]
    end
    ConstantFixedSizePaddedArray(mv)
end
@inline function SIMDPirates.vfnmadd(a::T, x::AbstractFixedSizePaddedArray{S,T,N,P,L}, y::AbstractFixedSizePaddedArray{S,T,N,P,L}) where {S,T<:Number,N,P,L}
    mv = MutableFixedSizePaddedArray{S,T,N,P,L}(undef)
    @fastmath @inbounds @simd ivdep for i ∈ 1:L
        mv[i] = y[i] - a * x[i]
    end
    ConstantFixedSizePaddedArray(mv)
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

    mulquote(ADR,N,P,ADR,XR,T,init)
end
function mulquote(M,N,P,ADR,XR,T,init=:initkernel!,prefetchAX=nothing)
    (L1S, L2S, L3S), num = blocking_structure(M, N, P, T)
    if num == 0
        if init == :kernel! || M*N*P > 14^3
            return cache_mulquote(M,N,P,ADR,XR,L1S,T,init,prefetchAX)
        else
            return kernel_quote(M,N,P,ADR,XR,T,true,true)
        end
        # return base_mulquote(M,N,P,ADR,XR,T)
    elseif num == 1
        # Loop over L1 cache blocks
        return cache_mulquote(M,N,P,ADR,XR,L1S,T,init,prefetchAX)
    elseif num == 2
        # Loop over L2 cache blocks
        return cache_mulquote(M,N,P,ADR,XR,L1S,L2S,T,init,prefetchAX)
    else #num == 3
        # Loop over L3 cache blocks.
        # return cache_mulquote(ADR,N,P,ADR,XR,L1S,L2S,L3S,T,init)
        # Except that they are fused, so we aren't doing anything special.
        return cache_mulquote(M,N,P,ADR,XR,L1S,L3S,T,init,prefetchAX) # should recurse calling mul and gemm!
    end
end

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

function cache_mulquote(M,N,P,stride_AD,stride_X,(L1M,L1N,L1P),::Type{T}, init = :initkernel!, primary = :kernel!) where T

    M_iter, M_remain = divrem(M, L1M)
    N_iter, N_remain = divrem(N, L1N)
    P_iter, P_remain = divrem(P, L1P)
    T_size = sizeof(T)

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
    initmul = ifelse(init == :initkernel!, :mul!, :gemm!)

    q = quote
        pD, pA, pX = pointer(D), pointer(A), pointer(X)


        for pᵢ ∈ 0:$(P_iter-1)
            pd_off = pᵢ*$(L2P*stride_AD*T_size)
            px_off = pᵢ*$(L2P*stride_X*T_size)
            for mᵢ ∈ 0:$(M_iter-1)
                m_off = mᵢ*$(L2M*T_size)
                pD_temp = PtrMatrix{$L2M,$L2P,$T,$stride_AD}(pD + m_off + pd_off)
                pA_temp = PtrMatrix{$L2M,$L2N,$T,$stride_AD}(pA + m_off)
                pX_temp = PtrMatrix{$L2N,$L2P,$T,$stride_X}(pX + px_off)
                $(prefetch_ ? quote
                    prefetch(pD_temp, Val(1))
                    prefetch(pA_temp, Val(0))
                    prefetch(pX_temp, Val(0))
                end : nothing )
                $initmul(pD_temp,
                     pA_temp,
                     pX_temp)
                for nᵢ ∈ 1:$(N_iter-1)
                    pA_temp = PtrMatrix{$L2M,$L2N,$T,$stride_AD}(pA + m_off + nᵢ*$(L2N*stride_AD*T_size))
                    pX_temp = PtrMatrix{$L2N,$L2P,$T,$stride_X}(pX + nᵢ*$(L2N*T_size) + px_off)
                    $(prefetch_ ? quote
                        prefetch(pA_temp, Val(0))
                        prefetch(pX_temp, Val(0))
                    end : nothing )
                    gemm!(pD_temp,
                         pA_temp,
                         pX_temp)
                end
                $(N_remain == 0 ? nothing : quote
                    pA_temp = PtrMatrix{$L2M,$N_remain,$T,$stride_AD}(pA + m_off + $(N_iter*L2N*stride_AD*T_size))
                    pX_temp = PtrMatrix{$N_remain,$L2P,$T,$stride_X}(pX + $(N_iter*L2N*T_size) + px_off)
                    $(prefetch_ ? quote
                        prefetch(pA_temp, Val(0))
                        prefetch(pX_temp, Val(0))
                    end : nothing )
                    gemm!(pD_temp, pA_temp, pX_temp)
                end)
            end
            ### Check if we need to add an expression for a remainder of M.
            $(M_remain == 0 ? nothing : quote
                pD_temp = PtrMatrix{$M_remain,$L2P,$T,$stride_AD}(pD + $(M_iter*L2M*T_size) + pd_off)
                pA_temp = PtrMatrix{$M_remain,$L2N,$T,$stride_AD}(pA + $(M_iter*L2M*T_size))
                pX_temp = PtrMatrix{$L2N,$L2P,$T,$stride_X}(pX + px_off)
                $(prefetch_ ? quote
                    prefetch(pD_temp, Val(1))
                    prefetch(pA_temp, Val(0))
                    prefetch(pX_temp, Val(0))
                end : nothing )
                $initmul(pD_temp,
                     pA_temp,
                     pX_temp)
                for nᵢ ∈ 1:$(N_iter-1)
                    pA_temp = PtrMatrix{$M_remain,$L2N,$T,$stride_AD}(pA + $(M_iter*L2M*T_size) + nᵢ*$(L2N*stride_AD*T_size))
                    pX_temp = PtrMatrix{$L2N,$L2P,$T,$stride_X}(pX + nᵢ*$(L2N*T_size) + px_off)
                    $(prefetch_ ? quote
                        prefetch(pA_temp, Val(0))
                        prefetch(pX_temp, Val(0))
                    end : nothing )
                    gemm!(pD_temp,
                         pA_temp,
                         pX_temp)
                end
                $(N_remain == 0 ? nothing : quote
                    pA_temp = PtrMatrix{$M_remain,$N_remain,$T,$stride_AD}(pA + $(M_iter*L2M*T_size + N_iter*L2N*stride_AD*T_size))
                    pX_temp = PtrMatrix{$N_remain,$L2P,$T,$stride_X}(pX + $(N_iter*L2N*T_size) + px_off)
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
                    pD_temp = PtrMatrix{$L2M,$P_remain,$T,$stride_AD}(pD + m_off + $(P_iter*L2P*stride_AD*T_size))
                    pA_temp = PtrMatrix{$L2M,$L2N,$T,$stride_AD}(pA + m_off)
                    pX_temp = PtrMatrix{$L2N,$P_remain,$T,$stride_X}(pX + $(P_iter*L2P*stride_X*T_size))
                    $(prefetch_ ? quote
                        prefetch(pD_temp, Val(1))
                        prefetch(pA_temp, Val(0))
                        prefetch(pA_temp, Val(0))
                    end : nothing )
                    $initmul(pD_temp,
                         pA_temp,
                         pX_temp)
                    for nᵢ ∈ 1:$(N_iter-1)
                        pA_temp = PtrMatrix{$L2M,$L2N,$T,$stride_AD}(pA + m_off + nᵢ*$(L2N*stride_AD*T_size))
                        pX_temp = PtrMatrix{$L2N,$P_remain,$T,$stride_X}(pX + nᵢ*$(L2N*T_size) + $(P_iter*L2P*stride_X*T_size))
                        $(prefetch_ ? quote
                            prefetch(pA_temp, Val(0))
                            prefetch(pX_temp, Val(0))
                        end : nothing )
                        gemm!(pD_temp,
                             pA_temp,
                             pX_temp)
                    end
                    $(N_remain == 0 ? nothing : quote
                        pA_temp = PtrMatrix{$L2M,$N_remain,$T,$stride_AD}(pA + m_off + $(N_iter*L2N*stride_AD*T_size))
                        pX_temp = PtrMatrix{$N_remain,$P_remain,$T,$stride_X}(pX + $(N_iter*L2N*T_size + P_iter*L2P*stride_X*T_size))
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
                    pD_temp = PtrMatrix{$M_remain,$P_remain,$T,$stride_AD}(pD + $(M_iter*L2M*T_size + P_iter*L2P*stride_AD*T_size))
                    pA_temp = PtrMatrix{$M_remain,$L2N,$T,$stride_AD}(pA + $(M_iter*L2M*T_size))
                    pX_temp = PtrMatrix{$L2N,$P_remain,$T,$stride_X}(pX + $(P_iter*L2P*stride_X*T_size))
                    $(prefetch_ ? quote
                        prefetch(pD_temp, Val(1))
                        prefetch(pA_temp, Val(0))
                        prefetch(pX_temp, Val(0))
                    end : nothing )
                    $initmul(pD_temp,
                         pA_temp,
                         pX_temp)
                    for nᵢ ∈ 1:$(N_iter-1)
                        pA_temp = PtrMatrix{$M_remain,$L2N,$T,$stride_AD}(pA + $(M_iter*L2M*T_size) + nᵢ*$(L2N*stride_AD*T_size))
                        pX_temp = PtrMatrix{$L2N,$P_remain,$T,$stride_X}(pX + nᵢ*$(L2N*T_size) + $(P_iter*L2P*stride_X*T_size))
                        $(prefetch_ ? quote
                            prefetch(pA_temp, Val(0))
                            prefetch(pX_temp, Val(0))
                        end : nothing )
                        gemm!(pD_temp,
                             pA_temp,
                             pX_temp)
                    end
                    $(N_remain == 0 ? nothing : quote
                        pA_temp = PtrMatrix{$M_remain,$N_remain,$T,$stride_AD}(pA + $(M_iter*L2M*T_size + N_iter*L2N*stride_AD*T_size))
                        pX_temp = PtrMatrix{$N_remain,$P_remain,$T,$stride_X}(pX + $(N_iter*L2N*T_size + P_iter*L2P*stride_X*T_size))
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

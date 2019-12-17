"""
Base.@pure Kernel(Mₖ,N,Pₖ,stride_AD,stride_X) = Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}()

The kernel is typed based on M and P. M is a critical component of vector length,
while the kernel is unrolled across P. It simply loops over N.
"""
abstract type AbstractKernel end
abstract type AbstractSizedKernel{Mₖ,Pₖ} <: AbstractKernel end
struct Kernel{Mₖ,Pₖ,stride_A,stride_X,stride_D,N} <: AbstractSizedKernel{Mₖ,Pₖ} end
Base.@pure Kernel(Mₖ,N,Pₖ,stride_A,stride_X,stride_D) = Kernel{Mₖ,Pₖ,stride_A,stride_X,stride_D,N}()
struct DKernel{Mₖ,Pₖ} <: AbstractSizedKernel{Mₖ,Pₖ}
    N::Int
    stride_D::Int
    stride_A::Int
    stride_X::Int
end
Base.@pure DKernel(Mₖ,Pₖ,N,stride_D,stride_A,stride_X) = Kernel{Mₖ,Pₖ}(N,stride_D,stride_A,stride_X)

struct DynamicKernel{T} <: AbstractKernel
    R::Int
    C::Int
    N::Union{Symbol,Int}
    stride_D::Union{Symbol,Int}
    stride_A::Union{Symbol,Int}
    stride_X::Union{Symbol,Int}
    X_transposed::Bool
    negative::Bool
end

@noinline function reps_and_rem(kernel::DynamicKernel{T}) where {T}
    @unpack R = kernel
    W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
    Riter = R >>> Wshift
    Rrem = R & (W-1)
    Riter, Rrem
end

@noinline function DynamicKernel{T}(
    R::Int, C::Int, N::Union{Symbol,Int},
    stride_D::Union{Symbol,Int}, stride_A::Union{Symbol,Int}, stride_X::Union{Symbol,Int}, X_transposed::Bool = false
) where {T}
    DynamicKernel{T}(R, C, N, stride_D, stride_A, stride_X, X_transposed, false)
end
@noinline function mul_block(V, W, R1, R2, m_rep, N, P, poffset::Int = 0, vA = :vA, B = :B, gemm = nothing)
    Prange = (1 + poffset):(P + poffset)
    loop_max = isa(N, Number) ? N - 1 : :($N - 1)
    if gemm === nothing
        loop_min = 1
        initialize = quote
            $([:(
                $(Symbol(:Acol_,mr)) = SIMDPirates.vload($V, $vA, $(W*(mr-1)))
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [ :(@inbounds $(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmul(
                $(Symbol(:Acol_,mr)), $B[ $(1 + (p-1)*R2) ])
            ) for mr ∈ 1:m_rep]...) for p ∈ Prange]...)
        end
    else
        loop_min = 0
        R3 = gemm
        initialize = quote
            $([Expr(:block, [ :($(Symbol(:C_, mr+1, :_, p)) = vload($V, vout, $(W*mr + (p-1)*R3))) for mr ∈ 0:m_rep-1]...) for p ∈ Prange]...)
        end
    end
    quote
        $initialize
        @inbounds for n ∈ $loop_min:$loop_max
            $([:(
                $(Symbol(:Acol_,mr)) = SIMDPirates.vload($V, $vA, $(W*(mr-1)) + n*$R1)
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [:($(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmuladd(
                $(Symbol(:Acol_,mr)), $B[n + $(1 + (p-1)*R2)],
                $(Symbol(:C_, mr, :_, p)) )) for mr ∈ 1:m_rep]...) for p ∈ Prange]...
            )
        end
    end
end
@noinline function mul_block(V, W, R1, R2, m_rep, N, P, poffset::Symbol, vA = :vA, B = :B, gemm = nothing)
    Prange = 1:P
    loop_max = isa(N, Number) ? N - 1 : :($N - 1)
    if gemm === nothing
        loop_min = 1
        initialize = quote
            $([:(
                $(Symbol(:Acol_,mr)) = SIMDPirates.vload($V, $vA, $(W*(mr-1)))
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [ :(@inbounds $(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmul(
                $(Symbol(:Acol_,mr)), $B[ 1 + ($(p-1)+$poffset)*$R2 ])
            ) for mr ∈ 1:m_rep]...) for p ∈ Prange]...)
        end
    else
        loop_min = 0
        R3 = gemm
        initialize = quote
            $([Expr(:block, [ :($(Symbol(:C_, mr+1, :_, p)) = vload($V, vout, $(W*mr) + ($(p-1)+$poffset)*$R3)) for mr ∈ 0:m_rep-1]...) for p ∈ Prange]...)
        end
    end
    quote
        $initialize
        @inbounds for n ∈ $loop_min:$loop_max
            $([:(
                $(Symbol(:Acol_,mr)) = vload($V, $vA, $(W*(mr-1)) + n*$R1)
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [:($(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmuladd(
                $(Symbol(:Acol_,mr)), $B[n + 1 + ($(p-1)+$poffset)*$R2],
                $(Symbol(:C_, mr, :_, p)) )) for mr ∈ 1:m_rep]...) for p ∈ Prange]...
            )
        end
    end
end

"""
The suffix: _nt
stands for
    n - not transposed
    t - transposed

Meaning the operation is
A * B′
ie, A is not transposed, and B is transposed.
"""
@noinline function mul_block_nt(V, W, R1, R2, m_rep, N, P, poffset::Int = 0, vA = :vA, B = :B, gemm = nothing)
    Prange = (1 + poffset):(P + poffset)
    loop_max = isa(N, Number) ? N - 1 : :($N - 1)
    if gemm === nothing
        loop_min = 1
        initialize = quote
            $([:(
                $(Symbol(:Acol_,mr)) = SIMDPirates.vload($V, $vA, $(W*(mr-1)))
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [ :(@inbounds $(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmul(
                $(Symbol(:Acol_,mr)), $B[ $p ])
                ) for mr ∈ 1:m_rep]...) for p ∈ Prange]...)
        end
    else
        loop_min = 0
        R3 = gemm
        initialize = quote
            $([Expr(:block, [ :($(Symbol(:C_, mr+1, :_, p)) = vload($V, vout, $(W*mr + (p-1)*R3))) for mr ∈ 0:m_rep-1]...) for p ∈ Prange]...)
        end
    end
    quote
        $initialize
        @inbounds for n ∈ $loop_min:$loop_max
            $([:(
                $(Symbol(:Acol_,mr)) = SIMDPirates.vload($V, $vA, $(W*(mr-1)) + n*$R1)
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [:($(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmuladd(
                $(Symbol(:Acol_,mr)), $B[n*$R2 + $p],
                $(Symbol(:C_, mr, :_, p)) )) for mr ∈ 1:m_rep]...) for p ∈ Prange]...
            )
        end
    end
end
@noinline function mul_block_nt(V, W, R1, R2, m_rep, N, P, poffset::Symbol, vA = :vA, B = :B, gemm = nothing)
    Prange = 1:P
    loop_max = isa(N, Number) ? N - 1 : :($N - 1)
    if gemm === nothing
        loop_min = 1
        initialize = quote
            $([:(
                $(Symbol(:Acol_,mr)) = SIMDPirates.vload($V, $vA, $(W*(mr-1)) )
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [ :(@inbounds $(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmul(
                $(Symbol(:Acol_,mr)), $B[ $p + $poffset ])
            ) for mr ∈ 1:m_rep]...) for p ∈ Prange]...)
        end
    else
        loop_min = 0
        R3 = gemm
        initialize = quote
            $([Expr(:block, [ :($(Symbol(:C_, mr+1, :_, p)) = vload($V, vout, $(W*mr) + ($(p-1)+$poffset)*$R3)) for mr ∈ 0:m_rep-1]...) for p ∈ Prange]...)
        end
    end
    quote
        $initialize
        @inbounds for n ∈ $loop_min:$loop_max
            $([:(
                $(Symbol(:Acol_,mr)) = SIMDPirates.vload($V, $vA, $(W*(mr-1)) + n*$R1)
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [:($(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmuladd(
                $(Symbol(:Acol_,mr)), $B[n*$R2 + $p+$poffset],
                $(Symbol(:C_, mr, :_, p)) )) for mr ∈ 1:m_rep]...) for p ∈ Prange]...
            )
        end
    end
end
@noinline function store_block(W, R1, m_rep, P, poffset::Int = 0)
    Prange = (1 + poffset):(P + poffset)
    q = quote end
    for p ∈ Prange, mr ∈ 1:m_rep
        push!(q.args, :(vstore!(vout + $((p-1)*R1 + (mr-1)*W), $(Symbol(:C_, mr, :_, p)))))
    end
    q
end
@noinline function store_block(W, R1, m_rep, P, poffset::Symbol)
    Prange = 1:P
    q = quote end
    for p ∈ Prange, mr ∈ 1:m_rep
        push!(q.args, :(vstore!(vout + ($poffset + $(p-1))*$R1 + $((mr-1)*W), $(Symbol(:C_, mr, :_, p)))))
    end
    q
end

@noinline function static_mul_quote(M,N,P,T,R1,R2)
    L3 = R1 * P
    W = VectorizationBase.pick_vector_width(R1, T)
    m_rep = R1 ÷ W
    V = Vec{W,T}
    num_reps = cld(L3 ÷ W + 3, VectorizationBase.REGISTER_COUNT)
    if num_reps == 1
        outtup = Expr(:tuple)
        for p ∈ 1:P, mr ∈ 1:m_rep, m ∈ 1:W
            push!(outtup.args, :($(Symbol(:C_, mr, :_, p))[$m].value ))
        end
        return quote
            $(Expr(:meta, :inline))
            vA = VectorizationBase.vectorizable(A)
            $(mul_block(V, W, R1, R2, m_rep, N, P))
            output_data = $outtup
        end
    end
    piter = cld(P, num_reps)
    q = quote
        $(Expr(:meta, :inline))
        out = FixedSizeMatrix{$M,$P,$T,$R1,$L3}(undef)
        vout = VectorizationBase.vectorizable(out)
        plow = 0
        vA = VectorizationBase.vectorizable(A)
        for pmax ∈ 1:$(num_reps-1)
            $(mul_block(V, W, R1, R2, m_rep, N, piter, :plow))
            $(store_block(W, R1, m_rep, piter, :plow))
            plow += $piter
        end
    end
    plow = piter * (num_reps-1)
    prem = P - plow
    prem > 0 && push!(q.args, mul_block(V, W, R1, R2, m_rep, N, prem, plow))
    prem > 0 && push!(q.args, store_block(W, R1, m_rep, prem, plow))
    push!(q.args, :(output_data = out.data))
    q
end

@noinline function static_mul_nt_quote(M,N,P,T,R1,R2)
    L3 = R1 * P
    W = VectorizationBase.pick_vector_width(R1, T)
    m_rep = R1 ÷ W
    V = Vec{W,T}
    num_reps = cld(L3 ÷ W + 3, VectorizationBase.REGISTER_COUNT)
    if num_reps == 1
        outtup = Expr(:tuple)
        for p ∈ 1:P, mr ∈ 1:m_rep, m ∈ 1:W
            push!(outtup.args, :($(Symbol(:C_, mr, :_, p))[$m].value ))
        end
        return quote
            $(Expr(:meta, :inline))
            vA = VectorizationBase.vectorizable(A)
            Bparent = B.parent
            $(mul_block_nt(V, W, R1, R2, m_rep, N, P, 0, :vA, :Bparent))
            output_data = $outtup
        end
    end
    piter = cld(P, num_reps)
    q = quote
        $(Expr(:meta, :inline))
        out = FixedSizeMatrix{$M,$P,$T,$R1,$L3}(undef)
        vout = VectorizationBase.vectorizable(out)
        plow = 0
        Bparent = B.parent
        vA = VectorizationBase.vectorizable(A)
        for pmax ∈ 1:$(num_reps-1)
            $(mul_block_nt(V, W, R1, R2, m_rep, N, piter, :plow, :vA, :Bparent))
            $(store_block(W, R1, m_rep, piter, :plow))
            plow += $piter
        end
    end
    plow = piter * (num_reps-1)
    prem = P - plow
    prem > 0 && push!(q.args, mul_block_nt(V, W, R1, R2, m_rep, N, prem, plow, :vA, :Bparent))
    prem > 0 && push!(q.args, store_block(W, R1, m_rep, prem, plow))
    push!(q.args, :(output_data = out.data))
    q
end

@noinline function mulinit(
    kernel::DynamicKernel{T}, mask_expr::Union{Symbol,Unsigned} = 0x00, force_inline::Bool = false, mask_ops::Bool = true
) where {T}
    @unpack R, C, stride_X, X_transposed, negative = kernel
    W = VectorizationBase.pick_vector_width(R, T)
    V = Vec{W,T}
    size_T = sizeof(T)::Int
    q = force_inline ? quote $(Expr(:meta,:inline)) end : quote end
    Riter, Rrem = reps_and_rem(kernel)
    if mask_expr isa Symbol
        mask_ops = true
        Riterl = Riter - 1
    else
        mask_ops &= Rrem > 0
        Riterl = mask_ops ? Riter : Riter - 1
    end
    if negative
        for c ∈ 0:C-1, r ∈ 0:Riterl
            push!(q.args, :($(Symbol(:vD_,r,:_,c)) = SIMDPirates.vbroadcast($V, zero($T))))
        end
        return q
    end
    Xsym = :vX_0
    push!(q.args, :($Xsym = vbroadcast($V, pX)))
    for r ∈ 0:Riterl
        if mask_ops && r == Riterl
            push!(q.args, Expr(:(=), Symbol(:vA_,r), :(vload($V, pA + $size_T*$W*$r, $mask_expr))))
        else
            push!(q.args, Expr(:(=), Symbol(:vA_,r), :(vload($V, pA + $size_T*$W*$r))))
        end
    end
    # push!(q.args, :($Xsym = vbroadcast($V, pX)))
    for c ∈ 1:C
        for r ∈ 0:Riterl
            Dsym = Symbol(:vD_,r,:_,c-1); Asym = Symbol(:vA_,r)
            push!(q.args, :($Dsym = vmul($Asym, $Xsym)))
        end
        Xsym = Symbol(:vX_,c)
        c < C && push!(q.args, :($Xsym = vbroadcast($V, pX + $size_T*$(X_transposed ? c : :($c*$stride_X)  ) )))
    end
    q
end
@noinline function gemminit(
    kernel::DynamicKernel{T}, mask_expr::Union{Symbol,Unsigned} = 0x00, force_inline::Bool = false, mask_ops::Bool = true
) where {T}
    @unpack R, C, stride_D, stride_X = kernel
    W = VectorizationBase.pick_vector_width(R, T)
    V = Vec{W,T}
    size_T = sizeof(T)
    q = force_inline ? quote $(Expr(:meta,:inline)) end : quote end
    Riter, Rrem = reps_and_rem(kernel)
    if mask_expr isa Symbol
        mask_ops = true
        Riterl = Riter - 1
    else
        mask_ops &= Rrem > 0
        Riterl = mask_ops ? Riter : Riter - 1
    end
    for c ∈ 0:C-1
        for r ∈ 0:Riterl
            Dsym = Symbol(:vD_,r,:_,c)
            Dpointer = :(pD + $size_T * ($W*$r + $stride_D*$c))
            if mask_ops && r == Riterl && c == C - 1
                push!(q.args, :($Dsym = vload($V, $Dpointer, $mask_expr)))
            else
                push!(q.args, :($Dsym = vload($V, $Dpointer)))
            end
        end
    end
    q
end


# using Core.Intrinsics: llvmcall
# struct PrefetchA
    # A::Int
# end
# struct PrefetchX
    # X::Int
# end
# struct PrefetchAX
    # A::Int
    # X::Int
# end
# prefetch_A(::Any, ::Any, ::Any, ::Any)    = (nothing, nothing, nothing)
# prefetch_X(::Any, ::Any, ::Any, ::Any, ::Any)    = (nothing, nothing, nothing, nothing)
# function prefetch_A(::Type{PF}, N, Qₚ, AD_stride) where {PF <: Union{PrefetchA, PrefetchAX}}
    # (
        # :(@nexprs $Qₚ q -> prefetch(pA + (pf.A + $(VectorizationBase.CACHELINE_SIZE))*(q-1), Val(3), Val(0))),
        # :(@nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $(VectorizationBase.CACHELINE_SIZE)*(q-1), Val(3), Val(0))),
        # :(@nexprs $Qₚ q -> prefetch(pA + pf.A + $((N-1)*AD_stride) + $(VectorizationBase.CACHELINE_SIZE)*(q-1), Val(3), Val(0)))
    # )
# end
# function prefetch_X(::Type{PF}, N, Pₖ, X_stride, T_size) where {PF <: Union{PrefetchA, PrefetchAX}}
    # (
        # :(@nexprs $Pₖ p -> prefetch(pX + pf.X + (p-1)*$X_stride, Val(3), Val(0))),
        # :(@nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))),
        # :(prefetch(pX + pf.X + $(N*T_size) + (p-1)*$X_stride, Val(3), Val(0))),
        # :(prefetch(pX + pf.X + $(N*T_size + (Pₖ-1)*X_stride), Val(3), Val(0)))
    # )
# end



# Base.:+(ptr::Ptr, offset::Prefetch) = ptr + offset.offset
# Base.:+(offset::Prefetch, ptr::Ptr) = ptr + offset.offset

# args are address, read/write, locality, cache type

# """
# prefetch(address, Val(Locality), Val(ReadOrWrite))
# Locality gives locality of the prefetch.
# Read = 0, write = 1.

# From LLVM documentation:

# address is the address to be prefetched, rw is the specyifier
# determining if the fetch should be for a read (0) or write (1),
# and locality is a temporal locality specifier ranging
# from (0) - no locality, to (3) - extremely local keep in cache.
# The cache type specifies whether the prefetch is performed on
# the data (1) or instruction (0) cache. The rw, locality and
# cache type arguments must be constant integers.
# """
# @generated function prefetch(address, ::Val{Locality} = Val(1), ::Val{RorW} = Val(0)) where {Locality, RorW}
    # prefetch_call_string = """%addr = inttoptr i64 %0 to i8*
    # call void @llvm.prefetch(i8* %addr, i32 $RorW, i32 $Locality, i32 1)
    # ret void"""
    # quote
        # $(Expr(:meta, :inline))
        # llvmcall(("declare void @llvm.prefetch(i8* , i32 , i32 , i32 )",
        # $prefetch_call_string), Cvoid, Tuple{Ptr{Cvoid}}, address)
    # end
# end

"""

pD, pA, and pX must be defined as Ptr{T}.
Similarly, to use a runtime mask, it must be named `__mask__`, and the expression to define it must be placed somewhere.

"""
function kernel_quote(kernel::DynamicKernel{T}; init::Bool = true, force_inline::Bool = true, mask_ops::Bool = true, runtime_mask::Bool = false) where {T}
    @unpack R, C, N, stride_D, stride_A, stride_X, X_transposed, negative = kernel
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
    Wm1 = W - 1
    Riter = R >>> Wshift
    rem = R & Wm1
    if runtime_mask == true
        Riterl = Riter - 1
        mask_ops = true
        mask = :__mask__
    else
        Riterl = rem > 0 ? Riter : Riter - 1
        mask_ops &= (rem > 0)
        mask = VectorizationBase.mask(T, rem)
    end
    V = Vec{W,T}
    n = gensym(:n)
    A_load_expr = quote $([Expr(:(=), Symbol(:vA_,r), :(vload($V, pA + $size_T*($W*$r + $n*$stride_A)))) for r ∈ 0:Riterl]...) end
    if mask_ops
        A_load_expr_mask = quote $([Expr(:(=), Symbol(:vA_,r), :(vload($V, pA + $size_T*($W*$r + $n*$stride_A)))) for r ∈ 0:Riter]...) end
        push!(A_load_expr_mask.args, Expr(:(=), Symbol(:vA_,Riterl), :(vload($V, pA + $size_T*($W*$Riterl + $n*$stride_A),$mask))))
    else
        A_load_expr_mask = A_load_expr
    end
    D_store_expr = quote end
    for c ∈ 0:C-1
        for r ∈ 0:Riterl
            if mask_ops && r == Riterl
                push!(D_store_expr.args, :(vstore!(pD + $size_T*($W*$r + $stride_D*$c), $(Symbol(:vD_,r,:_,c)), $mask)))
            else
                push!(D_store_expr.args, :(vstore!(pD + $size_T*($W*$r + $stride_D*$c), $(Symbol(:vD_,r,:_,c)))))
            end
        end
    end
    # Not bothering with prefetching for now
    q = if init
        mulinit(kernel, mask, force_inline, mask_ops)
    else
        gemminit(kernel, mask, force_inline, mask_ops)
    end
    loop_iter_sub = mask_ops ? 2 : 1
    max_loop_iter = N isa Int ? N - loop_iter_sub : :($N - $loop_iter_sub)
    min_loop_iter = Int(init & !negative) # only 1 if we're using mulinit without calculating the negative product

    f = negative ? :vfnmadd : :vmuladd
    
    mul_q = quote end
    Xsym = :vX_0
    for c ∈ 1:C
        for r ∈ 0:Riterl
            Dsym = Symbol(:vD_,r,:_,c-1)
            Asym = Symbol(:vA_,r)
            push!(mul_q.args, :($Dsym = SIMDPirates.$f($Asym, $Xsym, $Dsym)))
        end
        if c < C
            Xsym = Symbol(:vX_,c)
            X_linear_index = X_transposed ? :($c + $n*$stride_X) : :($n + $c*$stride_X)
            push!(mul_q.args, :($Xsym = SIMDPirates.vbroadcast($V, pX + $size_T * ($X_linear_index))))
        end
    end
    loop_quote = quote
        for $n ∈ $min_loop_iter:$max_loop_iter
            vX_0 = SIMDPirates.vbroadcast($V, pX + $size_T * $(X_transposed ? :($n * $stride_X) : n))
            $A_load_expr
            # vX_0 = SIMDPirates.vbroadcast($V, pX + $size_T * $(X_transposed ? :($n * $stride_X) : n))
            $mul_q
        end
    end
    push!(q.args, loop_quote)
    if mask_ops
        final_n = N isa Int ? N - 1 : :($N-1)
        masked_iter_quote = quote
            $n = $final_n
            vX_0 = SIMDPirates.vbroadcast($V, pX + $size_T * $(X_transposed ? :($final_n * $stride_X) : final_n))
            $A_load_expr_mask
            $mul_q
        end
        push!(q.args, masked_iter_quote)
    end
    push!(q.args, D_store_expr)
    q
end

@noinline function kernel_tn_quote(
    kernel::DynamicKernel{T}, init::Bool,inline::Bool = false,
    Asym::Symbol = :pA, Xsym::Symbol = :pX, Dsym::Symbol = :pD,
    d_isa_ptr::Bool = true, contract::Bool = true
) where {T}
    @unpack R, C, N, stride_D, stride_A, stride_X, negative = kernel
    if N isa Symbol
        W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    else#if N isa Int
        W, Wshift = VectorizationBase.pick_vector_width_shift(N, T)
    end
    V = Vec{W,T}
    size_T = sizeof(T)
    loop_body = quote end
    n = gensym(:n)
    for r ∈ 0:R-1
        push!(loop_body.args, Expr(:(=), Symbol(:A_,r), :($Asym[$n + $stride_A*$r])))
    end
    f = negative ? :(SIMDPirates.vfnmadd) : :(SIMDPirates.vmuladd)
    for c ∈ 0:C-1
        col_sym = Symbol(:B_,c)
        push!(loop_body.args, Expr(:(=), col_sym, :($Xsym[$n + $stride_X*$c])))
        for r ∈ 0:R-1
            push!(loop_body.args, Expr(:(=),Symbol(:vD_,r,:_,c), Expr(:call,f, Symbol(:A_,r), col_sym, Symbol(:vD_,r,:_,c))))
        end
    end
    loop_quote = quote
        @vvectorize $T for $n ∈ 1:$N
            $loop_body
        end
    end
    if d_isa_ptr
        dptr = Dsym
    else
        dptr = gensym(Dsym)
    end
    # dptr = gensym(:dptr)
    init_q = if contract || init
        quote $([Expr(:(=),Symbol(:vD_,r,:_,c), :(SIMDPirates.vbroadcast($V,zero($T)))) for r ∈ 0:R-1, c ∈ 0:C-1]...) end
    else # we are not contracting and not initializing, meaning we should load
        quote $([Expr(:(=),Symbol(:vD_,r,:_,c), :(SIMDPirates.vload($V, $dptr + $size_T*($r*$W + $c*$stride_D)))) for r ∈ 0:R-1, c ∈ 0:C-1]...) end
    end
    q = quote
        $init_q
        $(macroexpand(LoopVectorization, loop_quote))
        # $loop_quote
    end
    if !d_isa_ptr
        pushfirst!(q.args, :($dptr = pointer($Dsym)))
    end
    assign_quote = quote end
    if contract
        if init
            for c ∈ 0:C-1, r ∈ 0:R-1
                push!(assign_quote.args, :(VectorizationBase.store!($dptr + $size_T*($r+$stride_D*$c), SIMDPirates.vsum($(Symbol(:vD_,r,:_,c))))))
            end
        else
            for c ∈ 0:C-1, r ∈ 0:R-1
                dptrexpr = :($dptr + $size_T*($r+$stride_D*$c))
                push!(assign_quote.args, :(VectorizationBase.store!($dptrexpr, VectorizationBase.load($dptrexpr) + SIMDPirates.vsum($(Symbol(:vD_,r,:_,c))))))
            end
        end
    else
        for c ∈ 0:C-1, r ∈ 0:R-1
            push!(assign_quote.args, :(VectorizationBase.vstore!($dptr + $size_T*($r*$W+$stride_D*$c), $(Symbol(:vD_,r,:_,c)))))
        end
    end
    push!(q.args, assign_quote)
    # push!(q.args, Dsym)
    inline && pushfirst!(q.args, Expr(:meta,:inline))
    q
end

@generated function kernel!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_A,stride_X,stride_D,N}, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,stride_A,stride_X,stride_D,N,T,negative}
    kernel = DynamicKernel{T}(
        Mₖ, Pₖ, N, stride_D, stride_A, stride_X, false, negative
    )
    kernel_quote(kernel, init = false, force_inline = false, mask_ops = true, runtime_mask = false)
end

@generated function initkernel!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_A,stride_X,stride_D,N}, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,stride_A,stride_X,stride_D,N,T,negative}
    kernel = DynamicKernel{T}(
        Mₖ, Pₖ, N, stride_D, stride_A, stride_X, false, negative
    )
    kernel_quote(kernel, init = true, force_inline = false, mask_ops = true, runtime_mask = false)
end
@generated function kernel_nt!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_A,stride_X,stride_D,N}, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,stride_A,stride_X,stride_D,N,T,negative}
    kernel = DynamicKernel{T}(
        Mₖ, Pₖ, N, stride_D, stride_A, stride_X, true, negative
    )
    kernel_quote(kernel, init = false, force_inline = false, mask_ops = true, runtime_mask = false)
end

@generated function initkernel_nt!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_A,stride_X,stride_D,N}, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,stride_A,stride_X,stride_D,N,T,negative}
    kernel = DynamicKernel{T}(
        Mₖ, Pₖ, N, stride_D, stride_A, stride_X, true, negative
    )
    kernel_quote(kernel, init = true, force_inline = false, mask_ops = true, runtime_mask = false)
end

@generated function initkernel_tn!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_A,stride_X,stride_D,N}, ::Val{negative} = Val{false}()
) where {T,Mₖ,Pₖ,stride_A,stride_X,stride_D,N,negative}
    kernel = DynamicKernel{T}(
        Mₖ, Pₖ, N, stride_D, stride_A, stride_X, false, negative
    )
    kernel_tn_quote(kernel, true, false)
end

@generated function kernel!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, kernel::DKernel{Mₖ,Pₖ}, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,T,negative}
    kernel = DynamicKernel{T}(
        Mₖ, Pₖ, N, stride_D, stride_A, stride_X, false, negative
    )
    kq = kernel_quote(kernel, init = false, force_inline = false, mask_ops = true, runtime_mask = false)
    quote
        @unpack N,stride_D,stride_A,stride_X = kernel
        $kq
    end
end

@generated function initkernel!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, kernel::DKernel{Mₖ,Pₖ}, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,T,negative}
    kernelq = DynamicKernel{T}(
        Mₖ, Pₖ, :N, :stride_D, :stride_A, :stride_X, false, negative
    )
    kq = kernel_quote(kernelq, init = true, force_inline = false, mask_ops = true, runtime_mask = false)
    quote
        @unpack N,stride_D,stride_A,stride_X = kernel
        $kq
    end
end
@generated function kernel_nt!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, kernel::DKernel{Mₖ,Pₖ}, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,T,negative}
    kernelq = DynamicKernel{T}(
        Mₖ, Pₖ, :N, :stride_D, :stride_A, :stride_X, true, negative
    )
    kq = kernel_quote(kernelq, init = false, force_inline = false, mask_ops = true, runtime_mask = false)
    quote
        @unpack N,stride_D,stride_A,stride_X = kernel
        $kq
    end
end

@generated function initkernel_nt!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, kernel::DKernel{Mₖ,Pₖ}, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,T,negative}
    kernelq = DynamicKernel{T}(
        Mₖ, Pₖ, :N, :stride_D, :stride_A, :stride_X, true, negative
    )
    kq = kernel_quote(kernelq, init = true, force_inline = false, mask_ops = true, runtime_mask = false)
    quote
        @unpack N,stride_D,stride_A,stride_X = kernel
        $kq
    end
end
@generated function kernel!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, kernel::DKernel{Mₖ,Pₖ}, __mask__::Unsigned, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,T,negative}
    kernelq = DynamicKernel{T}(
        Mₖ, Pₖ, N, stride_D, stride_A, stride_X, false, negative
    )
    kq = kernel_quote(kernelq, init = false, force_inline = false, mask_ops = true, runtime_mask = true)
    quote
        @unpack N,stride_D,stride_A,stride_X = kernel
        $kq
    end
end

@generated function initkernel!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, kernel::DKernel{Mₖ,Pₖ}, __mask__::Unsigned, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,T,negative}
    kernelq = DynamicKernel{T}(
        Mₖ, Pₖ, :N, :stride_D, :stride_A, :stride_X, false, negative
    )
    kq = kernel_quote(kernelq, init = true, force_inline = false, mask_ops = true, runtime_mask = true)
    quote
        @unpack N,stride_D,stride_A,stride_X = kernel
        $kq
    end
end
@generated function kernel_nt!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, kernel::DKernel{Mₖ,Pₖ}, __mask__::Unsigned, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,T,negative}
    kernelq = DynamicKernel{T}(
        Mₖ, Pₖ, :N, :stride_D, :stride_A, :stride_X, true, negative
    )
    kq = kernel_quote(kernelq, init = false, force_inline = false, mask_ops = true, runtime_mask = true)
    quote
        @unpack N,stride_D,stride_A,stride_X = kernel
        $kq
    end
end

@generated function initkernel_nt!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, kernel::DKernel{Mₖ,Pₖ}, __mask__::Unsigned, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,T,negative}
    kernelq = DynamicKernel{T}(
        Mₖ, Pₖ, :N, :stride_D, :stride_A, :stride_X, true, negative
    )
    kq = kernel_quote(kernelq, init = true, force_inline = false, mask_ops = true, runtime_mask = true)
    quote
        @unpack N,stride_D,stride_A,stride_X = kernel
        $kq
    end
end
@generated function initkernel_tn!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, kernel::DKernel{Mₖ,Pₖ}, ::Val{negative} = Val{false}()
) where {T,Mₖ,Pₖ,negative}
    kernel = DynamicKernel{T}(
        Mₖ, Pₖ, :N, :stride_D, :stride_A, :stride_X, false, negative
    )
    quote
        @unpack N,stride_D,stride_A,stride_X = kernel
        $(kernel_tn_quote(kernel, true, false))
    end
end



@generated function inline_kernel!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_A,stride_X,stride_D,N}, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,stride_A,stride_X,stride_D,N,T,negative}
    kernel = DynamicKernel{T}(
        Mₖ, Pₖ, N, stride_D, stride_A, stride_X, false, negative
    )
    kernel_quote(kernel, init = false, force_inline = true, mask_ops = true, runtime_mask = false)
end

@generated function inline_initkernel!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_A,stride_X,stride_D,N}, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,stride_A,stride_X,stride_D,N,T,negative}
    kernel = DynamicKernel{T}(
        Mₖ, Pₖ, N, stride_D, stride_A, stride_X, false, negative
    )
    kernel_quote(kernel, init = true, force_inline = true, mask_ops = true, runtime_mask = false)
end
@generated function inline_kernel_nt!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_A,stride_X,stride_D,N}, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,stride_A,stride_X,stride_D,N,T,negative}
    kernel = DynamicKernel{T}(
        Mₖ, Pₖ, N, stride_D, stride_A, stride_X, true, negative
    )
    kernel_quote(kernel, init = false, force_inline = true, mask_ops = true, runtime_mask = false)
end

@generated function inline_initkernel_nt!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_A,stride_X,stride_D,N}, ::Val{negative} = Val{false}()
) where {Mₖ,Pₖ,stride_A,stride_X,stride_D,N,T,negative}
    kernel = DynamicKernel{T}(
        Mₖ, Pₖ, N, stride_D, stride_A, stride_X, true, negative
    )
    kernel_quote(kernel, init = true, force_inline = true, mask_ops = true, runtime_mask = false)
end

@generated function inline_initkernel_tn!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_A,stride_X,stride_D,N}, ::Val{negative} = Val{false}()
) where {T,Mₖ,Pₖ,stride_A,stride_X,stride_D,N,negative}
    kernel = DynamicKernel{T}(
        Mₖ, Pₖ, N, stride_D, stride_A, stride_X, false, negative
    )
    kernel_tn_quote(kernel, true, true)
end




# """
# num_cols = elements_per_register * ( REGISTER_COUNT - 2 ) / num_rows - 1
# # square, assume approx equal
# num_rows = elements_per_register * ( REGISTER_COUNT - 2 ) / num_rows - 1
# 0 = num_rows^2 + num_rows - elements_per_register * ( REGISTER_COUNT - 2 )
#
#
# Returns number of elements per register,
# and the number of rows and columns of the kernel.
# Ie, 8, 16, 14 means that 8 elements fit per register,
# and the optimal kernel is
# 16x14 = 16xN * Nx14
# matrix multiplication.
#
# Rather than all the noise above, we just pick something close to square because:
# (L1_cache_size - rows*colums) ÷ (rows + columns)
# will be maximized with relatively square blocks, making that friendliest for the cache.
# """
@noinline function pick_kernel_size(
    ::Type{T}, row_max::Int = typemax(Int), col_max::Int = typemax(Int);
    W::Int = VectorizationBase.pick_vector_width(row_max, T),
    NREG::Int = VectorizationBase.REGISTER_COUNT, verbose::Bool = false,
    max_aloads::Int = min(NREG >>> 1, row_max)
) where {T}
    T_size = sizeof(T)
    # @show max_aloads
    # cache_line = W
    # cache_line = CACHELINE_SIZE ÷ T_size
    max_total = W * NREG
    # num_cache_lines = cld(max_total, cache_line)
    prev_num_rows, prev_num_cols = 0, 0
    prev_ratio = -Inf
    rows = Vector{Int}(undef, max_aloads)
    cols = Vector{Int}(undef, max_aloads)
    ratios = Vector{Float64}(undef, max_aloads)
    for a_loads ∈ 1:max_aloads
        num_rows = a_loads * W
        num_cols = (NREG - a_loads - 1) ÷ a_loads # assumes we need only a single B
        num_cols = min(num_cols, col_max)
        if num_cols == 0
            ratios[a_loads:end] .= 0
            break
        end
        next_ratio_fc_fr = (num_rows * num_cols) / (num_cols + a_loads)
        if col_max != typemax(Int)
            nfullcol, remcol = divrem(col_max, num_cols)
            if remcol == 0
                next_ratio_rc_fr = next_ratio_fc_fr
                rem_col_indicator = 0
            else
                next_ratio_rc_fr = (num_rows * remcol) / (remcol + a_loads)
                rem_col_indicator = 1
            end
        else
            next_ratio_rc_fr = next_ratio_fc_fr
            nfullcol, remcol = 1, 0
            rem_col_indicator = 0
        end
        if row_max != typemax(Int)
            nfullrow, remrow = divrem(row_max, num_rows)
            if remrow == 0
                next_ratio_fc_rr = next_ratio_fc_fr
                rem_row_indicator = 0
            else
                next_ratio_fc_rr = (remrow * num_cols) / (num_cols + cld(remrow, W))
                rem_row_indicator = 1
            end
        else
            next_ratio_fc_rr = next_ratio_fc_fr
            nfullrow, remrow = 1, 0
            rem_row_indicator = 0
        end
        nrcrr = rem_col_indicator * rem_row_indicator
        if nrcrr != 0
            next_ratio_rc_rr = (remrow * remcol) / (remcol + cld(remrow, W))
        else
            next_ratio_rc_rr = next_ratio_fc_fr
        end
        nfcfr = nfullcol * nfullrow
        nfcrr = nfullcol * rem_row_indicator
        nrcfr = rem_col_indicator * nfullrow
        next_ratio = (next_ratio_fc_fr*nfcfr + next_ratio_fc_rr*nfcrr + next_ratio_rc_fr*nrcfr + next_ratio_rc_rr*nrcrr) / (nfcfr + nfcrr + nrcfr + nrcrr)
        rows[a_loads] = num_rows
        cols[a_loads] = num_cols
        ratios[a_loads] = next_ratio
        verbose && @show a_loads, num_rows, num_cols, next_ratio
        if num_rows >= row_max
            ratios[a_loads+1:end] .= 0
            break
        end
    end
    max_ratio, max_ind = findmax(ratios)
    W, rows[max_ind], cols[max_ind], max_ind
end
# function pick_kernel_size(::Type{T}; D_count = 1, A_count = 1, X_count = 1) where {T}
#     W = VectorizationBase.pick_vector_width(T)
#     square = sqrt( (VectorizationBase.REGISTER_COUNT - 1) / W )
#
#     vloads_per_row = round(Int, square, RoundUp)
#     number_of_rows = W * vloads_per_row
#     number_of_columns = (VectorizationBase.REGISTER_COUNT - 1 - vloads_per_row) ÷ vloads_per_row
#
#     W, number_of_rows, number_of_columns
# end

"""
Given matrices of size M, N, and P...
This matrix assumes 3 cache levels. This means it is not especially portable, beyond x86_64.
Level 1 and 2 is assumed to be local to each core, while level 3 is assumed shared.

How do I want it to work? Each level should divide well into the next. Means for one thing
I probably want to iterate over cache_size backwards?

Goal is to minimize actual data movement that goes on.
D (+)= A * X
Looking at the extreme of calculating a single kernel from D in full at a time,
we see that it
a) Minimizes loading and unloading D from registers.
    1) Does this maximize kernel performance? Kernels are then (m x N) * (N x p).
b)

Should add support for how many copies of each of the matrices we have, so that we can perform calculations such as
    D = A*X + C
    or
    D = A*(X + C)
in one step, without as much un/reloading of memory.
"""
@noinline function blocking_structure(M::Int, N::Int, P::Int, ::Type{T} = Float64;
        cache_size::NTuple{3,Int} = VectorizationBase.CACHE_SIZE) where T
    total_elements = M*N + N*P + M*P
    L1, L2, L3 = cache_size .÷ sizeof(T)
    if L1 > total_elements
        epr, m_1, p_1 = pick_kernel_size(T, M, P)
        return ((min(m_1,M),N,min(p_1,P)),(M,N,P),(M,N,P)),0
    end

    epr, m_1, p_1 = pick_kernel_size(T, M, P)
    Dmp_1 = m_1 * p_1

    n_1 = (L1 - Dmp_1) ÷ (m_1 + p_1)

    # I need to consider the case where
    # m_1 or p_1 can be some multiple of themselves due to N being too small.
    if n_1 > N
        n_2 = n_1 = N
    else
        n_2 = n_1
    end

    if L2 > total_elements
        return ((m_1, n_1, p_1), (M, N, P), (M, N, P)),1
    end

    # Try to upper bound size of m_2, p_2
    # Consider safety factors, for other things (instructions?) in this cache?
    m_2, p_2 = divide_into_rough_square(L2, M, P, n_2, m_1, p_1)#, D_count = D_count, A_count = A_count, X_count = X_count)

    if L3 > total_elements
        return ((m_1, n_1, p_1), (m_2, n_2, p_2), (M, N, P)),2
    end

    Dmp_2 = m_2 * p_2
    n_3 = (L3 - Dmp_2) ÷ (m_2 + p_2)
    if n_3 > N
        n_3 = N
        m_3, p_3 = divide_into_rough_square(L3, M, P, N, m_2, p_2)#, D_count = D_count, A_count = A_count, X_count = X_count)
    else
        m_3, p_3 = m_2, p_2
    end

    # (Base.Cartesian.@ntuple 3 i -> (m_i, n_i, p_i)),3
    ((m_1,n_1,p_1),(m_2,n_2,p_2),(m_3,n_3,p_3)),3
end

@noinline function round_x_to_nearest_y(x::T, y::T) where {T}
    z = x/y
    round(T, z) * y
end

@noinline function divide_into_rough_square(L, M, P, n, mbase, pbase)
    Mhalf = max(round_x_to_nearest_y(M>>>1, mbase), mbase)
    Phalf = max(round_x_to_nearest_y(P>>>1, pbase), pbase)
    bsize = Mhalf*Phalf + Mhalf*n + n*Phalf
    if bsize < L
        return Mhalf, Phalf
    end
    L_upper_bound = floor(Int, sqrt(abs2(n) + L) - n)
    m_2 = max(round_x_to_nearest_y(L_upper_bound, mbase), mbase)
    if m_2 >= M
        m_2 = M
        p_2 = min(P, (L - m_2*n) ÷ (m_2 + n) )
    else
        p_2 = L_upper_bound ÷ pbase * pbase
        if p_2 >= P
            p_2 = P
            m_2 = min(M, (L - p_2*n) ÷ (p_2 + n) )
        end
    end
    m_2, p_2
end

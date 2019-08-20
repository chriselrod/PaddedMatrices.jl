"""
Base.@pure Kernel(Mₖ,N,Pₖ,stride_AD,stride_X) = Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}()

The kernel is typed based on M and P. M is a critical component of vector length,
while the kernel is unrolled across P. It simply loops over N.
"""
struct Kernel{Mₖ,Pₖ,stride_AD,stride_X,N} end
Base.@pure Kernel(Mₖ,N,Pₖ,stride_AD,stride_X) = Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}()
struct DKernel{Mₖ,Pₖ,stride_AD,stride_X} N::Int end
Base.@pure DKernel(Mₖ,N,Pₖ,stride_AD,stride_X) = Kernel{Mₖ,Pₖ,stride_AD,stride_X}(N)

@inline tuple_join(x) = x
@inline tuple_join(x, y) = (x..., y...)
@inline tuple_join(x, y, z...) = (x..., tuple_join(y, z...)...)
@generated function to_tuple(x::NTuple{N,Core.VecElement{T}}) where {N,T}
    quote
        $(Expr(:meta,:inline))
        @inbounds $(Expr(:tuple, [:(x[$n].value) for n ∈ 1:N]...))
    end
end
@inline to_tuple(x::NTuple{N}) where {N} = x
@inline extract_value(x::Core.VecElement{T}) where {T} = x.value
@inline extract_value(x) = x
@inline to_tuple2(x::NTuple{N,Core.VecElement{T}}) where {N,T} = extract_value.(x)

function mul_block(V, W, R1, R2, m_rep, N, P, poffset::Int = 0, vA = :vA, B = :B, gemm = nothing)
    Prange = (1 + poffset):(P + poffset)
    loop_max = isa(N, Number) ? N - 1 : :($N - 1)
    if gemm == nothing
        loop_min = 1
        initialize = quote
            $([:(
                $(Symbol(:Acol_,mr)) = SIMDPirates.vload($V, $vA + $(W*(mr-1)))
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [ :(@inbounds $(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmul(
                $(Symbol(:Acol_,mr)), $B[ $(1 + (p-1)*R2) ])
            ) for mr ∈ 1:m_rep]...) for p ∈ Prange]...)
        end
    else
        loop_min = 0
        R3 = gemm
        initialize = quote
            $([Expr(:block, [ :($(Symbol(:C_, mr+1, :_, p)) = vload($V, vout + $(W*mr + (p-1)*R3))) for mr ∈ 0:m_rep-1]...) for p ∈ Prange]...)
        end
    end
    quote
        $initialize
        @inbounds for n ∈ $loop_min:$loop_max
            $([:(
                $(Symbol(:Acol_,mr)) = SIMDPirates.vload($V, $vA + $(W*(mr-1)) + n*$R1)
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [:($(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmuladd(
                $(Symbol(:Acol_,mr)), $B[n + $(1 + (p-1)*R2)],
                $(Symbol(:C_, mr, :_, p)) )) for mr ∈ 1:m_rep]...) for p ∈ Prange]...
            )
        end
    end
end
function mul_block(V, W, R1, R2, m_rep, N, P, poffset::Symbol, vA = :vA, B = :B, gemm = nothing)
    Prange = 1:P
    loop_max = isa(N, Number) ? N - 1 : :($N - 1)
    if gemm == nothing
        loop_min = 1
        initialize = quote
            $([:(
                $(Symbol(:Acol_,mr)) = SIMDPirates.vload($V, $vA + $(W*(mr-1)))
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [ :(@inbounds $(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmul(
                $(Symbol(:Acol_,mr)), $B[ 1 + ($(p-1)+$poffset)*$R2 ])
            ) for mr ∈ 1:m_rep]...) for p ∈ Prange]...)
        end
    else
        loop_min = 0
        R3 = gemm
        initialize = quote
            $([Expr(:block, [ :($(Symbol(:C_, mr+1, :_, p)) = vload($V, vout + $(W*mr) + ($(p-1)+$poffset)*$R3)) for mr ∈ 0:m_rep-1]...) for p ∈ Prange]...)
        end
    end
    quote
        $initialize
        @inbounds for n ∈ $loop_min:$loop_max
            $([:(
                $(Symbol(:Acol_,mr)) = vload($V, $vA + $(W*(mr-1)) + n*$R1)
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
    t - tranposed

Meaning the operation is
A * B′
ie, A is not transposed, and B is tranposed.
"""
function mul_block_nt(V, W, R1, R2, m_rep, N, P, poffset::Int = 0, vA = :vA, B = :B, gemm = nothing)
    Prange = (1 + poffset):(P + poffset)
    loop_max = isa(N, Number) ? N - 1 : :($N - 1)
    if gemm == nothing
        loop_min = 1
        initialize = quote
            $([:(
                $(Symbol(:Acol_,mr)) = SIMDPirates.vload($V, $vA + $(W*(mr-1)))
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [ :(@inbounds $(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmul(
                $(Symbol(:Acol_,mr)), $B[ $p ])
                ) for mr ∈ 1:m_rep]...) for p ∈ Prange]...)
        end
    else
        loop_min = 0
        R3 = gemm
        initialize = quote
            $([Expr(:block, [ :($(Symbol(:C_, mr+1, :_, p)) = vload($V, vout + $(W*mr + (p-1)*R3))) for mr ∈ 0:m_rep-1]...) for p ∈ Prange]...)
        end
    end
    quote
        $initialize
        @inbounds for n ∈ $loop_min:$loop_max
            $([:(
                $(Symbol(:Acol_,mr)) = SIMDPirates.vload($V, $vA + $(W*(mr-1)) + n*$R1)
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [:($(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmuladd(
                $(Symbol(:Acol_,mr)), $B[n*$R2 + $p],
                $(Symbol(:C_, mr, :_, p)) )) for mr ∈ 1:m_rep]...) for p ∈ Prange]...
            )
        end
    end
end
function mul_block_nt(V, W, R1, R2, m_rep, N, P, poffset::Symbol, vA = :vA, B = :B, gemm = nothing)
    Prange = 1:P
    loop_max = isa(N, Number) ? N - 1 : :($N - 1)
    if gemm == nothing
        loop_min = 1
        initialize = quote
            $([:(
                $(Symbol(:Acol_,mr)) = SIMDPirates.vload($V, $vA + $(W*(mr-1)) )
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [ :(@inbounds $(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmul(
                $(Symbol(:Acol_,mr)), $B[ $p + $poffset ])
            ) for mr ∈ 1:m_rep]...) for p ∈ Prange]...)
        end
    else
        loop_min = 0
        R3 = gemm
        initialize = quote
            $([Expr(:block, [ :($(Symbol(:C_, mr+1, :_, p)) = vload($V, vout + $(W*mr) + ($(p-1)+$poffset)*$R3)) for mr ∈ 0:m_rep-1]...) for p ∈ Prange]...)
        end
    end
    quote
        $initialize
        @inbounds for n ∈ $loop_min:$loop_max
            $([:(
                $(Symbol(:Acol_,mr)) = SIMDPirates.vload($V, $vA + $(W*(mr-1)) + n*$R1)
            ) for mr ∈ 1:m_rep]...)
            $([Expr(:block, [:($(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmuladd(
                $(Symbol(:Acol_,mr)), $B[n*$R2 + $p+$poffset],
                $(Symbol(:C_, mr, :_, p)) )) for mr ∈ 1:m_rep]...) for p ∈ Prange]...
            )
        end
    end
end
# function mul_block_right_symmetric(sub2ind, W, R1, R2, m_rep, N, P, poffset::Int = 0)
#     Prange = (1 + poffset):(P + poffset)
#     quote
#         $([:(
#             $(Symbol(:Acol_,mr)) = @inbounds $(
#                 Expr(:tuple, [:(Core.VecElement(A[ $(m + (mr-1)*W) ])) for m ∈ 1:W]...)
#             )
#         ) for mr ∈ 1:m_rep]...)
#         $(
#             [Expr(:block,
#             [ :($(Symbol(:C_, mr, :_, p)) = SIMDPirates.evmul(
#                 $(Symbol(:Acol_,mr)), @inbounds B[ $(1 + (p-1)*R2) ])
#             ) for mr ∈ 1:m_rep]...) for p ∈ Prange]...
#         )
#         @inbounds for n ∈ 1:$(N-1)
#             $([:(
#                 $(Symbol(:Acol_,mr)) = @inbounds $(Expr(:tuple,
#                 [:(Core.VecElement(A[ $(m + (mr-1)*W) + n*$R1 ])) for m ∈ 1:W]...))
#             ) for mr ∈ 1:m_rep]...)
#             $([Expr(:block, [:($(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmuladd(
#                 $(Symbol(:Acol_,mr)), B[n + $(1 + (p-1)*R2)],
#                 $(Symbol(:C_, mr, :_, p)) )) for mr ∈ 1:m_rep]...) for p ∈ Prange]...
#             )
#         end
#     end
# end
# function mul_block_right_symmetric(sub2ind, W, R1, R2, m_rep, N, P, poffset::Symbol)
#     Prange = 1:P
#     quote
#         $([:(
#             $(Symbol(:Acol_,mr)) = @inbounds $(Expr(:tuple,
#             [:(Core.VecElement(A[ $(m + (mr-1)*W) ])) for m ∈ 1:W]...)
#         )
#         ) for mr ∈ 1:m_rep]...)
#         $([Expr(:block, [ :($(Symbol(:C_, mr, :_, p)) = SIMDPirates.evmul(
#             $(Symbol(:Acol_,mr)), @inbounds B[ 1 + ($(p-1)+$poffset)*$R2 ])
#             ) for mr ∈ 1:m_rep]...) for p ∈ Prange]...)
#         @inbounds for n ∈ 1:$(N-1)
#             $([:(
#                 $(Symbol(:Acol_,mr)) = @inbounds $(Expr(:tuple,
#                     [:(Core.VecElement(A[ $(m + (mr-1)*W) + n*$R1 ])) for m ∈ 1:W]...))
#             ) for mr ∈ 1:m_rep]...)
#             $([Expr(:block, [:($(Symbol(:C_, mr, :_, p)) = SIMDPirates.vmuladd(
#                 $(Symbol(:Acol_,mr)), B[n + 1 + ($(p-1)+$poffset)*$R2],
#                 $(Symbol(:C_, mr, :_, p)) )) for mr ∈ 1:m_rep]...) for p ∈ Prange]...
#             )
#         end
#     end
# end
function store_block(W, R1, m_rep, P, poffset::Int = 0)
    Prange = (1 + poffset):(P + poffset)
    q = quote end
    for p ∈ Prange, mr ∈ 1:m_rep
        push!(q.args, :(vstore!(vout + $((p-1)*R1 + (mr-1)*W), $(Symbol(:C_, mr, :_, p)))))
    end
    q
end
function store_block(W, R1, m_rep, P, poffset::Symbol)
    Prange = 1:P
    q = quote end
    for p ∈ Prange, mr ∈ 1:m_rep
        push!(q.args, :(vstore!(vout + ($poffset + $(p-1))*$R1 + $((mr-1)*W), $(Symbol(:C_, mr, :_, p)))))
    end
    q
end

function static_mul_quote(M,N,P,T,R1,R2)

    L3 = R1 * P
    W = VectorizationBase.pick_vector_width(R1, T)
    m_rep = R1 ÷ W
    V = Vec{W,T}
    num_reps = cld(L3 ÷ W + 3, VectorizationBase.REGISTER_COUNT)
    if num_reps == 1
        outtup = Expr(:tuple)
        # outtup = Expr(:call, :tuple_join)
        for p ∈ 1:P, mr ∈ 1:m_rep, m ∈ 1:W
            # push!(outtup.args, :($(Symbol(:C_, mr, :_, p))[$m] ))
            push!(outtup.args, :($(Symbol(:C_, mr, :_, p))[$m].value ))
            # push!(outtup.args, :($(Symbol(:C_, mr, :_, p))))
        end
        return quote
            $(Expr(:meta, :inline))
            vA = VectorizationBase.vectorizable(A)
            $(mul_block(V, W, R1, R2, m_rep, N, P))
            # ConstantFixedSizePaddedMatrix{$M,$P,$T,$R1,$L3}(
            #     $outtup
            # )
            output_data = $outtup
        end
    end

    piter = cld(P, num_reps)
    q = quote
        $(Expr(:meta, :inline))
        out = MutableFixedSizePaddedMatrix{$M,$P,$T,$R1,$L3}(undef)
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
    # push!(q.args,  :(ConstantFixedSizePaddedMatrix( out )) )
    push!(q.args, :(output_data = out.data))
    q
end

function static_mul_nt_quote(M,N,P,T,R1,R2)

    L3 = R1 * P
    W = VectorizationBase.pick_vector_width(R1, T)
    m_rep = R1 ÷ W
    V = Vec{W,T}
    num_reps = cld(L3 ÷ W + 3, VectorizationBase.REGISTER_COUNT)
    if num_reps == 1
        outtup = Expr(:tuple)
        # outtup = Expr(:call, :tuple_join)
        for p ∈ 1:P, mr ∈ 1:m_rep, m ∈ 1:W
            # push!(outtup.args, :($(Symbol(:C_, mr, :_, p))[$m] ))
            push!(outtup.args, :($(Symbol(:C_, mr, :_, p))[$m].value ))
            # push!(outtup.args, :($(Symbol(:C_, mr, :_, p))))
        end
        return quote
            $(Expr(:meta, :inline))
            vA = VectorizationBase.vectorizable(A)
            Bparent = B.parent
            $(mul_block_nt(V, W, R1, R2, m_rep, N, P, 0, :vA, :Bparent))
            # ConstantFixedSizePaddedMatrix{$M,$P,$T,$R1,$L3}(
            #     $outtup
            # )
            output_data = $outtup
        end
    end

    piter = cld(P, num_reps)
    q = quote
        $(Expr(:meta, :inline))
        out = MutableFixedSizePaddedMatrix{$M,$P,$T,$R1,$L3}(undef)
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
    # push!(q.args,  :(ConstantFixedSizePaddedMatrix( out )) )
    push!(q.args, :(output_data = out.data))
    q
end


# function static_by_sym_mul_quote(M,P,T,R1,R2)
#
#     L3 = R1 * P
#     W = VectorizationBase.pick_vector_width(R1, T)
#     m_rep = R1 ÷ W
#     outtup = Expr(:tuple)
#     # outtup = Expr(:call, :tuple_join)
#     for p ∈ 1:P, mr ∈ 1:m_rep, m ∈ 1:W
#         # push!(outtup.args, :($(Symbol(:C_, mr, :_, p))[$m] ))
#         push!(outtup.args, :($(Symbol(:C_, mr, :_, p))[$m].value ))
#         # push!(outtup.args, :($(Symbol(:C_, mr, :_, p))))
#     end
#
#     num_reps = cld(L3 ÷ W + 3, VectorizationBase.REGISTER_COUNT)
#     if num_reps == 1
#         return quote
#             $(Expr(:meta, :inline))
#             $(mul_block(W, R1, R2, m_rep, N, P))
#             ConstantFixedSizePaddedMatrix{$M,$P,$T,$R1,$L3}(
#                 $outtup
#             )
#         end
#     end
#     piter = cld(P, num_reps)
#     q = quote
#         $(Expr(:meta, :inline))
#         out = MutableFixedSizePaddedMatrix{$M,$P,$T,$R1,$L3}(undef)
#         vout = VectorizationBase.vectorizable(out)
#         plow = 0
#         for pmax ∈ 1:$(num_reps-1)
#             $(mul_block(W, R1, R2, m_rep, N, piter, :plow))
#             $(store_block(W, R1, m_rep, piter, :plow))
#             plow += $piter
#         end
#     end
#     plow = piter * (num_reps-1)
#     prem = P - plow
#     prem > 0 && push!(q.args, mul_block(W, R1, R2, m_rep, N, prem, plow))
#     prem > 0 && push!(q.args, store_block(W, R1, m_rep, prem, plow))
#     push!(q.args,  :(ConstantFixedSizePaddedMatrix( out )) )
#     q
# end

function mulinit(V, WT, Q, Pₖ, X_stride, r, mask_expr, inline_expr, pfA_1, X_transposed = false)
    q_load_expr = :(@nexprs $Q q -> vA_q = vload($V, pA + $WT*(q-1)))
    if X_transposed
        X_load_expr = :(pX + (p-1))
    else
        X_load_expr = :(pX + (p-1)*$X_stride)
    end
    q = quote
        $q_load_expr
        @nexprs $Pₖ p -> begin
            vX = vbroadcast($V, VectorizationBase.load($X_load_expr))
            @nexprs $Q q -> Dx_q_p = SIMDPirates.vmul(vA_q, vX)
        end
    end
    inline_expr == :nothing || pushfirst!(q.args, inline_expr)
    pfA_1 == :nothing || push!(q.args, pfA_1)
    q

end
function gemminit(V, WT, Q, Pₖ, AD_stride, r, mask_expr, inline_expr, X_transposed = false)
    if r == 0
        expr = quote
            @nexprs $Pₖ p -> @nexprs $Q q -> Dx_q_p = vload($V, pD + $WT*(q-1) + $AD_stride*(p-1))
        end
    else
        expr = quote
            @nexprs $(Pₖ-1) p -> @nexprs $Q q -> Dx_q_p = vload($V, pD + $WT*(q-1) + $AD_stride*(p-1))
        end
        for q ∈ 1:Q-1
            push!(expr.args, :($(Symbol(:Dx_,q,:_,Pₖ)) = vload($V, pD + $(WT*(q-1) + AD_stride*(Pₖ-1)))))
        end
        push!(expr.args, :($(Symbol(:Dx_,Q,:_,Pₖ)) = vload($V, pD + $(WT*(Q-1) + AD_stride*(Pₖ-1)),$mask_expr)))
    end
    inline_expr == :nothing || pushfirst!(expr.args, inline_expr)
    expr
end
function create_mask(W, r)
    if W <= 8
        mask = :(UInt8($(UInt8(2)^r-UInt8(1))))
    elseif W <= 16
        mask = :(UInt16($(UInt16(2)^r-UInt16(1))))
    elseif W <= 32
        mask = :(UInt32($(UInt32(2)^r-UInt32(1))))
    elseif W <= 64
        mask = :(UInt64($(UInt64(2)^r-UInt64(1))))
    else #W <= 128
        mask = :(UInt128($(UInt128(2)^r-UInt128(1))))
    end
    mask
end
using Core.Intrinsics: llvmcall

struct PrefetchA
    A::Int
end
struct PrefetchX
    X::Int
end
struct PrefetchAX
    A::Int
    X::Int
end
prefetch_A(::Any, ::Any, ::Any, ::Any)    = (nothing, nothing, nothing)
prefetch_X(::Any, ::Any, ::Any, ::Any, ::Any)    = (nothing, nothing, nothing, nothing)
function prefetch_A(::Type{PF}, N, Qₚ, AD_stride) where {PF <: Union{PrefetchA, PrefetchAX}}
    (
        :(@nexprs $Qₚ q -> prefetch(pA + $(pf.A + VectorizationBase.CACHELINE_SIZE)*(q-1), Val(3), Val(0))),
        :(@nexprs $Qₚ q -> prefetch(pA + $(pf.A) + n*$AD_stride + $(VectorizationBase.CACHELINE_SIZE)*(q-1), Val(3), Val(0))),
        :(@nexprs $Qₚ q -> prefetch(pA + $(pf.A + (N-1)*AD_stride) + $(VectorizationBase.CACHELINE_SIZE)*(q-1), Val(3), Val(0)))
    )
end
function prefetch_X(::Type{PF}, N, Pₖ, X_stride, T_size) where {PF <: Union{PrefetchA, PrefetchAX}}
    (
        :(@nexprs $Pₖ p -> prefetch(pX + $(pf.X) + (p-1)*$X_stride, Val(3), Val(0))),
        :(@nexprs $Pₖ p -> prefetch(pX + $(pf.X) + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))),
        :(prefetch(pX + $(pf.X + N*T_size) + (p-1)*$X_stride, Val(3), Val(0))),
        :(prefetch(pX + $(pf.X + N*T_size + (Pₖ-1)*X_stride), Val(3), Val(0)))
    )
end

# Base.:+(ptr::Ptr, offset::Prefetch) = ptr + offset.offset
# Base.:+(offset::Prefetch, ptr::Ptr) = ptr + offset.offset

# args are address, read/write, locality, cache type

"""
prefetch(address, Val(Locality), Val(ReadOrWrite))
Locality gives locality of the prefetch.
Read = 0, write = 1.

From LLVM documentation:

address is the address to be prefetched, rw is the specifier
determining if the fetch should be for a read (0) or write (1),
and locality is a temporal locality specifier ranging
from (0) - no locality, to (3) - extremely local keep in cache.
The cache type specifies whether the prefetch is performed on
the data (1) or instruction (0) cache. The rw, locality and
cache type arguments must be constant integers.
"""
@generated function prefetch(address, ::Val{Locality} = Val(1), ::Val{RorW} = Val(0)) where {Locality, RorW}
    prefetch_call_string = """%addr = inttoptr i64 %0 to i8*
    call void @llvm.prefetch(i8* %addr, i32 $RorW, i32 $Locality, i32 1)
    ret void"""
    quote
        $(Expr(:meta, :inline))
        llvmcall(("declare void @llvm.prefetch(i8* , i32 , i32 , i32 )",
        $prefetch_call_string), Cvoid, Tuple{Ptr{Cvoid}}, address)
    end
end

function kernel_quote(Mₖ,Pₖ,stride_A,stride_X,N,T,init,inline = true, pf = nothing, stride_D = stride_A)
    T_size = sizeof(T)
    A_stride = stride_A * T_size
    D_stride = stride_D * T_size
    X_stride = stride_X * T_size
    W = VectorizationBase.REGISTER_SIZE ÷ T_size
    while W >= 2Mₖ
        W >>= 1
    end
    WT = W * T_size
    Q, r = divrem(Mₖ, W) #In case Mₖ is not a multiple of W

    V = Vec{W,T}
    if r == 0
        mask = create_mask(W, 0)
        A_load_expr = :(@nexprs $Q q -> vA_q = vload($V, pA + n*$A_stride + $WT*(q-1)))
        D_store1 = :(@nexprs $Q q -> vstore!(pD + $WT*(q-1) + $D_stride*(p-1), Dx_q_p))
        D_store2 = :(@nexprs $Q q -> vstore!(pD + $WT*(q-1) + $(D_stride*(Pₖ-1)),$(Symbol(:Dx_q,:_,Pₖ))))
    else
        mask = create_mask(W, r)
        if Q == 0
            Q = 1
            A_load_expr = :($(Symbol(:vA_, Q)) = vload($V, pA + $((N-1)*A_stride) + $(WT*(Q-1)), $mask))
        else
            A_load_expr = quote
                @nexprs $Q q -> vA_q = vload($V, pA + $((N-1)*A_stride) + $WT*(q-1))
            end
            Q += 1
            push!(A_load_expr.args, :($(Symbol(:vA_, Q)) = vload($V, pA + $((N-1)*A_stride) + $(WT*(Q-1)), $mask)))
        end

        # D_store1 = :(@nexprs $Q q -> vstore!(pD + $WT*(q-1) + $A_stride*(p-1), Dx_q_p))
        D_store1 = quote
            @nexprs $(Q-1) q -> vstore!(pD + $WT*(q-1) + $A_stride*(p-1), Dx_q_p)
            vstore!(pD + $(WT*(Q-1)) + $D_stride*(p-1), $(Symbol(:Dx_, Q, :_p)), $mask)
        end
        
        # if stride_D == Mₖ
            # if stride_D == Mₖ, successive stores will overwrite the trailing elements from previous
            # stores. Therefore, we only have to mask the last store.
        D_store2 = quote
            @nexprs $(Q-1) q -> vstore!(pD + $WT*(q-1) + $(A_stride*(Pₖ-1)), $(Symbol(:Dx_q_,Pₖ)))
            vstore!(pD + $(WT*(Q-1) + D_stride*(Pₖ-1)), $(Symbol(:Dx_, Q, :_, Pₖ)), $mask)
        end
        # else
        #     # Otherwise, we mask every store.
        #     D_store2 = quote
        #         @nexprs $Q q -> vstore!(pD + $WT*(q-1) + $(A_stride*(Pₖ-1)), $(Symbol(:Dx_q_,Pₖ)), $mask)
        #     end
        # end
    end
    C = min(VectorizationBase.CACHELINE_SIZE ÷ T_size,N)
    Qₚ = cld(Mₖ, C)
    # Check whether we are prefetching A and/or X.
    pfA_1, pfA_2, pfA_3 = prefetch_A(pf, N, Qₚ, A_stride)
    pfX_1, pfX_2, pfX_3, pfX_4 = prefetch_X(pf, N, Pₖ, X_stride, T_size)
    inline_expr = inline ? Expr(:meta, :inline) : :(nothing)
    if init
        q = mulinit(V, WT, Q, Pₖ, X_stride, r, mask, inline_expr, pfA_1)
    else
        q = gemminit(V, WT, Q, Pₖ, A_stride, r, mask, inline_expr)
    end

    if pfX_1 == nothing
        push!(q.args,
        quote
            for n ∈ $(Int(init)):$(r == 0 ? N-1 : N-2 )
                @nexprs $Q q -> vA_q = vload($V, pA + n*$A_stride + $WT*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, VectorizationBase.load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_q_p = vmuladd(vA_q, vX, Dx_q_p)
                end
                $pfA_2
                # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$A_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
        end)
        if r > 0
            push!(q.args,
            quote
                $A_load_expr
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, VectorizationBase.load(pX + $((N-1)*T_size) + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_q_p = vmuladd(vA_q, vX, Dx_q_p)
                end
                $pfA_3
                @nexprs $(Pₖ-1) p -> $D_store1
                $D_store2
                # @nexprs $(Pₖ-1) p -> $D_store2
                nothing
            end )
        else
            push!(q.args,
            quote
                @nexprs $Pₖ p -> $D_store1
                nothing
            end)
        end
    else
        push!(q.args,
        quote
            # @nexprs $Qₚ q -> prefetch(pA + pf.A + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            for n ∈ $(Int(init)):$(C-1)
                @nexprs $Q q -> vA_q = vload($V, pA + n*$A_stride + $WT*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, VectorizationBase.load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_q_p = vmuladd(vA_q, vX, Dx_q_p)
                end
                $pfA_2
                # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$A_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
            # @nexprs $Pₖ p -> prefetch(pX + pf.X + (p-1)*$X_stride, Val(3), Val(0))
            $pfX_1
        end)
        if (N - (N % C) == N) && (r > 0)
            C_upper_bound = N - 2C
            must_finish_iter = true
            remaining_iterations = N-2C+1:N-C-1
        else
            C_upper_bound = N - C
            must_finish_iter = N - (N % C) < (r == 0 ? N-1 : N-2 )
            remaining_iterations = (N - (N % C)):(r == 0 ? N-1 : N-2 )
        end
        push!(q.args,
        quote
            for n₁ ∈ $C:$C:$C_upper_bound
                for n ∈ n₁:n₁+$(C-1)
                    @nexprs $Q q -> vA_q = vload($V, pA + n*$A_stride + $WT*(q-1))
                    @nexprs $Pₖ p -> begin
                        vX = vbroadcast($V, VectorizationBase.load(pX + n*$T_size + (p-1)*$X_stride))
                        @nexprs $Q q -> Dx_q_p = vmuladd(vA_q, vX, Dx_q_p)
                    end
                    $pfA_2
                    # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$A_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
                end
                # @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
                $pfX_2
            end
        end)
        if must_finish_iter
            push!(q.args,
            quote
                for n ∈ $remaining_iterations
                    @nexprs $Q q -> vA_q = vload($V, pA + n*$A_stride + $WT*(q-1))
                    @nexprs $Pₖ p -> begin
                        vX = vbroadcast($V, VectorizationBase.load(pX + n*$T_size + (p-1)*$X_stride))
                        @nexprs $Q q -> Dx_q_p = vmuladd(vA_q, vX, Dx_q_p)
                    end
                    $pfA_2
                    # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$A_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
                end
            end)
        end
        if r > 0
            push!(q.args,
            quote
                $A_load_expr
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, VectorizationBase.load(pX + $((N-1)*T_size) + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_q_p = vmuladd(vA_q, vX, Dx_q_p)
                end
                $pfA_3
            end )
        end

        push!(q.args,
        quote
            @nexprs $(Pₖ-1) p -> begin
                # prefetch(pX + pf.X + $(N*T_size) + (p-1)*$X_stride, Val(3), Val(0))
                $pfX_3
                $D_store1
                # $D_store2
            end
            $pfX_4
            $D_store2
            nothing
        end)
    end
    q
end

@generated function kernel!(pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}) where {Mₖ,Pₖ,stride_AD,stride_X,N,T}
    kernel_quote(Mₖ,Pₖ,stride_AD,stride_X,N,T,false,true)
end

@generated function initkernel!(pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, K::Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}) where {Mₖ,Pₖ,stride_AD,stride_X,N,T}
    kernel_quote(Mₖ,Pₖ,stride_AD,stride_X,N,T,true,true)
end

@generated function kernel!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_AD,stride_X,N},::Val{PF}
) where {Mₖ,Pₖ,stride_AD,stride_X,N,T,PF}
    kernel_quote(Mₖ,Pₖ,stride_AD,stride_X,N,T,false,true,PF)
end

@generated function initkernel!(
    pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, K::Kernel{Mₖ,Pₖ,stride_AD,stride_X,N},::Val{PF}
) where {Mₖ,Pₖ,stride_AD,stride_X,N,T,PF}
    kernel_quote(Mₖ,Pₖ,stride_AD,stride_X,N,T,true,true,PF)
end



"""
quote for
    D (+)= A * X'
"""
function kernel_nt_quote(Mₖ,Pₖ,stride_AD,stride_X,N,T,init,inline = false, pf = nothing)
    T_size = sizeof(T)
    AD_stride = stride_AD * T_size
    X_stride = stride_X * T_size
    W = VectorizationBase.REGISTER_SIZE ÷ T_size
    while W >= 2Mₖ
        W >>= 1
    end
    WT = W * T_size
    Q, r = divrem(Mₖ, W) #Assuming Mₖ is a multiple of W
    V = Vec{W,T}
    if r == 0
        mask = create_mask(W, 0)
        A_load_expr = :(@nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $WT*(q-1)))
        D_store1 = :(@nexprs $Q q -> vstore!(pD + $WT*(q-1) + $AD_stride*(p-1), Dx_q_p))
        D_store2 = :(@nexprs $Q q -> vstore!(pD + $WT*(q-1) + $(AD_stride*(Pₖ-1)),$(Symbol(:Dx_q_,Pₖ))))
    else
        mask = create_mask(W, r)
        if Q == 0
            Q = 1
            A_load_expr = :($(Symbol(:vA_, Q)) = vload($V, pA + $((N-1)*AD_stride) + $(WT*(Q-1)), $mask))
        else
            A_load_expr = quote
                @nexprs $Q q -> vA_q = vload($V, pA + $((N-1)*AD_stride) + $WT*(q-1))
            end
            Q += 1
            push!(A_load_expr.args, :($(Symbol(:vA_, Q)) = vload($V, pA + $((N-1)*AD_stride) + $(WT*(Q-1)), $mask)))
        end

        D_store1 = :(@nexprs $Q q -> vstore!(pD + $WT*(q-1) + $AD_stride*(p-1), Dx_q_p))
        D_store2 = quote
            @nexprs $(Q-1) q -> vstore!(pD + $WT*(q-1) + $(AD_stride*(Pₖ-1)), $(Symbol(:Dx_q_,Pₖ)))
            vstore!(pD + $(WT*(Q-1) + AD_stride*(Pₖ-1)), $(Symbol(:Dx_, Q, :_, Pₖ)), $mask)
        end
    end
    C = min(VectorizationBase.CACHELINE_SIZE ÷ T_size,N)
    Qₚ = cld(Mₖ, C)
    # Check whether we are prefetching A and/or X.
    pfA_1, pfA_2, pfA_3 = prefetch_A(pf, N, Qₚ, AD_stride)
    pfX_1, pfX_2, pfX_3, pfX_4 = prefetch_X(pf, N, Pₖ, X_stride, T_size)
    inline_expr = inline ? Expr(:meta, :inline) : :(nothing)
    if init
        q = mulinit(V, WT, Q, Pₖ, X_stride, r, mask, inline_expr, pfA_1)
    else
        q = gemminit(V, WT, Q, Pₖ, AD_stride, r, mask, inline_expr)
    end

    if pfX_1 == nothing
        push!(q.args,
        quote
            for n ∈ $(Int(init)):$(r == 0 ? N-1 : N-2 )
                @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $WT*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, VectorizationBase.load(pX + n*$X_stride + (p-1)*$T_size))
                    @nexprs $Q q -> Dx_q_p = vmuladd(vA_q, vX, Dx_q_p)
                end
                $pfA_2
                # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
        end)
        if r > 0
            push!(q.args,
            quote
                $A_load_expr
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, VectorizationBase.load(pX + $((N-1)*X_stride) + (p-1)*$T_size))
                    @nexprs $Q q -> Dx_q_p = vmuladd(vA_q, vX, Dx_q_p)
                end
                $pfA_3
                @nexprs $Pₖ p -> $D_store1
                nothing
            end )
        else
            push!(q.args,
            quote
                @nexprs $(Pₖ-1) p -> $D_store1
                $D_store2
                nothing
            end)
        end
    else
        push!(q.args,
        quote
            # @nexprs $Qₚ q -> prefetch(pA + pf.A + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            for n ∈ $(Int(init)):$(C-1)
                @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $WT*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, VectorizationBase.load(pX + n*$X_stride + (p-1)*$T_size))
                    @nexprs $Q q -> Dx_q_p = vmuladd(vA_q, vX, Dx_q_p)
                end
                $pfA_2
                # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
            # @nexprs $Pₖ p -> prefetch(pX + pf.X + (p-1)*$X_stride, Val(3), Val(0))
            $pfX_1
        end)
        if (N - (N % C) == N) && (r > 0)
            C_upper_bound = N - 2C
            must_finish_iter = true
            remaining_iterations = N-2C+1:N-C-1
        else
            C_upper_bound = N - C
            must_finish_iter = N - (N % C) < (r == 0 ? N-1 : N-2 )
            remaining_iterations = (N - (N % C)):(r == 0 ? N-1 : N-2 )
        end
        push!(q.args,
        quote
            for n₁ ∈ $C:$C:$C_upper_bound
                for n ∈ n₁:n₁+$(C-1)
                    @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $WT*(q-1))
                    @nexprs $Pₖ p -> begin
                        vX = vbroadcast($V, VectorizationBase.load(pX + n*$X_stride + (p-1)*$T_size))
                        @nexprs $Q q -> Dx_q_p = vmuladd(vA_q, vX, Dx_q_p)
                    end
                    $pfA_2
                    # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
                end
                # @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
                $pfX_2
            end
        end)
        if must_finish_iter
            push!(q.args,
            quote
                for n ∈ $remaining_iterations
                    @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $WT*(q-1))
                    @nexprs $Pₖ p -> begin
                        vX = vbroadcast($V, VectorizationBase.load(pX + n*$X_stride + (p-1)*$T_size))
                        @nexprs $Q q -> Dx_q_p = vmuladd(vA_q, vX, Dx_q_p)
                    end
                    $pfA_2
                    # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
                end
            end)
        end
        if r > 0
            push!(q.args,
            quote
                $A_load_expr
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, VectorizationBase.load(pX + $((N-1)*X_stride) + (p-1)*$T_size))
                    @nexprs $Q q -> Dx_q_p = vmuladd(vA_q, vX, Dx_q_p)
                end
                $pfA_3
            end )
        end

        push!(q.args,
        quote
            @nexprs $(Pₖ-1) p -> begin
                # prefetch(pX + pf.X + $(N*T_size) + (p-1)*$X_stride, Val(3), Val(0))
                $pfX_3
                $D_store1
            end
            $pfX_4
            $D_store2
            nothing
        end)
    end
    q
end

@generated function kernel_nt!(pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}) where {Mₖ,Pₖ,stride_AD,stride_X,N,T}
    kernel_quote(Mₖ,Pₖ,stride_AD,stride_X,N,T,false,true)
end

@generated function initkernel_nt!(pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, K::Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}) where {Mₖ,Pₖ,stride_AD,stride_X,N,T}
    kernel_quote(Mₖ,Pₖ,stride_AD,stride_X,N,T,true,true)
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
function pick_kernel_size(
    ::Type{T}, row_max = typemax(Int), col_max = typemax(Int);
    D_count = 1, A_count = 1, X_count = 1,
    W = VectorizationBase.REGISTER_SIZE ÷ sizeof(T),
    NREG = VectorizationBase.REGISTER_COUNT, verbose = false,
    max_aloads = min(NREG >> 1, row_max)
) where {T}
    T_size = sizeof(T)
    
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
#            rem_col_indicator = remcol > 0 ? 1 : 0
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
#            rem_row_indicator = remrow > 0 ? 1 : 0
        else
            next_ratio_fc_rr = rem_row_fc_fr
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
        if verbose
            @show a_loads, num_rows, num_cols, next_ratio
        end
#        if next_ratio < prev_ratio
#            break
#        else
#            prev_ratio = next_ratio
#            prev_num_rows, prev_num_cols = num_rows, num_cols
#        end
        if num_rows >= row_max
            ratios[a_loads+1:end] .= 0
            break
        end
    end
    max_ratio, max_ind = findmax(ratios)
    W, rows[max_ind], cols[max_ind]
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
function blocking_structure(M, N, P, ::Type{T} = Float64;
        cache_size::NTuple{3,Int} = VectorizationBase.CACHE_SIZE,
                D_count = 1, A_count = 1, X_count = 1) where T
    total_elements = M*N*D_count + N*P*A_count + M*P*X_count
    L1, L2, L3 = cache_size .÷ sizeof(T)
    if L1 > total_elements
        epr, m_1, p_1 = pick_kernel_size(T, M, P, D_count = D_count, A_count = A_count, X_count = X_count)
        # if m_1 <= M && p_1 <= P
        #     return ((M,N,P),(M,N,P),(M,N,P)),-1
        # else
        return ((min(m_1,M),N,min(p_1,P)),(M,N,P),(M,N,P)),0
        # end
        # return ((M,N,P),(M,N,P),(M,N,P)),0
    end

    epr, m_1, p_1 = pick_kernel_size(T, M, P, D_count = D_count, A_count = A_count, X_count = X_count)
    # @show m_1, p_1, L1, L2, L3
    Dmp_1 = m_1 * p_1

    n_1 = (L1 - Dmp_1) ÷ (m_1 + p_1)

    # I need to consider the case where
    # m_1 or p_1 can be some multiple of themselves due to N being too small.
    # n_2 = n_1 = min(N, n_1)
    if n_1 > N
        n_2 = n_1 = N
        # m_1, p_1 = divide_into_rough_square(L1, M, P, n_1, m_1, p_1)#, D_count = D_count, A_count = A_count, X_count = X_count)
    else
        n_2 = n_1
    end

    # else # currently not bothering to handle the "else".

    # end
    # num_elements = cache_size[i+1] ÷ sizeofT
    # 0 = m^2 + 2m*n_2 - L2

    if L2 > total_elements
        return ((m_1, n_1, p_1), (M, N, P), (M, N, P)),1
    end

    # Try to upper bound size of m_2, p_2
    # Consider safety factors, for other things (instructions?) in this cache?
    m_2, p_2 = divide_into_rough_square(L2, M, P, n_2, m_1, p_1)#, D_count = D_count, A_count = A_count, X_count = X_count)

    if L3 > total_elements
        # if m_2 == M && p_2 == P
        #     return ((m_1, n_1, p_1), (M, N, p_2), (M, N, P)),1
        # else
            return ((m_1, n_1, p_1), (m_2, n_2, p_2), (M, N, P)),2
        # end
    end

    Dmp_2 = m_2 * p_2
    n_3 = (L3 - Dmp_2) ÷ (m_2 + p_2)
    if n_3 > N
        n_3 = N
        m_3, p_3 = divide_into_rough_square(L3, M, P, N, m_2, p_2)#, D_count = D_count, A_count = A_count, X_count = X_count)
    else
        m_3, p_3 = m_2, p_2
    end

    (Base.Cartesian.@ntuple 3 i -> (m_i, n_i, p_i)),3

end

function round_x_to_nearest_y(x::T, y::T) where {T}
    z = x/y
    round(T, z) * y
end

function divide_into_rough_square(L, M, P, n, mbase, pbase)
    Mhalf = max(round_x_to_nearest_y(M>>1, mbase), mbase)
    Phalf = max(round_x_to_nearest_y(P>>1, pbase), pbase)
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

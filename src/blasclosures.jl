
# runfunc!(LoopMulClosure{false}(_C, _A, _B, α, β, msize, K, nsize), (tnum += 1))

@generated function cfuncpointer(::T) where {T}
    quote
        @cfunction($T, Cvoid, (ChannelArgs,))
    end
end


struct LoopMulFunc{P,TC,TA,TB,Α,Β,Md,Kd,Nd} <: Function end
    # C::TC
    # A::TA
    # B::TB
    # α::Α # \Alpha
    # β::Β # \Beta
    # M::Md
    # K::Kd
    # N::Nd
# end


function (::LoopMulFunc{P,TC,TA,TB,Α,Β,Md,Kd,Nd})(
    data::ChannelArgs
) where {P,TC,TA,TB,Α,Β,Md,Kd,Nd}
    atomicp, _, (uα, uβ, ptr_args, uM, uK, uN), __, ___ = data

    C, A, B = reassemble(TC, TA, TB, ptr_args)
    α = coef(Α, uα); β = coef(Β, uβ);
    M = coef(Md, uM); K = coef(Kd, uK); N = coef(Nd, uN);

    call_loopmul!(atomicp, C, A, B, α, β, M, K, N, Val{P}())
    nothing
end
function call_loopmul!(atomicp, C, A, B, α, β, M, K, N, ::Val{false})
    loopmul!(C, A, B, α, β, (CloseOpen(M), CloseOpen(K), CloseOpen(N)))
    _atomic_add!(atomicp, one(UInt))
    nothing
end
function call_loopmul!(atomicp, C, A, B, α, β, M, K, N, ::Val{true})
    Mc,Kc,Nc = matmul_params(T)
    jmulpackAonly!(C, A, B, α, β, Mc, Kc, Nc, (M, K, N))
    _atomic_add!(atomicp, one(UInt))
    nothing
end

function mul_call_setup(C, A, B, α, β, M, K, N)
    uα = ucoef(α); uβ = ucoef(β)
    ptr_args = dissassemble(C, A, B)
    uα, uβ, ptr_args, M % UInt32, K % UInt32, N % UInt32
end
function loopmul_call_thread!(
    atomicp::Ptr{UInt}, C::TC, A::TA, B::TB, α::Α, β::Β, M::Md, K::Kd, N::Nd, ::Val{P}, tid::Integer
) where {P, TC, TA, TB, Α, Β, Md, Kd, Nd}
    fptr = cfuncpointer(LoopMulFunc{P,TC,TA,TB,Α,Β,Md,Kd,Nd})
    
    fptr_args = fptr, (atomicp, C_NULL, mul_call_setup(C, A, B, α, β, M, K, N), zero(UInt32), zero(UInt32))

    put!(FCHANNEL[tid], fptr_args)
    ccall(:jl_wakeup_thread, Cvoid, (Int16,), tid % Int16)
    nothing
end


struct SyncClosure{Mc,Kc,Nc,TC,TA,TB,Α,Β,Md,Kd,Nd,CA} end
    # C::TC
    # A::TA
    # B::TB
    # α::Α
    # β::Β
    # M::Md
    # K::Kd
    # N::Nd
    # p::Ptr{UInt}
    # bc::Ptr{CA}
    # id::UInt
    # last_id::UInt
# end

function (::SyncClosure{Mc,Kc,Nc,TC,TA,TB,Α,Β,Md,Kd,Nd,CA})(
    data::ChannelArgs
) where {Mc, Kc, Nc, TC, TA, TB, Α, Β, Md, Kd, Nd, CA}
    atomicp, ubc, (uα, uβ, ptr_args, uM, uK, uN), id, last_id = data

    C, A, B = reassemble(TC, TA, TB, ptr_args)
    α = coef(Α, uα); β = coef(Β, uβ);
    M = coef(Md, uM); K = coef(Kd, uK); N = coef(Nd, uN);

    bc = reinterpret(Ptr{CA}, ubc)
    
    sync_mul!(C, A, B, α, β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), M, K, N, atomicp, bc, id, last_id)
end

function sync_mul_thread!(
    atomicp::Ptr{UInt}, pbc::Ptr{CA}, C::TC, A::TA, B::TB, α::Α, β::Β, M::Md, K::Kd, N::Nd, id::UInt32, last_id::UInt32, ::StaticInt{Mc}, ::StaticInt{Kc}, ::StaticInt{Nc}, tid::Integer
) where {Mc, Kc, Nc, TC, TA, TB, Α, Β, Md, Kd, Nd, CA}
    fptr = cfuncpointer(SyncClosure{Mc, Kc, Nc, TC, TA, TB, Α, Β, Md, Kd, Nd, CA})
    fptr_args = fptr, (atomicp, reinterpret(Ptr{Cvoid}, pbc), mul_call_setup(C, A, B, α, β, M, K, N), id, last_id)
    put!(FCHANNEL[tid], fptr_args)
    ccall(:jl_wakeup_thread, Cvoid, (Int16,), tid % Int16)
    nothing
end

# function SyncClosure{Mc,Kc,Nc}(
#     C::TC, A::TA, B::TB, α::Α, β::Β, M::Md, K::Kd, N::Nd, p::Ptr{UInt}, bc::Ptr{CA}, id::UInt, last_id::UInt
# ) where {Mc,Kc,Nc,TC,TA,TB,Α,Β,Md,Kd,Nd,CA}
#     SyncClosure{Mc,Kc,Nc,TC,TA,TB,Α,Β,Md,Kd,Nd,CA}(C, A, B, α, β, M, K, N, p, bc, id, last_id)
# end

# function (sc::SyncClosure{Mc,Kc,Nc})() where {Mc,Kc,Nc}
#     sync_mul!(sc.C, sc.A, sc.B, sc.α, sc.β, StaticInt{Mc}(), StaticInt{Kc}(), StaticInt{Nc}(), sc.M, sc.K, sc.N, sc.p, sc.bc, sc.id, sc.last_id)
# end



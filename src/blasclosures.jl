


"""
  LoopMulClosure{P,TC,TA,TB,Α,Β,Md,Kd,Nd}


Closure for simple loopmuls. `P` indicates whether or not it should pack `A`.
"""
struct LoopMulClosure{P,TC,TA,TB,Α,Β,Md,Kd,Nd}
    C::TC
    A::TA
    B::TB
    α::Α # \Alpha
    β::Β # \Beta
    M::Md
    K::Kd
    N::Nd
end
function LoopMulClosure{P}(
    C::TC, A::TA, B::TB, α::Α, β::Β, M::Md, K::Kd, N::Nd
) where {P,TC<:AbstractStridedPointer,TA<:AbstractStridedPointer,TB<:AbstractStridedPointer,Α,Β,Md,Kd,Nd}
    LoopMulClosure{P,TC,TA,TB,Α,Β,Md,Kd,Nd}(C, A, B, α, β, M, K, N)
end
function LoopMulClosure{P}(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β, M, K, N) where {P} # if not packing, discard `PtrArray` wrapper
    LoopMulClosure{P}(zstridedpointer(C), zstridedpointer(A), zstridedpointer(B), α, β, M, K, N)
end
(m::LoopMulClosure{false})() = loopmul!(m.C, m.A, m.B, m.α, m.β, m.M, m.K, m.N)
function (m::LoopMulClosure{true,TC})() where {T,TC <: AbstractStridedPointer{T}}
    Mc,Kc,Nc = matmul_params(T)
    jmulpackAonly!(m.C, m.A, m.B, m.α, m.β, m.M, m.K, m.N, Mc, Kc, Nc)
end


struct SyncClosure{W₁,W₂,R₁,R₂,TC,TA,TB,Α,Β,Md,Kd,Nd,CA}
    C::TC
    A::TA
    B::TB
    α::Α
    β::Β
    M::Md
    K::Kd
    N::Nd
    p::Ptr{UInt}
    bc::Ptr{CA}
    id::UInt
    last_id::UInt
end

function SyncClosure{W₁,W₂,R₁,R₂}(
    C::TC, A::TA, B::TB, α::Α, β::Β, M::Md, K::Kd, N::Nd, p::Ptr{UInt}, bc::Ptr{CA}, id::UInt, last_id::UInt
) where {W₁,W₂,R₁,R₂,TC,TA,TB,Α,Β,Md,Kd,Nd,CA}
    SyncClosure{W₁,W₂,R₁,R₂,TC,TA,TB,Α,Β,Md,Kd,Nd,CA}(C, A, B, α, β, M, K, N, p, bc, id, last_id)
end

function (sc::SyncClosure{W₁,W₂,R₁,R₂})() where {W₁,W₂,R₁,R₂}
    sync_mul!(
        sc.C, sc.A, sc.B, sc.α, sc.β, sc.M, sc.K, sc.N, sc.p, sc.bc, sc.id, sc.last_id,
        StaticFloat{W₁}(), StaticFloat{W₂}(), StaticFloat{R₁}(), StaticFloat{R₂}()
    )
end


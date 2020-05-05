function matmul_sizes(C, A, B)
    MC, NC = maybestaticsize(C, Val{1:2}())
    MA, KA = maybestaticsize(A, Val{1:2}())
    KB, NB = maybestaticsize(B, Val{1:2}())
    M = VectorizationBase.static_promote(MC, MA)
    K = VectorizationBase.static_promote(KA, KB)
    N = VectorizationBase.static_promote(NC, NB)
    M, K, N
end

let GEMMLOOPSET = LoopVectorization.LoopSet(
    :(for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
          Cₘₙ = zero(eltype(C))
          for k ∈ 1:size(A,2)
              Cₘₙ += A[m,k] * B[k,n]
          end
          C[m,n] += Cₘₙ
      end)
    );
    order = LoopVectorization.choose_order(GEMMLOOPSET)
    mr = order[5]
    nr = last(order)
    @eval const mᵣ = $mr
    @eval const nᵣ = $nr
    # for T ∈ [Int16, Int32, Int64, Float32, Float64]
    # end

    
end

function loopmul!(C, A, B, ::Val{1}, ::Val{0}, (M, K, N) = matmul_sizes(C, A, B))
    @avx for n ∈ 1:N, m ∈ 1:M
        Cₘₙ = zero(eltype(C))
        for k ∈ 1:K
            Cₘₙ += A[m,k] * B[k,n]
        end
        C[m,n] = Cₘₙ
    end
    nothing
end
function loopmul!(C, A, B, ::Val{1}, ::Val{1}, (M, K, N) = matmul_sizes(C, A, B))
    @avx for n ∈ 1:N, m ∈ 1:M
        Cₘₙ = zero(eltype(C))
        for k ∈ 1:K
            Cₘₙ += A[m,k] * B[k,n]
        end
        C[m,n] += Cₘₙ
    end
    nothing
end
function loopmul!(C, A, B, α, ::Val{0}, (M, K, N) = matmul_sizes(C, A, B))
    @avx for n ∈ 1:N, m ∈ 1:M
        Cₘₙ = zero(eltype(C))
        for k ∈ 1:K
            Cₘₙ += A[m,k] * B[k,n]
        end
        C[m,n] = α * Cₘₙ
    end
    nothing
end

function loopmul!(C, A, B, α, ::Val{1}, (M, K, N) = matmul_sizes(C, A, B))
    @avx for n ∈ 1:N, m ∈ 1:M
        Cₘₙ = zero(eltype(C))
        for k ∈ 1:K
            Cₘₙ += A[m,k] * B[k,n]
        end
        C[m,n] += α * Cₘₙ
    end
    nothing
end
function loopmul!(C, A, B, α, β, (M, K, N) = matmul_sizes(C, A, B))
    @avx for n ∈ 1:N, m ∈ 1:M
        Cₘₙ = zero(eltype(C))
        for k ∈ 1:K
            Cₘₙ += A[m,k] * B[k,n]
        end
        C[m,n] = α * Cₘₙ + β * C[m,n]
    end
    nothing
end

const loopmulprefetch! = loopmul!

@inline function inlineloopmul!(C, A, B, ::Val{1}, ::Val{0})
    M, K, N = matmul_sizes(C, A, B)
    @avx for n ∈ 1:N, m ∈ 1:M
        Cₘₙ = zero(eltype(C))
        for k ∈ 1:K
            Cₘₙ += A[m,k] * B[k,n]
        end
        C[m,n] = Cₘₙ
    end
    C
end
@inline function inlineloopmul!(C, A, B, α, ::Val{0})
    M, K, N = matmul_sizes(C, A, B)
    @avx for n ∈ 1:N, m ∈ 1:M
        Cₘₙ = zero(eltype(C))
        for k ∈ 1:K
            Cₘₙ += A[m,k] * B[k,n]
        end
        C[m,n] = α * Cₘₙ
    end
    C
end
@inline function inlineloopmul!(C, A, B, α, β)
    M, K, N = matmul_sizes(C, A, B)
    @avx for n ∈ 1:N, m ∈ 1:M
        Cₘₙ = zero(eltype(C))
        for k ∈ 1:K
            Cₘₙ += A[m,k] * B[k,n]
        end
        C[m,n]  = β * C[m,n] + α * Cₘₙ
    end
    C
end

function two_cols_per_vector_add_kn_block!(q::Expr, Kunroll, Nunroll, Koffset, Noffset, Ashuffle, Bshuffle, MindA, MindC)
    for kk ∈ 1:Kunroll
        kt = kk + Koffset
        push!(q.args, Expr(:(=), Symbol(:A_,kk), Expr(:call, :shufflevector, Expr(:call, :vload, :ptrA, Expr(:tuple, MindA, kt)), Ashuffle)))
    end
    for kk ∈ 1:Kunroll, nn ∈ 1:Nunroll
        kt = kk + Koffset
        nt2 = 2(nn + Noffset); nt1 = nt2 - 1
        Bsym = Symbol(:B_,kk,:_,nn)
        Csym = Symbol(:C_, nn)
        # push!(q.args, Expr(:(=), Bsym, Expr(:call, :shufflevector, Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt)), Bshuffle)))
        push!(q.args, Expr(:(=), Bsym, Expr(:call, :shufflevector, Expr(:tuple,
                                                                        Expr(:call, :(Core.VecElement), Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt1))),
                                                                        Expr(:call, :(Core.VecElement), Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt2)))), Bshuffle)))
        if iszero(Koffset) & isone(kk)
            push!(q.args, Expr(:(=), Csym, Expr(:call, :vmul, Symbol(:A_, kk), Bsym)))
        else
            push!(q.args, Expr(:(=), Csym, Expr(:call, :vfmadd_fast, Symbol(:A_, kk), Bsym, Csym)))
        end
    end
    for nn ∈ 1:Nunroll
        push!(q.args, Expr(:call, :vnoaliasstore!, :ptrC, Symbol(:C_, nn), Expr(:tuple, MindC, 2(nn + Noffset) - 1)))
    end
end

function initmulquote()
    Expr(
        :block,
        Expr(:meta, :inline),
        Expr(:(=), :ptrC, Expr(:call, :stridedpointer, :C)),
        Expr(:(=), :ptrA, Expr(:call, :stridedpointer, :A)),
        Expr(:(=), :ptrB, Expr(:call, :stridedpointer, :B))
    )
end

function two_cols_per_vector_quote(K, N, W, Wshift)
    # Calculate 2 columns of C at a time
    q = two_cols_per_vector_quote!(initmulquote(), K, N, W, Wshift)
    push!(q.args, :C)
    q
end

function two_cols_per_vector_quote!(q, K, N, W, Wshift, Noffbase = 0)
    Ndoublereps = N >> 1
    Wh = W >>> 1
    Whs = Wshift - 1

    Kloop = LoopVectorization.Loop(:k, 1, K, Symbol(""), Symbol(""), true, true)
    Nloop = LoopVectorization.Loop(:n, 1, Ndoublereps, Symbol(""), Symbol(""), true, true)
    cost_vec     = Float64[ 64.0, 32.0, 8.0, 0.0 ] # Values chosen to hopefully produce desired behavior... TODO: do this better?
    reg_pressure = Float64[ 1.0, 1.0, 1.0, 0.0 ]
    Kunroll, Nunroll, cost = LoopVectorization.solve_unroll(
        :k, :n, cost_vec, reg_pressure, W, :m, Kloop, Nloop
    )
    @assert isfinite(cost)
    
    Nrep, Nrem = divrem(Ndoublereps, Nunroll)
    Krep, Krem = divrem(K, Kunroll)
    Ashuffle = Expr(:call, Expr(:curly, :Val, Expr(:tuple, (0:Wh-1)..., (0:Wh-1)...)))
    Bshuffle = Expr(:call, Expr(:curly, :Val, Expr(:tuple, (0 for _ ∈ 1:Wh)..., (1 for _ ∈ 1:Wh)...)))
    MindA = Expr(:call, Expr(:curly, :_MM, Wh), 1)
    MindC = Expr(:call, Expr(:curly, :_MM, W), 1)
    for n ∈ 0:Nrep-1
        for k ∈ 0:Krep-1
            two_cols_per_vector_add_kn_block!(q, Kunroll, Nunroll, Kunroll*k, Nunroll*n + Noffbase, Ashuffle, Bshuffle, MindA, MindC)
        end
        if Krem > 0
            two_cols_per_vector_add_kn_block!(q, Krem, Nunroll, Kunroll*Krep, Nunroll*n + Noffbase, Ashuffle, Bshuffle, MindA, MindC)
        end
    end
    if Nrem > 0
        for k in 0:Krep-1
            two_cols_per_vector_add_kn_block!(q, Kunroll, Nrem, Kunroll*k, Nunroll*Nrep + Noffbase, Ashuffle, Bshuffle, MindA, MindC)
        end
        if Krem > 0
            two_cols_per_vector_add_kn_block!(q, Krem, Nrem, Kunroll*Krep, Nunroll*Nrep + Noffbase, Ashuffle, Bshuffle, MindA, MindC)
        end        
    end
    if isodd(N)
        Csym = Symbol(:C_,1)
        for k ∈ 1:K
            Asym = Symbol(:A_,k)
            Bsym = Symbol(:B_,k)
            push!(q.args, Expr(:(=), Asym, Expr(:call, :vload, :ptrA, Expr(:tuple, MindA, k))))
            push!(q.args, Expr(:(=), Bsym, Expr(:call, :vload, :ptrB, Expr(:tuple, k, N))))
            if k == 1
                push!(q.args, Expr(:(=), Csym, Expr(:call, :vmul, Asym, Bsym)))
            else
                push!(q.args, Expr(:(=), Csym, Expr(:call, :vfmadd_fast, Asym, Bsym, Csym)))
            end
        end
        push!(q.args, Expr(:call, :vnoaliasstore!, :ptrC, Csym, Expr(:tuple, MindA, N))) #MindA, because we want half-vector
    end
    q
end

@generated function LinearAlgebra.mul!(
    C::AbstractMutableFixedSizeMatrix{M,N,T,1,4,false},
    A::AbstractMutableFixedSizeMatrix{M,K,T,1,XA},
    B::AbstractMutableFixedSizeMatrix{K,N,T}
) where {M,K,N,XA,T}
# ) where {M,K,N,T,XA}
    W, Wshift = VectorizationBase.pick_vector_width_shift(4N, T)
    nvectors_per_col = W >> 2
    # if nvectors_per_col ≤ 1 || (K * N > 4VectorizationBase.REGISTER_COUNT) || XA < 4
    if nvectors_per_col != 2 || (K * N > 4VectorizationBase.REGISTER_COUNT) || XA < 4
        return Expr(:block, Expr(:meta,:inline), Expr(:call, :jmul!, :C, :A, :B))
    else#if nvectors_per_col == 2
        return two_cols_per_vector_quote(K, N, W, Wshift)
    # else
        # return Expr(:block, Expr(:meta,:inline), Expr(:call, :jmul!, :C, :A, :B))
    end
end


function four_cols_per_vector_add_kn_block!(q::Expr, Kunroll, Nunroll, Koffset, Noffset, Ashuffle, Bshuffle, MindA, MindC)
    for kk ∈ 1:Kunroll
        kt = kk + Koffset
        push!(q.args, Expr(:(=), Symbol(:A_,kk), Expr(:call, :shufflevector, Expr(:call, :vload, :ptrA, Expr(:tuple, MindA, kt)), Ashuffle)))
    end
    for kk ∈ 1:Kunroll, nn ∈ 1:Nunroll
        kt = kk + Koffset
        nt2 = 4(nn + Noffset); nt1 = nt2 - 1
        Bsym = Symbol(:B_,kk,:_,nn)
        Csym = Symbol(:C_, nn)
        # push!(q.args, Expr(:(=), Bsym, Expr(:call, :shufflevector, Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt)), Bshuffle)))
        push!(q.args, Expr(:(=), Bsym, Expr(:call, :shufflevector, Expr(:tuple,
                                                                        Expr(:call, :(Core.VecElement), Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt1))),
                                                                        Expr(:call, :(Core.VecElement), Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt2))),
                                                                        Expr(:call, :(Core.VecElement), Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt2+1))),
                                                                        Expr(:call, :(Core.VecElement), Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt2+2)))), Bshuffle)))
        if iszero(Koffset) & isone(kk)
            push!(q.args, Expr(:(=), Csym, Expr(:call, :vmul, Symbol(:A_, kk), Bsym)))
        else
            push!(q.args, Expr(:(=), Csym, Expr(:call, :vfmadd_fast, Symbol(:A_, kk), Bsym, Csym)))
        end
    end
    for nn ∈ 1:Nunroll
        push!(q.args, Expr(:call, :vnoaliasstore!, :ptrC, Symbol(:C_, nn), Expr(:tuple, MindC, 4(nn + Noffset) - 1)))
    end
end
function four_cols_per_vector_quote(K, N, W, Wshift)
    # Calculate 2 columns of C at a time
    Nquadreps = N >> 2
    Nquadrem = N & 3
    # Nisodd = N & 1
    q = initmulquote()
    Wq = W >>> 2
    Wqs = Wshift - 2

    Kloop = LoopVectorization.Loop(:k, 1, K, Symbol(""), Symbol(""), true, true)
    Nloop = LoopVectorization.Loop(:n, 1, Nquadreps, Symbol(""), Symbol(""), true, true)
    cost_vec     = Float64[ 64.0, 32.0, 8.0, 0.0 ] # Values chosen to hopefully produce desired behavior... TODO: do this better?
    reg_pressure = Float64[ 1.0, 1.0, 1.0, 0.0 ]
    Kunroll, Nunroll, cost = LoopVectorization.solve_unroll(
        :k, :n, cost_vec, reg_pressure, W, :m, Kloop, Nloop
    )
    @assert isfinite(cost)
    
    Nrep, Nrem = divrem(Nquadreps, Nunroll)
    Krep, Krem = divrem(K, Kunroll)
    Ashuffle = Expr(:call, Expr(:curly, :Val, Expr(:tuple, (0:Wq-1)..., (0:Wq-1)..., (0:Wq-1)..., (0:Wq-1)...)))
    Bshuffle = Expr(:call, Expr(:curly, :Val, Expr(:tuple, (0 for _ ∈ 1:Wq)..., (1 for _ ∈ 1:Wq)..., (0 for _ ∈ 1:Wq)..., (1 for _ ∈ 1:Wq)...)))
    MindA = Expr(:call, Expr(:curly, :_MM, Wq), 1)
    MindC = Expr(:call, Expr(:curly, :_MM, W), 1)
    for n ∈ 0:Nrep-1
        for k ∈ 0:Krep-1
            four_cols_per_vector_add_kn_block!(q, Kunroll, Nunroll, Kunroll*k, Nunroll*n, Ashuffle, Bshuffle, MindA, MindC)
        end
        if Krem > 0
            four_cols_per_vector_add_kn_block!(q, Krem, Nunroll, Kunroll*Krep, Nunroll*n, Ashuffle, Bshuffle, MindA, MindC)
        end
    end
    if Nrem > 0
        for k in 0:Krep-1
            four_cols_per_vector_add_kn_block!(q, Kunroll, Nrem, Kunroll*k, Nunroll*Nrep, Ashuffle, Bshuffle, MindA, MindC)
        end
        if Krem > 0
            four_cols_per_vector_add_kn_block!(q, Krem, Nrem, Kunroll*Krep, Nunroll*Nrep, Ashuffle, Bshuffle, MindA, MindC)
        end
    end
    if Nquadrem > 0
        two_cols_per_vector_quote!(q, K, N, W >> 1, Wshift - 1, N & -4)
    end
    push!(q.args, :C)
    q
end

@generated function LinearAlgebra.mul!(
    C::AbstractMutableFixedSizeMatrix{M,N,T,1,2,false},
    A::AbstractMutableFixedSizeMatrix{M,K,T,1,XA},
    B::AbstractMutableFixedSizeMatrix{K,N,T}
) where {M,K,N,XA,T}
# ) where {M,K,N,T,XA}
    W, Wshift = VectorizationBase.pick_vector_width_shift(2N, T)
    nvectors_per_col = W >> 2
    if nvectors_per_col ≤ 1 || (K * N > 4VectorizationBase.REGISTER_COUNT) || XA < 2
    # if nvectors_per_col != 2 || (K * N > 4VectorizationBase.REGISTER_COUNT) || XA < 4
        return Expr(:block, Expr(:meta,:inline), Expr(:call, :jmul!, :C, :A, :B))
    elseif nvectors_per_col == 2
        return two_cols_per_vector_quote(K, N, W, Wshift)
    elseif nvectors_per_col == 4
        return four_cols_per_vector_quote(K, N, W, Wshift)
    else
        return Expr(:block, Expr(:meta,:inline), Expr(:call, :jmul!, :C, :A, :B))
    end
end


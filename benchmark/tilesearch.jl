
using PaddedMatrices, VectorizationBase

function kcc(K, kc)
    num_k_iter = cld(K, kc)
    kreps_per_iter = ((div(K, num_k_iter)) + 3) & -4
    Krem = K - (num_k_iter - 1) * kreps_per_iter
    kreps_per_iter, Krem
end



using PaddedMatrices, VectorizationBase, ProgressMeter
function pick_K(M_c, r = 1.6, Koff=0)
    L2 = VectorizationBase.CACHE_SIZE[2] ÷ sizeof(Float64)
    
    Kbase = round(Int, (L2 * r) / M_c)
    Kbase + Koff
end
function pick_N(K_c, r)
    L3 = VectorizationBase.CACHE_SIZE[3] ÷ sizeof(Float64)
    round(Int, (L3 * r) / (K_c * PaddedMatrices.nᵣ)) * PaddedMatrices.nᵣ
end

# function jmultpackab!(C, A, B, ::Val{M_c}, ::Val{K_c}, ::Val{N_c}) where {M_c, K_c, N_c}
#     M, N = size(C); K = size(B,1)
#     zc, za, zb = PaddedMatrices.zstridedpointer.((C,A,B))
#     @elapsed(PaddedMatrices.jmultpackAB!(zc, za, zb, StaticInt{1}(), StaticInt{0}(), StaticInt{M_c}(), StaticInt{K_c}(), StaticInt{N_c}(), M, K, N, PaddedMatrices.CloseOpen(0, VectorizationBase.NUM_CORES), Val{VectorizationBase.CACHE_COUNT[3]}()))
# end
function jmultpackab!(C, A, B, ::Val{M_c}, ::Val{K_c}, ::Val{N_c}) where {M_c, K_c, N_c}
    M, N = size(C); K = size(B,1)
    GC.@preserve C A B begin
        zc, za, zb = PaddedMatrices.zstridedpointer.((C,A,B))
        @elapsed(PaddedMatrices.jmultpackAB!(zc, za, zb, StaticInt{1}(), StaticInt{0}(), Val{M_c}(), Val{K_c}(), Val{N_c}(), M, K, N, PaddedMatrices.CloseOpen(0, VectorizationBase.NUM_CORES), Val{VectorizationBase.CACHE_COUNT[3]}()))
    end
end

function bench_size(S, Cs, As, Bs, ::Val{M_c}, ::Val{K_c}, ::Val{N_c}) where {M_c, K_c, N_c}
    jmultpackab!(first(Cs), first(As), first(Bs), Val{M_c}(), Val{K_c}(), Val{N_c}()) # compile
    gflop = 0.0
    for sCAB ∈ zip(S,As,Bs,Cs)
        (s,C,A,B) = sCAB::Tuple{Int,Matrix{Float64},Matrix{Float64},Matrix{Float64}}
        # sleep(0.5)
        t = jmultpackab!(C, A, B, Val{M_c}(), Val{K_c}(), Val{N_c}())
        gf = 2e-9*s*s*s / t
        gflop += gf
    end
    gflop / length(S)
end

# function search(Mc_range = 72:24:120, Kc_range = 1.5:0.25:1.7, Nc_range = 1.0:0.1:1.4)
#     best = Ref((0,0,0,-Inf))
#     gflop_array = let S = round.(Int, exp.(range(log(1500), stop = log(10000), length = 100))), As = map(s -> rand(s,s), S), Bs = map(s -> rand(s,s), S), Cs = similar.(As), iter_prod = Iterators.product(Mc_range, Kc_range, Nc_range), p = Progress(length(iter_prod)), best = best
#         map(iter_prod) do P
#             M_c, Koff, r = P
#             K_c = pick_K(M_c, Koff)
#             N_c = pick_N(K_c, r)
#             gflops = bench_size(S, Cs, As, Bs, Val(M_c), Val(K_c), Val(N_c))
#             b = best[]
#             recent = (M_c,K_c,N_c,gflops)
#             bb = if last(b) > gflops
#                 b
#             else
#                 best[] = recent
#             end
#             ProgressMeter.next!(p, showvalues = [(:Last, recent), (:Best, bb)])
#             gflops
#         end
#     end
#     gflop_array, best
# end
function search(Mc_range = 72:24:120, Kc_range = 1.5:0.25:1.7, Nc_range = 1.0:0.1:1.4)
    best = Ref((0,0,0,(0.0,0.0),-Inf))
    gflop_array = let S = round.(Int, exp.(range(log(1500), stop = log(10000), length = 100))), As = map(s -> rand(s,s), S), Bs = map(s -> rand(s,s), S), Cs = similar.(As), iter_prod = Iterators.product(Mc_range, Kc_range, Nc_range), p = Progress(length(iter_prod)), best = best
        map(iter_prod) do P
            M_c, rk, rn = P
            gflops = bench_size(S, Cs, As, Bs, Val(M_c), Val(rk), Val(rn))
            b = best[]
            K_c = pick_K(M_c, rk)
            N_c = pick_N(K_c, rn)
            recent = (M_c, K_c, N_c, (rk, rn), gflops)
            bb = if last(b) > gflops
                b
            else
                best[] = recent
            end
            ProgressMeter.next!(p, showvalues = [(:Last, recent), (:Best, bb)])
            gflops
        end
    end
    gflop_array, best
end

search_range = (72:24:120, 0.52:0.02:0.56, 0.9:0.01:0.92)
gflop_array, best = search(search_range...); getindex.(search_range, Tuple(argmax(gflop_array)))



# julia> gflop_array, best = search(search_range...); getindex.(search_range, Tuple(argmax(gflop_array)))
# Progress: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:23:06
#   Last:  (120, 557, 5301, (0.51, 0.91), 1468.106008702092)
#   Best:  (72, 819, 3600, (0.45, 0.91), 1563.9777269382319)
# (72, 0.45, 0.91)

# 72 with (0.4-0.44) x 0.9-9.92 looks good
# julia> permutedims(gflop_array, (2,3,1))
# 7×5×2 Array{Float64, 3}:
# [:, :, 1] =   # (96 x 0.40:0.01:0.46 x 0.9:0.005:0.92)
#  1566.74  1512.84  1521.73  1558.3   1559.93
#  1569.36  1563.85  1563.67  1559.56  1563.38
#  1566.34  1551.83  1562.32  1558.95  1561.91
#  1567.51  1561.19  1560.29  1557.78  1564.33
#  1567.02  1560.08  1547.32  1559.42  1565.47
#  1501.42  1550.91  1484.06  1554.05  1553.47
#  1503.18  1545.86  1549.18  1550.41  1548.85

# [:, :, 2] =   # (72 x 0.40:0.01:0.46 x 0.9:0.005:0.92)
#  1531.42  1521.71  1520.14  1520.35  1520.72
#  1539.86  1536.97  1532.17  1532.33  1536.38
#  1543.98  1541.24  1535.58  1540.67  1536.43
#  1550.4   1543.33  1542.23  1540.41  1548.76
#  1552.08  1544.08  1548.16  1547.86  1546.74
#  1548.65  1487.99  1544.13  1542.94  1541.12
#  1476.7   1512.51  1542.46  1537.65  1535.09

using JuMP
model = Model()
@variable(model, M_cf, Int)
@variable(model, K_c, Int)
@variable(model, N_cf, Int)
@constraints(
    model, begin
    1 <= M_cf <= 20 # M_c = M_cf * 24
    1 <= K_c <= 4000
    1 <= N_cf <= 2000 # N_c = N_cf * 9
    end
)
@objective(model, Max, gflops)

using MaBLAS, BenchmarkTools

function matmul_params(::Type{T}, mᵣ, nᵣ, mcratio, kcratio, ncratio) where {T}
    L₁, L₂, L₃ = MaBLAS.VectorizationBase.CACHE_SIZE .÷ sizeof(T)
    kc = round(Int, (kcratio * L₁ / nᵣ))
    mc = round(Int, (mcratio * L₂ / (kc*mᵣ))) * mᵣ
    nc = round(Int, (ncratio * L₃ / (kc*nᵣ))) * nᵣ
    mc, kc, nc
end





function search()
    kernels = [
        (16,12),
        (16,14),
        (24,9),
        (32,6),
        (40,5)
    ]
    Kcratio = 0.4:0.125:0.9
    Mcratio = 0.4:0.125:0.9
    Ncratio = 0.4:0.125:0.9
    matmul_sizes = (6:20) .^ 3
    GFLOPS = fill(-Inf, length(Mcratio), length(Kcratio), length(Ncratio), length(kernels), length(matmul_sizes))
    params = Array{Tuple{Tuple{Int,Int},NTuple{3,Int}}}(undef, length(Mcratio), length(Kcratio), length(Ncratio), length(kernels))
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = timelim = 1e-2

    for (szi,sz) ∈ enumerate(matmul_sizes)
        C = Matrix{Float64}(undef, sz, sz);
        A = rand(sz, sz); B = rand(sz, sz);
        for (rri,(mr,nr)) ∈ enumerate(kernels), (nci,ncf) ∈ enumerate(Ncratio), (kci,kcf) ∈ enumerate(Kcratio), (mci,mcf) ∈ enumerate(Mcratio)
            mc, kc, nc = matmul_params(Float64, mr, nr, mcf, kcf, ncf)
            
            t1 = @elapsed MaBLAS.mul!(C, A, B; packing=true, cache_params=(cache_m=mc, cache_k=kc, cache_n=nc), kernel_params=(Val(mr), Val(nr)))
            t0 = if isone(szi) || t1 < 1e-3
                t2 = @belapsed MaBLAS.mul!($C, $A, $B; packing=true, cache_params=(cache_m=$mc, cache_k=$kc, cache_n=$nc), kernel_params=(Val($mr), Val($nr)))
                min(t1,t2)
            elseif 5t1 < timelim
                t2 = @elapsed MaBLAS.mul!(C, A, B; packing=true, cache_params=(cache_m=mc, cache_k=kc, cache_n=nc), kernel_params=(Val(mr), Val(nr)))
                min(t1,t2)
            else
                t1
            end
            GFLOPS[mci,kci,nci,rri,szi] = gflops = 2e-9*sz^3/t0
            isone(szi) && (params[mci,kci,nci,rri] = ((mr,nr), (mc,kc,nc)))
            @show sz, (mr,nr), (mc,kc,nc), gflops
        end
    end
    GFLOPS, params
end


#gflops, params = search()




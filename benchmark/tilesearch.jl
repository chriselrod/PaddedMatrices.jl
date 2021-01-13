

using PaddedMatrices, VectorizationBase, ProgressMeter
using PaddedMatrices: StaticFloat
function jmultpackab!(C, A, B, ::Val{W₁}, ::Val{W₂}, ::Val{R₁}, ::Val{R₂}) where {W₁, W₂, R₁, R₂}
    M, N = size(C); K = size(B,1)
    zc, za, zb = PaddedMatrices.zstridedpointer.((C,A,B))
    @elapsed(
        PaddedMatrices.jmultpackAB!(
            zc, za, zb, StaticInt{1}(), StaticInt{0}(), M, K, N, PaddedMatrices.CloseOpen(0, VectorizationBase.NUM_CORES),
            StaticFloat{W₁}(), StaticFloat{W₂}(), StaticFloat{R₁}(), StaticFloat{R₂}(), Val{VectorizationBase.CACHE_COUNT[3]}()
        )
    )
end

function bench_size(Cs, As, Bs, ::Val{W₁}, ::Val{W₂}, ::Val{R₁}, ::Val{R₂}) where {W₁, W₂, R₁, R₂}
    jmultpackab!(first(Cs), first(As), first(Bs), Val{W₁}(), Val{W₂}(), Val{R₁}(), Val{R₂}())
    gflop = 0.0
    for (C,A,B) ∈ zip(Cs,As,Bs)
        M, K, N = PaddedMatrices.matmul_sizes(C, A, B)
        # sleep(0.5)
        t = jmultpackab!(C, A, B, Val{W₁}(), Val{W₂}(), Val{R₁}(), Val{R₂}())
        gf = 2e-9M*K*N / t
        gflop += gf
    end
    gflop / length(As)
end
matrix_sizes(s::Int) = (s,s,s)
matrix_sizes(MKN::NTuple{3,Int}) = MKN
function matrix_range(l, u, len, ::Type{T} = Float64) where {T}
    matrix_range(round.(Int, exp.(range(log(l), stop = log(u), length = len))), T)
end
function matrix_range(S, ::Type{T} = Float64) where {T}
    Alen = 0; Blen = 0; Clen = 0;
    for s ∈ S
        M, K, N = matrix_sizes(s)
        Alen = max(Alen, M*K)
        Blen = max(Blen, K*N)
        Clen = max(Clen, M*N)
    end
    Abuf = rand(T, Alen)
    Bbuf = rand(T, Blen)
    Cbuf = rand(T, Clen)
    As = Vector{Base.ReshapedArray{T, 2, SubArray{T, 1, Vector{T}, Tuple{Base.OneTo{Int}}, true}, Tuple{}}}(undef, length(S))
    Bs = similar(As); Cs = similar(As);
    for (i,s) ∈ enumerate(S)
        M, K, N = matrix_sizes(s)
        As[i] = reshape(view(Abuf, Base.OneTo(M * K)), (M, K))
        Bs[i] = reshape(view(Bbuf, Base.OneTo(K * N)), (K, N))
        Cs[i] = reshape(view(Cbuf, Base.OneTo(M * N)), (M, N))
    end
    Cs, As, Bs
end
function gflop_map(Cs, As, Bs, ::Val{M_c}, ::Val{K_c}, ::Val{N_c}, ::Val{R₂}, ::Val{R₃}) where {M_c, K_c, N_c, R₂, R₃}
    jmultpackab!(first(Cs), first(As), first(Bs), Val{M_c}(), Val{K_c}(), Val{N_c}(), Val{R₂}(), Val{R₃}())
    gflops = Vector{Float64}(undef, length(S))
    for (C,A,B) ∈ zip(Cs,As,Bs)
        M, K, N = PaddedMatrices.matmul_sizes(C, A, B)
        t = jmultpackab!(C, A, B, Val{W₁}(), Val{W₂}(), Val{R₁}(), Val{R₂}())
        gflops[i] = 2e-9M*K*N / t
    end
    gflops
end

function gridsearch(
    CsAsBs = matrix_range(1_500, 10_000, 100), w₁range = 0.012:0.0005:0.013, w₂range = 0.024:0.001:0.026, r₁range = 0.44:0.01:0.5, r₂range = 0.75:0.025:0.8
)
    search_space = Iterators.product(w₁range, w₂range, r₁range, r₂range)
    best = Ref(((0.0,0.0),(0.0,0.0),-Inf))
    gflop_array = let (Cs,As,Bs) = CsAsBs, iter_prod = search_space, p = Progress(length(iter_prod)), best = best
        map(iter_prod) do (W₁, W₂, R₁, R₂)
            gflops = bench_size(Cs, As, Bs, Val(W₁), Val(W₂), Val(R₁), Val(R₂))
            b = best[]
            recent = ((M_c, K_c, N_c), (R₂, R₃), gflops)
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






# search_range = (120:24:120, 700:200:1100, 4000:2000:6000, 0.44:0.01:0.47, 0.80:0.025:0.875)
# gflop_array, best = search(search_range...); getindex.(search_range, Tuple(argmax(gflop_array)))


# S, Cs, As, Bs = matrix_range(1500, 10_000, 100);

# search_range = (120:24:120, 1000:100:1100, 5000:1000:5000, 0.44:0.01:0.47, 0.80:0.025:0.825)
# gflop_array, best = search((S,Cs,As,Bs), search_range...); getindex.(search_range, Tuple(argmax(gflop_array)))



# gflops_range = gflop_map(S, Cs, As, Bs, Val{120}(), Val{1000}(), Val{5000}(), Val{0.45}(), Val{0.8}());

# summarystats(gflops_range)


# using StatsBase, UnicodePlots

# lineplot(S, gflops_range, title = "Square Matrix GFLOPS", xlabel = "Size", ylabel = "GFLOPS")

# S[10]
# StatsBase.summarystats(@view(gflops_range[30:end]))
# findmin(gflops_range)

# julia> search_range = (120:24:120, 700:200:1100, 4000:2000:6000, 0.40:0.01:0.46, 0.85:0.025:1)
# (120:24:120, 700:200:1100, 4000:2000:6000, 0.4:0.01:0.46, 0.85:0.025:1.0)

# julia> getindex.(search_range, Tuple(argmax(gflop_array)))
# (120, 900, 4000, 0.45, 0.875)

# julia> size(gflop_array)
# (1, 3, 2, 7, 7)

# julia> argmax(gflop_array)
# CartesianIndex(1, 2, 1, 6, 2)



# function genmats(N)
#     A = rand(N,N)
#     B = rand(N,N)
#     C = similar(A); p = PaddedMatrices.zstridedpointer
#     C, A, B, p(C), p(A), p(B)
# end
# C10_000,A10_000,B10_000,zc10_000,za10_000,zb10_000 = genmats(10_000);
# @time(PaddedMatrices.jmultpackAB!(zc10_000, za10_000, zb10_000, StaticInt{1}(), StaticInt{0}(), Val(72), Val(1.875), Val(1.05), 10_000,10_000,10_000, PaddedMatrices.CloseOpen(0, 18), Val(1)))



# search_range = (72:24:120, 1.80:0.025:1.9, 1.00:0.25:1.1);
# search_range = (96:24:144, 1.6:0.05:1.9, 1.0:0.05:1.2);



# julia> search_range = (72:24:120, 1.85:0.025:1.95, 1.1:0.25:1.2);

#   Last:  (120, 583, 5562, (1.875, 1.0), 1553.379336642954)
#   Best:  (120, 607, 5346, (1.8, 1.0), 1559.8622396497135)
# (120, 1.8, 1.0)

# julia> gflop_array, best = search(search_range...); getindex.(search_range, Tuple(argmax(gflop_array)))
# Progress: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:05:24
#   Last:  (120, 560, 5265, 1514.4606921365082)
#   Best:  (96, 738, 3996, 1560.6626281274844)
# (96, 1.85, 1.1)

# julia> best_gflop_array, best_search_range = gflop_array, search_range;

# julia> search_range = (72:24:120, 1.80:0.025:1.9, 1.05:0.25:1.1);

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



# using BlackBoxOptim

# # Noisy test problem
# using Random
# function randrosenbrock(x)
#   return sum( 100*( x[2:end] .- x[1:end-1].^2 ).^2 .+ ( x[1:end-1] .- 1 ).^2 ) + randexp()
# end
# res = compare_optimizers(randrosenbrock; SearchRange = (-5.0, 5.0), NumDimensions = 4, MaxFuncEvals = 10_000);
# # :adaptive_de_rand_1_bin_radiuslimited performed well here and is recomended in `BlackBoxOptim`'s README




using BlackBoxOptim

const CsConst, AsConst, BsConst = matrix_range(1_500, 10_000, 100, Float64)

function matmul_objective(W₁, W₂, R₁, R₂)
    print("(W₁ = $(round(W₁, sigdigits=4)), W₂ = $(round(W₂, sigdigits=4)), R₁: $(round(R₁, sigdigits=4)), R₂: $(round(R₂, sigdigits=4))): ")
    gflop = bench_size(CsConst, AsConst, BsConst, Val{W₁}(), Val{W₂}(), Val{R₁}(), Val{R₂}())
    println(gflop)
    - gflop
end


bbres = bboptimize(
    matmul_objective;
    SearchRange = [(0.001, 0.1), (0.01,10.0), (0.25, 0.8), (0.25, 0.99)],
    Method = :adaptive_de_rand_1_bin_radiuslimited,
    MaxTime = 60.0 * 60.0 * 4.0 # 4 hours
)



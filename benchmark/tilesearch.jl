

using PaddedMatrices, VectorizationBase, ProgressMeter

function pick_max_N(::Type{T}, Rm, Rk, Rn) where {T}
    Mc = Rn
    Kc = ( PaddedMatrices.core_cache_size(T, Val{2}()) ÷ Mc ) * Rm
    Nc = ( PaddedMatrices.cache_size(T, Val{3}()) / Kc ) * Rk
    Mc, Kc, Nc
end
pick_max_N(Float64, 0.46, 0.8, 120)


function jmultpackab!(C, A, B, ::Val{M_c}, ::Val{K_c}, ::Val{N_c}, ::Val{R₂}, ::Val{R₃}) where {M_c, K_c, N_c, R₂, R₃}
    M, N = size(C); K = size(B,1)
    zc, za, zb = PaddedMatrices.zstridedpointer.((C,A,B))
    @elapsed(
        PaddedMatrices.jmultpackAB!(
            zc, za, zb, StaticInt{1}(), StaticInt{0}(), Val{M_c}(), Val{K_c}(), Val{N_c}(), Val{R₂}(), Val{R₃}(),
            M, K, N, PaddedMatrices.CloseOpen(0, VectorizationBase.NUM_CORES), Val{VectorizationBase.CACHE_COUNT[3]}()
        )
    )
end

function bench_size(S, Cs, As, Bs, ::Val{M_c}, ::Val{K_c}, ::Val{N_c}, ::Val{R₂}, ::Val{R₃}) where {M_c, K_c, N_c, R₂, R₃}
    jmultpackab!(first(Cs), first(As), first(Bs), Val{M_c}(), Val{K_c}(), Val{N_c}(), Val{R₂}(), Val{R₃}()) # compile
    gflop = 0.0
    for sCAB ∈ zip(S,Cs,As,Bs)
        (s,C,A,B) = sCAB::Tuple{Int,Matrix{Float64},Matrix{Float64},Matrix{Float64}}
        # sleep(0.5)
        t = jmultpackab!(C, A, B, Val{M_c}(), Val{K_c}(), Val{N_c}(), Val{R₂}(), Val{R₃}())
        gf = 2e-9*s*s*s / t
        gflop += gf
    end
    gflop / length(S)
end
function matrix_range(l, u, len)
    S = round.(Int, exp.(range(log(l), stop = log(u), length = 100)))
    As = map(s -> rand(s,s), S)
    Bs = map(s -> rand(s,s), S)
    Cs = map(similar, As)
    S, Cs, As, Bs
end
function gflop_map(S, Cs, As, Bs, ::Val{M_c}, ::Val{K_c}, ::Val{N_c}, ::Val{R₂}, ::Val{R₃}) where {M_c, K_c, N_c, R₂, R₃}
    jmultpackab!(first(Cs), first(As), first(Bs), Val{M_c}(), Val{K_c}(), Val{N_c}(), Val{R₂}(), Val{R₃}())
    gflops = Vector{Float64}(undef, length(S))
    for (i,s) ∈ enumerate(S)
        t = jmultpackab!(Cs[i], As[i], Bs[i], Val{M_c}(), Val{K_c}(), Val{N_c}(), Val{R₂}(), Val{R₃}())
        gflops[i] = 2e-9*s*s*s / t
    end
    gflops
end

function search(
    SCsAsBs = matrix_range(1_500, 10_000, 100), Mc_range = 96:24:144, Kc_range = 700:200:1100, Nc_range = 2000:1000:4000, r₂range = 0.44:0.01:0.47, r₃range = 0.8:0.025:0.875
)
    search_space = Iterators.product(Mc_range, Kc_range, Nc_range, r₂range, r₃range)
    best = Ref(((0,0,0),(0.0,0.0),-Inf))
    gflop_array = let (S,Cs,As,Bs) = SCsAsBs, iter_prod = search_space, p = Progress(length(iter_prod)), best = best
        map(iter_prod) do (M_c, K_c, N_c, R₂, R₃)
            gflops = bench_size(S, Cs, As, Bs, Val(M_c), Val(K_c), Val(N_c), Val(R₂), Val(R₃))
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

search_range = (120:24:120, 700:200:1100, 4000:2000:6000, 0.44:0.01:0.47, 0.80:0.025:0.875)
gflop_array, best = search(search_range...); getindex.(search_range, Tuple(argmax(gflop_array)))


S, Cs, As, Bs = matrix_range(1500, 10_000, 100);

search_range = (120:24:120, 1000:100:1100, 5000:1000:5000, 0.44:0.01:0.47, 0.80:0.025:0.825)
gflop_array, best = search((S,Cs,As,Bs), search_range...); getindex.(search_range, Tuple(argmax(gflop_array)))



gflops_range = gflop_map(S, Cs, As, Bs, Val{120}(), Val{1000}(), Val{5000}(), Val{0.45}(), Val{0.8}());

summarystats(gflops_range)


using StatsBase, UnicodePlots

lineplot(S, gflops_range, title = "Square Matrix GFLOPS", xlabel = "Size", ylabel = "GFLOPS")

S[10]
StatsBase.summarystats(@view(gflops_range[30:end]))
findmin(gflops_range)

# julia> search_range = (120:24:120, 700:200:1100, 4000:2000:6000, 0.40:0.01:0.46, 0.85:0.025:1)
# (120:24:120, 700:200:1100, 4000:2000:6000, 0.4:0.01:0.46, 0.85:0.025:1.0)

# julia> getindex.(search_range, Tuple(argmax(gflop_array)))
# (120, 900, 4000, 0.45, 0.875)

# julia> size(gflop_array)
# (1, 3, 2, 7, 7)

# julia> argmax(gflop_array)
# CartesianIndex(1, 2, 1, 6, 2)



function genmats(N)
    A = rand(N,N)
    B = rand(N,N)
    C = similar(A); p = PaddedMatrices.zstridedpointer
    C, A, B, p(C), p(A), p(B)
end
C10_000,A10_000,B10_000,zc10_000,za10_000,zb10_000 = genmats(10_000);
@time(PaddedMatrices.jmultpackAB!(zc10_000, za10_000, zb10_000, StaticInt{1}(), StaticInt{0}(), Val(72), Val(1.875), Val(1.05), 10_000,10_000,10_000, PaddedMatrices.CloseOpen(0, 18), Val(1)))



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
# Progress: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:05:27
#   Last:  (120, 575, 5373, 1445.457326021271)
#   Best:  (72, 971, 3186, 1551.317089863005)
# (72, 1.875, 1.05)



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




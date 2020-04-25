
using PaddedMatrices, StaticArrays, LinearAlgebra, BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.samples = 1_000_000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

function runbenches(sr)
    bench_results = Matrix{Float64}(undef, length(sr), 5)
    for (i,s) ∈ enumerate(sr)
        Astatic = @SMatrix rand(s, s);
        Bstatic = @SMatrix rand(s, s);
        bench_results[i,1] = @belapsed $(Ref(Astatic))[] * $(Ref(Bstatic))[]
        Amutable = MArray(Astatic);
        Bmutable = MArray(Bstatic);
        Cmutable = similar(Amutable);
        bench_results[i,2] = @belapsed mul!($Cmutable, $Amutable, $Bmutable)
        Afixed = FixedSizeMatrix{s,s,Float64,s}(undef) .= Amutable
        Bfixed = FixedSizeMatrix{s,s,Float64,s}(undef) .= Bmutable
        Cfixed = FixedSizeMatrix{s,s,Float64,s}(undef)
        bench_results[i,3] = @belapsed mul!($Cfixed, $Afixed, $Bfixed)
        Apadded = FixedSizeMatrix{s,s,Float64}(undef) .= Amutable
        Bpadded = FixedSizeMatrix{s,s,Float64}(undef) .= Bmutable
        Cpadded = FixedSizeMatrix{s,s,Float64}(undef)
        bench_results[i,4] = @belapsed mul!($Cpadded, $Apadded, $Bpadded)
        A = Array(Apadded); B = Array(Bpadded); C = similar(A);
        bench_results[i,5] = @belapsed PaddedMatrices.jmul!($C, $A, $B)
        @assert Array(Cmutable) ≈ Cfixed ≈ Cpadded ≈ C
        v = @view(bench_results[i,:])'
        @show s, v
    end
    bench_results
end

br = runbenches(2:40);
using DataFrames, VegaLite

gflops = @. 2e-9 * (2:40) ^ 3 / br;

df = DataFrame(gflops);
names!(df, [:SMatrix, :MMatrix, :FixedSizeArray, :PaddedArray, :DynamicMul]);
df.Size = 2:40

dfs = stack(df, [:SMatrix, :MMatrix, :FixedSizeArray, :PaddedArray, :DynamicMul], variable_name = :MatMulType, value_name = :GFLOPS);
p = dfs |> @vlplot(:line, x = :Size, y = :GFLOPS, width = 900, height = 600, color = {:MatMulType});
save(joinpath(pkgdir(PaddedMatrices), "docs/src/assets/sizedarraysbenchmarks.svg"), p)




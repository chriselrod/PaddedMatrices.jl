
using PaddedMatrices, StaticArrays, LinearAlgebra, BenchmarkTools

BLAS.set_num_threads(1)

# BenchmarkTools.DEFAULT_PARAMETERS.samples = 1_000_000
# BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

function runbenches(sr, ::Type{T} = Float64) where {T}
    bench_results = Matrix{T}(undef, length(sr), 6)
    for (i,s) ∈ enumerate(sr)
        if s ≤ 20
            Astatic = @SMatrix rand(T, s, s);
            Bstatic = @SMatrix rand(T, s, s);
            bench_results[i,1] = @belapsed $(Ref(Astatic))[] * $(Ref(Bstatic))[]
            Amutable = MArray(Astatic);
            Bmutable = MArray(Bstatic);
            Cmutable = similar(Amutable);
        else
            bench_results[i,1] = Inf
            Amutable = MMatrix{s,s,T}(undef)
            Bmutable = MMatrix{s,s,T}(undef)
            Cmutable = MMatrix{s,s,T}(undef)
            @inbounds for i ∈ 1:s^2
                Amutable[i] = rand()
                Bmutable[i] = rand()
            end
        end
        bench_results[i,2] = @belapsed mul!($Cmutable, $Amutable, $Bmutable)
        Afixed = FixedSizeMatrix{s,s,T}(undef) .= Amutable
        Bfixed = FixedSizeMatrix{s,s,T}(undef) .= Bmutable
        Cfixed = FixedSizeMatrix{s,s,T}(undef)
        bench_results[i,3] = @belapsed mul!($Cfixed, $Afixed, $Bfixed)
        Apadded = FixedSizeMatrix{s,s,T}(undef, Val(true)) .= Amutable
        Bpadded = FixedSizeMatrix{s,s,T}(undef, Val(true)) .= Bmutable
        Cpadded = FixedSizeMatrix{s,s,T}(undef, Val(true))
        Cpaddedptr = FixedSizeMatrix{s,s,T}(undef)
        bench_results[i,4] = @belapsed mul!($Cpadded, $Apadded, $Bpadded)
        Aptr = PtrArray(Apadded); Bptr = PtrArray(Bpadded); Cptr = PtrArray(Cpaddedptr);
        GC.@preserve Apadded Bpadded Cpadded begin
            bench_results[i,5] = @belapsed mul!($Cptr, $Aptr, $Bptr)
        end
        A = Array(Apadded); B = Array(Bpadded); C = similar(A);
        bench_results[i,6] = @belapsed PaddedMatrices.jmul!($C, $A, $B)
        @assert Array(Cmutable) ≈ Cfixed ≈ Cpadded ≈ Cpaddedptr ≈ C
        v = @view(bench_results[i,:])'
        @show s, v
    end
    bench_results
end

sizerange = 2:48
br = runbenches(sizerange);
using DataFrames, VegaLite

gflops = @. 2e-9 * (sizerange) ^ 3 / br;

df = DataFrame(gflops);
matmulmethodnames = [:SMatrix, :MMatrix, :FixedSizeArray, :PaddedArray, :PtrArray, :DynamicMul];
names!(df, matmulmethodnames);
df.Size = sizerange

dfs = stack(df, matmulmethodnames, variable_name = :MatMulType, value_name = :GFLOPS);
p = dfs |> @vlplot(:line, x = :Size, y = :GFLOPS, width = 900, height = 600, color = {:MatMulType});
save(joinpath(pkgdir(PaddedMatrices), "docs/src/assets/sizedarraybenchmarks.svg"), p)




# PaddedMatrices

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/PaddedMatrices.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://chriselrod.github.io/PaddedMatrices.jl/latest)
[![Build Status](https://travis-ci.com/chriselrod/PaddedMatrices.jl.svg?branch=master)](https://travis-ci.com/chriselrod/PaddedMatrices.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/chriselrod/PaddedMatrices.jl?svg=true)](https://ci.appveyor.com/project/chriselrod/PaddedMatrices-jl)
[![Codecov](https://codecov.io/gh/chriselrod/PaddedMatrices.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/PaddedMatrices.jl)

# Usage

This library provides a few array types, as well as pure-Julia matrix multiplication.

The native types are optionally statically sized, and optionally given padding (the default) to ensure that all columns are aligned. The following chart shows single-threaded benchmarks on a few different CPUS, comparing:

* `SMatrix` and `MMatrix` multiplication from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl). Beyond 14x14x14, MMatrix will switch to using `LinearAlgebra.BLAS.gemm!`.
* `FixedSizeArray` from this library without any padding.
* `FixedSizeArray` from this library with padding, named `PaddedArray` in the legend.
* The base `Matrix{Float64}` type, using the pure-Julia `PaddedMatrices.jmul!` method.

All matrices were square; the `x`-axis reports size of each dimension. Benchmarks ranged from `2`x`2` matrices through `48`x`48`. The `y`-axis reports double-precision GFLOPS. That is billions of double precision floating point operations per second. Higher is better.

10980XE, a Cascadelake-X CPU with AVX512:
![Cascadelake-X SizedArrayBenchmarks](docs/src/assets/sizedarraybenchmarksAVX512_cascadelakeX.svg)
, a Skylake CPU with AVX2:
![Skylake SizedArrayBenchmarks](docs/src/assets/sizedarraybenchmarksAVX2_skylake.svg)
, a Haswell CPU with AVX2:
![Haswell SizedArrayBenchmarks](docs/src/assets/sizedarraybenchmarksAVX2_haswell.svg)

`MMatrix` performed much better beyond 14x14 relative to the others on Haswell because `LinearAlgebra.BLAS.gemm!` on that computer was using `MKL` instead of `OpenBLAS` (the easiest way to change this is using [MKL.jl](https://github.com/JuliaComputing/MKL.jl)).

`StaticArray`s currently relies on unrolling the operations, and taking advantage of LLVM's [SLP vectorizer](https://llvm.org/docs/Vectorizers.html#the-slp-vectorizer). This approach can work well for very small arrays, but scales poorly. With AVX2, dynamically-sized matrix multiplication of regular `Array{Float64,2}` arrays was faster starting from `7`x`7`, despite not being able to specialize on the size of the arrays, unlike the `SMatrix` and `MMatrix` versions. This also means that the method didn't have to recompile (in order to specialize) on the `7`x`7` `Matrix{Float64}`s.

With AVX512, the `SMatrix` method was faster than the dynamically sized method until the matrices were `9`x`9`, but quickly fell behind after this.

The size-specializing methods for `FixedSizeArray`s and `PtrArray`s matched `SMatrix`'s performance from the beginning, leaving the `SMatrix` method behind starting with `5`x`5` on the AVX2 systems, and `3`x`3` with AVX512.

PaddedMatrices relies on [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl) for code-generation.

One of the goals of PaddedMatrices.jl is to provide good performance across a range of practical sizes.

How does the dynamic `jmul!` compare with OpenBLAS and MKL at larger sizes? Below are more single-threaded `Float64` benchmarks on the 10980XE. Size range from `2`x`2` through `256`x`256`:
![dgemmbenchmarkssmall](docs/src/assets/gemmFloat64_2_256_skylake_AVX512.svg)
Benchmarks from `256`x`256` through `2000`x`2000`:
![dgemmbenchmarksmedium](docs/src/assets/gemmFloat64_256_2000_skylake_AVX512.svg)

Skylake laptop (same as earlier):
![dgemmbenchmarkssmall](docs/src/assets/gemmFloat64_2_256_skylake_AVX2.svg)
![dgemmbenchmarksmedium](docs/src/assets/gemmFloat64_256_2000_skylake_AVX2.svg)

Performance still needs work. In particular
1) Tuning of blocking parameters at larger sizes
2) Possibly switching packed arrays from column to panel-major storage.
3) Diagnosing and fixing the cause of erratic performance.

As an illustration of the last point, consider multiplication of `71`x`71` matrices. Setup:
```julia
julia> using PaddedMatrices, LinuxPerf

julia> M = K = N = 71;

julia> A = rand(M,K); B = rand(K,N); C2 = @time(A * B); C1 = similar(C2);
  0.000093 seconds (2 allocations: 39.516 KiB)

julia> @time(PaddedMatrices.jmul!(C1,A,B)) ≈ C1 # time to first matmul
  9.937127 seconds (21.34 M allocations: 1.234 GiB, 2.46% gc time)
true

julia> foreachmklmul!(C, A, B, N) = foreach(_ -> PaddedMatrices.jmul!(C, A, B), Base.OneTo(N))
```
And then sample `@pstats` results:
```
julia> @pstats "cpu-cycles,(instructions,branch-instructions,branch-misses),(task-clock,context-switches,cpu-migrations,page-faults),(L1-dcache-load-misses,L1-dcache-loads,L1-icache-load-misses),(dTLB-load-misses,dTLB-loads),(iTLB-load-misses,iTLB-loads)" (@time foreachmul!(C2, A, B, 1_000_000))
  6.701603 seconds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╶ cpu-cycles               2.74e+10   42.9%  #  4.1 cycles per ns
┌ instructions             7.96e+10   28.6%  #  2.9 insns per cycle
│ branch-instructions      1.80e+09   28.6%  #  2.3% of instructions
└ branch-misses            3.39e+06   28.6%  #  0.2% of branch instructions
┌ task-clock               6.70e+09  100.0%
│ context-switches         9.00e+00  100.0%
│ cpu-migrations           0.00e+00  100.0%
└ page-faults              0.00e+00  100.0%
┌ L1-dcache-load-misses    6.78e+09   14.3%  # 26.4% of loads
│ L1-dcache-loads          2.57e+10   14.3%
└ L1-icache-load-misses    3.08e+06   14.3%
┌ dTLB-load-misses         3.43e+02   14.3%
└ dTLB-loads               2.56e+10   14.3%
┌ iTLB-load-misses         1.03e+05   28.6%
└ iTLB-loads               3.16e+04   28.6%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

julia> @pstats "cpu-cycles,(instructions,branch-instructions,branch-misses),(task-clock,context-switches,cpu-migrations,page-faults),(L1-dcache-load-misses,L1-dcache-loads,L1-icache-load-misses),(dTLB-load-misses,dTLB-loads),(iTLB-load-misses,iTLB-loads)" (@time foreachmul!(C2, A, B, 1_000_000))
  7.222783 seconds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╶ cpu-cycles               2.95e+10   42.9%  #  4.1 cycles per ns
┌ instructions             7.96e+10   28.6%  #  2.7 insns per cycle
│ branch-instructions      1.80e+09   28.6%  #  2.3% of instructions
└ branch-misses            3.50e+06   28.6%  #  0.2% of branch instructions
┌ task-clock               7.22e+09  100.0%
│ context-switches         1.00e+01  100.0%
│ cpu-migrations           0.00e+00  100.0%
└ page-faults              0.00e+00  100.0%
┌ L1-dcache-load-misses    6.81e+09   14.3%  # 26.6% of loads
│ L1-dcache-loads          2.57e+10   14.3%
└ L1-icache-load-misses    3.38e+06   14.3%
┌ dTLB-load-misses         2.10e+02   14.3%
└ dTLB-loads               2.57e+10   14.3%
┌ iTLB-load-misses         1.12e+05   28.6%
└ iTLB-loads               3.60e+04   28.6%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
The difference in `L1-dcache-load-misses` looks relatively minor between these runs, yet we see nearly an 8% difference in performance and instructions per clock cycle. The first time corresponds to a mean of nearly 107 GFLOPS across the 1 million matrix multiplications, while the second corresponds to less than 100. I'm still investigating the cause. I do not see such erratic performance with MKL, for example.


Additionally, the library uses [VectorizedRNG.jl](https://github.com/chriselrod/VectorizedRNG.jl) for random number generation. Unfortunately, here is where we pay the price of GC.
```julia
julia> using PaddedMatrices, StaticArrays, BenchmarkTools

julia> @benchmark @SMatrix rand(8,8)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     90.861 ns (0.00% GC)
  median time:      91.515 ns (0.00% GC)
  mean time:        91.556 ns (0.00% GC)
  maximum time:     122.468 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     968

julia> @benchmark @FixedSize rand(8,8)
BenchmarkTools.Trial:
  memory estimate:  576 bytes
  allocs estimate:  1
  --------------
  minimum time:     52.154 ns (0.00% GC)
  median time:      169.038 ns (0.00% GC)
  mean time:        189.961 ns (20.91% GC)
  maximum time:     18.303 μs (99.10% GC)
  --------------
  samples:          10000
  evals/sample:     986

julia> @benchmark @SMatrix randn(8,8)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     220.342 ns (0.00% GC)
  median time:      227.329 ns (0.00% GC)
  mean time:        227.426 ns (0.00% GC)
  maximum time:     430.583 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     535

julia> @benchmark @FixedSize randn(8,8)
BenchmarkTools.Trial:
  memory estimate:  576 bytes
  allocs estimate:  1
  --------------
  minimum time:     133.292 ns (0.00% GC)
  median time:      262.179 ns (0.00% GC)
  mean time:        280.922 ns (13.84% GC)
  maximum time:     20.800 μs (99.02% GC)
  --------------
  samples:          10000
  evals/sample:     873
```
Thus, it is recomended you either preallocate and mutate existing arrays, or find some other approach to working with these.
In the future, I'll try to ensure that a large number of basic functions and operations (e.g. matrix multiplication, broadcasting, creation)
inline for small arrays, so that the compiler will be able to stack-allocate them and avoid the heap and GC altogether, so long as they don't escape.

These arrays also use [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl) for broadcasts:
```julia
julia> using PaddedMatrices, StaticArrays, BenchmarkTools

julia> Afs = @FixedSize randn(13,29); Asm = SMatrix{13,29}(Array(Afs));

julia> bfs = @FixedSize rand(13); bsv = SVector{13}(bfs);

julia> cfs = @FixedSize rand(29); csv = SVector{29}(cfs);

julia> Dfs = @. exp(Afs) + bfs * log(cfs');

julia> Dfs ≈ @. exp(Asm) + bsv * log(csv')
true

julia> @benchmark @. exp($Afs) + $bfs * log($cfs') # FixedSizeArrays, allocating
BenchmarkTools.Trial:
  memory estimate:  3.75 KiB
  allocs estimate:  1
  --------------
  minimum time:     659.503 ns (0.00% GC)
  median time:      715.554 ns (0.00% GC)
  mean time:        850.659 ns (11.74% GC)
  maximum time:     51.319 μs (96.06% GC)
  --------------
  samples:          10000
  evals/sample:     149

julia> @benchmark @. exp($Asm) + $bsv * log($csv') # StaticArrays, non-allocating but much slower
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     3.676 μs (0.00% GC)
  median time:      3.686 μs (0.00% GC)
  mean time:        3.691 μs (0.00% GC)
  maximum time:     6.073 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     8

julia> @benchmark @. $Dfs = exp($Afs) + $bfs * log($cfs') # FixedSizeArrays, using pre-allocated output
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     496.433 ns (0.00% GC)
  median time:      498.995 ns (0.00% GC)
  mean time:        499.542 ns (0.00% GC)
  maximum time:     624.820 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     194
  ```



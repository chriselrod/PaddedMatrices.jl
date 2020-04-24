# PaddedMatrices

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/PaddedMatrices.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://chriselrod.github.io/PaddedMatrices.jl/latest)
[![Build Status](https://travis-ci.com/chriselrod/PaddedMatrices.jl.svg?branch=master)](https://travis-ci.com/chriselrod/PaddedMatrices.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/chriselrod/PaddedMatrices.jl?svg=true)](https://ci.appveyor.com/project/chriselrod/PaddedMatrices-jl)
[![Codecov](https://codecov.io/gh/chriselrod/PaddedMatrices.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/PaddedMatrices.jl)

# Usage

This library provides a few array types, as well as pure-Julia matrix multiplication.

The native types are optionally statically sized, and optionally given padding to ensure that all columns are aligned. The following chart shows benchmarks on a 10980XE CPU, comparing:

* `SMatrix` and `MMatrix` multiplication from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
* `FixedSizeArray` from this library without any padding.
* `FixedSizeArray` from this library with padding, named `PaddedArray` in the legend.
* The base `Matrix{Float64}` type, using the `PaddedMatrices.jmul!` method.

![SizedArrayBenchmarks](docs/src/assets/sizedarrayBenchmarks.svg)

All matrices were square and filled with `Float64` elements. Size refers to the number of rows and columns.
Inplace multiplication was used for all but the `SArray`. For the `FixedSizeArray`s, `LinearAlgebra.mul!` simple redirects to `jmul!`, which is capable of taking advantage of static size information.

`StaticArray`s currently relies on unrolling the operations, and taking advantage of LLVM's powerful [SLP vectorizer](https://llvm.org/docs/Vectorizers.html#the-slp-vectorizer). It performs best for `2x2`, `3x3`, `4x4`, `6x6`, and `8x8` matrices on this architecture. Between stack allocation and marking these operations for inline, `SMatrix` achieves better performance than the alternatives at these sizes.

PaddedMatrices relies on [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl) to generate microkernels. Perhaps I should make it unroll more agressively at small static sizes, and also mark it for inlining. For now, it doesn't achieve quite the same performance as an `SMatrix` at `8x8`, `6x6`, and `4x4` and below. However, at `9x9` and beyond, even the dynamically sized `PaddedMatrices.jmul!` method achieves better performance than `SMatrix` or `MMatrix`. Of course, `StaticArrays` is primarily concerned with `10x10` matrices and smaller, and the `SMatrix` type allows you to use the more convenient non-mutating API without worrying about allocations or memory management.

How does `jmul!` compare with OpenBLAS and MKL at larger sizes? Single threaded `Float64` benchmarks:
![dgemmbenchmarks](docs/src/assets/gemmf64.svg)
It's slower than both OpenBLAS and MKL, but (on this architecture) it's closer to MKL than MKL is to OpenBLAS at large sizes. I also added [Gaius.jl](https://github.com/MasonProtter/Gaius.jl) for comparison. It also uses LoopVectorization to generate the microkernels, but uses divide and conquer to improve cache locality, rather than tiling and packing like the others. The divide and conquer approach yields much better performance than not handling cache locality; I may add a naive implementation for comparison for purposes of comparison eventually, but Julia's generic matmul -- which still makes some effort for cache optimality -- could give some perspective in the integer benchmarks below.

But before moving onto integers, `Float32` benchmarks:
![sgemmbenchmarks](docs/src/assets/gemmf32.svg)
Both BLAS libraries again beat `jmul!`. OpenBLAS and MKL are now neck and neck.

The BLAS libraries do not support integer multiplication, so the comparison is now with Julia's [generic matmul](https://github.com/JuliaLang/julia/blob/b1f51df1088b2ab4e1c954537fd8c22b9b5f19ac/stdlib/LinearAlgebra/src/matmul.jl#L730); `Int64`:
![i64gemmbenchmarks](docs/src/assets/gemmi64.svg)
64-bit integer multiplication is very slow on most platforms. With AVX2, it is implemented with repeated 32-bit integer multiplications, shifts, and additions (`(a + b)*(c + d) = ad + bc + bd`; you can drop the `ac` because it overflows). With AVX512 (like the benchmark rig), it uses the `vpmullq` instruction, which is slow.

![i32gemmbenchmarks](docs/src/assets/gemmi32.svg)
`Int32` multiplication is much faster, but still lags behind `Float64` performance.
For some reason, the generic matmul is slower for `Int32`; I have not investigated why.



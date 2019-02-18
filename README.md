# PaddedMatrices

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/PaddedMatrices.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://chriselrod.github.io/PaddedMatrices.jl/latest)
[![Build Status](https://travis-ci.com/chriselrod/PaddedMatrices.jl.svg?branch=master)](https://travis-ci.com/chriselrod/PaddedMatrices.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/chriselrod/PaddedMatrices.jl?svg=true)](https://ci.appveyor.com/project/chriselrod/PaddedMatrices-jl)
[![Codecov](https://codecov.io/gh/chriselrod/PaddedMatrices.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/PaddedMatrices.jl)

# Usage

This is an optimized linear algebra library. To aid optimization, it provides various memory layouts.
If you have an application that would benefit from a certain memory layout that isn't supported, a PR is welcome!

The simplest example of this is the straightforward padded matrix, where the stride between columns is a multiple of SIMD vector width.
For example, a 7x7 matrix will have a stride of 8 between columns, so that on avx2 architectures a column can be loaded into registers as 2 vectors / avx512 architectures can load the column as a single vector.
If the stride were 7, the naive workaround would be to load in three pieces of lengths 4, 2, and 1. (Masked load/store operations are an alternative, but they currently aren't supported with `structs` in Julia, because we can't get pointers to them.)

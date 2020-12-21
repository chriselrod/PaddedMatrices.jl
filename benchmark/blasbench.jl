
#using Gaius

# using Gaius, MaBLAS, PaddedMatrices, StructArrays, LinearAlgebra, BenchmarkTools
# using MaBLAS, PaddedMatrices, StructArrays, LinearAlgebra, BenchmarkTools
using PaddedMatrices, StructArrays, LinearAlgebra, BenchmarkTools, Libdl
# BLAS.set_num_threads(1); Base.Threads.nthreads()
# For laptops that thermally throttle, you can set the `JULIA_SLEEP_BENCH` environment variable for #seconds to sleep before each `@belapsed`
const SLEEPTIME = parse(Float64, get(ENV, "JULIA_SLEEP_BENCH", "0"))
maybe_sleep() = iszero(SLEEPTIME) || sleep(SLEEPTIME)

# function check_if_should_pack(C, A, cache_params)
#     M, K = size(A)
#     N = size(C,2)
#     mc, kc, nc = cache_params
#     pack_a = M > 72 && !((2M ≤ 5mc) & iszero(stride(A,2) % MaBLAS.VectorizationBase.pick_vector_width(eltype(C))))
#     pack_b = K * N > kc * nc
#     pack_a, pack_b
# end

# # for 16x14m perhaps bigger k than in 256 x 263 x 4928
# function ma24x9!(C, A, B)
#     cache_params = (cache_m = 192, cache_k = 353, cache_n = 7119)
#     dopack = check_if_should_pack(C, A, cache_params)
#     MaBLAS.mul!(C, A, B; packing=dopack, cache_params=cache_params, kernel_params=(Val(24), Val(9)))
# end
# function ma32x6!(C, A, B)
#     cache_params = (cache_m = 128, cache_k = 529, cache_n = 2454)
#     dopack = check_if_should_pack(C, A, cache_params)
#     MaBLAS.mul!(C, A, B; packing=dopack, cache_params=cache_params, kernel_params=(Val(32), Val(6)))
# end
# function ma40x5!(C, A, B)
#     cache_params = (cache_m = 120, cache_k = 532, cache_n = 2440)
#     dopack = check_if_should_pack(C, A, cache_params)
#     MaBLAS.mul!(C, A, B; packing=dopack, cache_params=cache_params, kernel_params=(Val(40), Val(5)))
# end

randa(::Type{T}, dim...) where {T} = rand(T, dim...)
randa(::Type{T}, dim...) where {T <: Signed} = rand(T(-100):T(200), dim...)

using MKL_jll, OpenBLAS_jll

const libMKL = Libdl.dlopen(MKL_jll.libmkl_rt)
const DGEMM_MKL = Libdl.dlsym(libMKL, :dgemm)
const SGEMM_MKL = Libdl.dlsym(libMKL, :sgemm)
const DGEMV_MKL = Libdl.dlsym(libMKL, :dgemv)
const MKL_SET_NUM_THREADS = Libdl.dlsym(libMKL, :MKL_Set_Num_Threads)

const libOpenBLAS = Libdl.dlopen(OpenBLAS_jll.libopenblas)
const DGEMM_OpenBLAS = Libdl.dlsym(libOpenBLAS, :dgemm_64_)
const SGEMM_OpenBLAS = Libdl.dlsym(libOpenBLAS, :sgemm_64_)
const DGEMV_OpenBLAS = Libdl.dlsym(libOpenBLAS, :dgemv_64_)
const OPENBLAS_SET_NUM_THREADS = Libdl.dlsym(libOpenBLAS, :openblas_set_num_threads64_)

istransposed(x) = 'N'
istransposed(x::Adjoint{<:Real}) = 'T'
istransposed(x::Adjoint) = 'C'
istransposed(x::Transpose) = 'T'
for (lib,f) ∈ [(:GEMM_MKL,:gemmmkl!), (:GEMM_OpenBLAS,:gemmopenblas!)]
    for (T,prefix) ∈ [(Float32,:S),(Float64,:D)]
        fm = Symbol(prefix, lib)
        @eval begin
            function $f(C::AbstractMatrix{$T}, A::AbstractMatrix{$T}, B::AbstractMatrix{$T})
                transA = istransposed(A)
                transB = istransposed(B)
                M, N = size(C); K = size(B, 1)
                pA = parent(A); pB = parent(B)
                ldA = stride(pA, 2)
                ldB = stride(pB, 2)
                ldC = stride(C, 2)
                α = one($T)
                β = zero($T)
                ccall(
                    $fm, Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ref{Int64}, Ref{$T}, Ref{$T},
                     Ref{Int64}, Ref{$T}, Ref{Int64}, Ref{$T}, Ref{$T}, Ref{Int64}),
                    transA, transB, M, N, K, α, pA, ldA, pB, ldB, β, C, ldC
                )
            end
        end
    end
end
mkl_set_num_threads(N::Integer) = ccall(MKL_SET_NUM_THREADS, Cvoid, (Int32,), N % Int32)
mkl_set_num_threads(1)
openblas_set_num_threads(N::Integer) = ccall(OPENBLAS_SET_NUM_THREADS, Cvoid, (Int64,), N)
openblas_set_num_threads(1)

function benchmark_fun!(f!, C, A, B, force_belapsed = false, reference = nothing)
    maybe_sleep()
    tmin = @elapsed f!(C, A, B)
    if force_belapsed || 2tmin < BenchmarkTools.DEFAULT_PARAMETERS.seconds
        maybe_sleep()
        tmin = min(tmin, @belapsed $f!($C, $A, $B))
    elseif tmin < BenchmarkTools.DEFAULT_PARAMETERS.seconds
        maybe_sleep()
        tmin = min(tmin, @elapsed f!(C, A, B))
        if tmin < 2BenchmarkTools.DEFAULT_PARAMETERS.seconds
            maybe_sleep()
            tmin = min(tmin, @elapsed f!(C, A, B))
        end
    end
    isnothing(reference) || @assert C ≈ reference
    tmin
end


# [2:255..., ( round.(Int, range(26.15105483720698,length=201) .^ 1.6989476505010863))...]
function runbench(::Type{T}, sizes = [2:255..., round.(Int, range(57.16281374121401, length=200) .^ 1.3705658916944428)...]) where {T}
    (StructVector ∘ map)(sizes) do sz
        n, k, m = sz, sz, sz
        C1 = Matrix{T}(undef, n, m)
        C2 = similar(C1);
        C3 = similar(C1);
        # C4 = similar(C1);
        # C5 = similar(C1);
        # C6 = similar(C1);
        A  = randa(T, n, k)
        B  = randa(T, k, m)
        # BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.05

        # ma24 = benchmark_fun!(ma24x9!, C3, A, B, sz == first(sizes), C1)
        # ma32 = benchmark_fun!(ma32x6!, C4, A, B, sz == first(sizes), C1)
        # ma40 = benchmark_fun!(ma40x5!, C5, A, B, sz == first(sizes), C1)
        # gt = benchmark_fun!(blocked_mul!, C5, A, B, sz == first(sizes), C1)
        jmlt = benchmark_fun!(PaddedMatrices.jmul!, C1, A, B, sz == first(sizes))
        res = if T <: Integer
            (matrix_size=sz, MaBLAS_24x9=ma24, MaBLAS_32x6=ma32, MaBLAS_40x5=ma40, PaddedMatrices=jmlt)
        else
            opbt = benchmark_fun!(gemmopenblas!, C2, A, B, sz == first(sizes), C1)
            mklbt= benchmark_fun!(gemmmkl!, C3, A, B, sz == first(sizes), C1)
            # (matrix_size=sz, OpenBLAS=opbt, MKL=mklbt, MaBLAS_24x9=ma24, MaBLAS_32x6=ma32, MaBLAS_40x5=ma40, PaddedMatrices=jmlt)
            (matrix_size=sz, OpenBLAS=opbt, MKL=mklbt, PaddedMatrices=jmlt)
        end
        @show res
    end
end
#tf64 = runbench(Float64);

#=
tf32 = runbench(Float32);
ti64 = runbench(Int64);
ti32 = runbench(Int32);
=#

calcgflops(sz, st) = 2e-9 * sz^3 /st
using VectorizationBase: REGISTER_SIZE, FMA
# I don't know how to query GHz;
# Your best bet would be to check your bios
# Alternatives are to look up your CPU model or `watch -n1 "cat /proc/cpuinfo | grep MHz"`
# Boosts and avx downclocking complicate it.
const GHz = 4.1 
const W64 = REGISTER_SIZE ÷ sizeof(Float64) # vector width
const FMA_RATIO = FMA ? 2 : 1
const INSTR_PER_CLOCK = 2 # I don't know how to query this, but true for most recent CPUs
const PEAK_DGFLOPS = GHz * W64 * FMA_RATIO * INSTR_PER_CLOCK

using DataFrames, VegaLite
function create_df(res)
    df = DataFrame()
    for field in fieldnames(typeof(fieldarrays(res)))
        setproperty!(df, field, getproperty(res, field))
    end
    rename!(df, :matrix_size => :Size)
    dfs = stack(df, Not(:Size), variable_name = :Library, value_name = :Time);
    dfs.GFLOPS = calcgflops.(dfs.Size, dfs.Time);
    dfs
end

function pick_suffix(desc = "")
    suffix = if PaddedMatrices.VectorizationBase.AVX512F
        "AVX512"
    elseif PaddedMatrices.VectorizationBase.AVX2
        "AVX2"
    elseif PaddedMatrices.VectorizationBase.REGISTER_SIZE == 32
        "AVX"
    else
        "REGSUZE$(PaddedMatrices.VectorizationBase.REGISTER_SIZE)"
    end
    if desc != ""
        suffix *= '_' * desc
    end
    "$(Sys.CPU_NAME)_$suffix"
end

function plot(tf, ::Type{T} = Float64, desc = "") where {T}
    res = create_df(tf)
    l, u = extrema(tf.matrix_size)
    plt = res |> @vlplot(
        :line, color = :Library,
       # x = {:Size, scale={type=:log}}, y = {:GFLOPS},#, scale={type=:log}},
        x = {:Size}, y = {:GFLOPS},#, scale={type=:log}},
        width = 1200, height = 600
    )
    suffix = pick_suffix(desc)
    save(joinpath(pkgdir(PaddedMatrices), "docs/src/assets/gemm$(string(T))_$(l)_$(u)_$(suffix).svg"), plt)
    save(joinpath(pkgdir(PaddedMatrices), "docs/src/assets/gemm$(string(T))_$(l)_$(u)_$(suffix).png"), plt)
end

#= 
tf64 = runbench(Float64);
plot(tf64[begin:255])
plot(tf64[255:end])
=#

#=
using MaBLAS
using Base.Threads; Threads.nthreads()
function findblock(C, A, B, Mc, Kc, Nc)
    res = Array{Float64}(undef, length(Mc), length(Kc), length(Nc))
    Threads.@sync for i ∈ eachindex(Nc), j ∈ eachindex(Kc), k ∈ eachindex(Mc)
        Threads.@spawn begin
            res[k,j,i] = @belapsed MaBLAS.mul!($(copy(C)), $A, $B, packing=true, cache_params = $(Mc[k], Kc[j], Nc[i]), kernel_params=(Val(16), Val(12)))
        end
    end
    res
end

res = findblock(C1, A, B, 96:16:192, 300:325:500, 2000:500:4500)

=#

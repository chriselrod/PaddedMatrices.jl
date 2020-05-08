
#using Gaius

using Gaius, MaBLAS, PaddedMatrices, StructArrays, LinearAlgebra, BenchmarkTools
BLAS.set_num_threads(1); Base.Threads.nthreads()

function check_if_should_pack(C, A, cache_params)
    M, K = size(A)
    N = size(C,2)
    mc, kc, nc = cache_params
    pack_a = M > 72 && !((2M ≤ 5mc) & iszero(stride(A,2) % MaBLAS.VectorizationBase.pick_vector_width(eltype(C))))
    pack_b = K * N > kc * nc
    pack_a, pack_b
end

# for 16x14m perhaps bigger k than in 256 x 263 x 4928
function ma24x9!(C, A, B)
    cache_params = (cache_m = 192, cache_k = 353, cache_n = 7119)
    dopack = check_if_should_pack(C, A, cache_params)
    MaBLAS.mul!(C, A, B; packing=dopack, cache_params=cache_params, kernel_params=(Val(24), Val(9)))
end
function ma32x6!(C, A, B)
    cache_params = (cache_m = 128, cache_k = 529, cache_n = 2454)
    dopack = check_if_should_pack(C, A, cache_params)
    MaBLAS.mul!(C, A, B; packing=dopack, cache_params=cache_params, kernel_params=(Val(32), Val(6)))
end
function ma40x5!(C, A, B)
    cache_params = (cache_m = 120, cache_k = 532, cache_n = 2440)
    dopack = check_if_should_pack(C, A, cache_params)
    MaBLAS.mul!(C, A, B; packing=dopack, cache_params=cache_params, kernel_params=(Val(40), Val(5)))
end

randa(::Type{T}, dim...) where {T} = rand(T, dim...)
randa(::Type{T}, dim...) where {T <: Signed} = rand(T(-100):T(200), dim...)

const LIBDIRECTCALLJIT = "/home/chriselrod/.julia/dev/LoopVectorization/benchmark/libdcjtest.so"
istransposed(x) = false
istransposed(x::Adjoint) = true
istransposed(x::Transpose) = true
mkl_set_num_threads(N::Integer) = ccall((:set_num_threads, LIBDIRECTCALLJIT), Cvoid, (Ref{UInt32},), Ref(N % UInt32))
function mklmul!(C::AbstractVecOrMat{Float32}, A::AbstractVecOrMat{Float32}, B::AbstractVecOrMat{Float32})
    M, N = size(C); K = size(B, 1)
    ccall(
        (:sgemmjit, LIBDIRECTCALLJIT), Cvoid,
        (Ptr{Float32},Ptr{Float32},Ptr{Float32},Ref{UInt32},Ref{UInt32},Ref{UInt32},Ref{Bool},Ref{Bool}),
        parent(C), parent(A), parent(B),
        Ref(M % UInt32), Ref(K % UInt32), Ref(N % UInt32),
        Ref(istransposed(A)), Ref(istransposed(B))
    )
end
function mklmul!(C::AbstractVecOrMat{Float64}, A::AbstractVecOrMat{Float64}, B::AbstractVecOrMat{Float64})
    M, N = size(C); K = size(B, 1)
    ccall(
        (:dgemmjit, LIBDIRECTCALLJIT), Cvoid,
        (Ptr{Float64},Ptr{Float64},Ptr{Float64},Ref{UInt32},Ref{UInt32},Ref{UInt32},Ref{Bool},Ref{Bool}),
        parent(C), parent(A), parent(B),
        Ref(M % UInt32), Ref(K % UInt32), Ref(N % UInt32),
        Ref(istransposed(A)), Ref(istransposed(B))
    )
end

mkl_set_num_threads(1)


function benchmark_fun!(f!, C, A, B, force_belapsed = false, reference = nothing)
    tmin = @elapsed f!(C, A, B)
    if force_belapsed || tmin < BenchmarkTools.DEFAULT_PARAMETERS.seconds
        tmin = min(tmin, @belapsed $f!($C, $A, $B))
    elseif tmin < 2BenchmarkTools.DEFAULT_PARAMETERS.seconds
        tmin = min(tmin, @elapsed f!(C, A, B))
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
        C4 = similar(C1);
        C5 = similar(C1);
        C6 = similar(C1);
        A  = randa(T, n, k)
        B  = randa(T, k, m)
        BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.05

        opbt = benchmark_fun!(mul!, C1, A, B, sz == first(sizes))
        ma24 = benchmark_fun!(ma24x9!, C2, A, B, sz == first(sizes), C1)
        ma32 = benchmark_fun!(ma32x6!, C3, A, B, sz == first(sizes), C1)
        ma40 = benchmark_fun!(ma40x5!, C4, A, B, sz == first(sizes), C1)
        gt = benchmark_fun!(blocked_mul!, C5, A, B, sz == first(sizes), C1)
        jmlt = benchmark_fun!(PaddedMatrices.jmul!, C6, A, B, sz == first(sizes), C1)
        res = if T <: Integer
            (matrix_size=sz, OpenBLAS=opbt, MaBLAS_24x9=ma24, MaBLAS_32x6=ma32, MaBLAS_40x5=ma40, Gaius=gt, PaddedMatrices=jmlt)
        else
            C7 = similar(C1);
            mklb = benchmark_fun!(mklmul!, C7, A, B, sz == first(sizes), C1)
            (matrix_size=sz, OpenBLAS=opbt, MaBLAS_24x9=ma24, MaBLAS_32x6=ma32, MaBLAS_40x5=ma40, Gaius=gt, PaddedMatrices=jmlt, MKL = mklb)
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

gflops(sz, st) = 2e-9 * sz^3 /st
using VectorizationBase: REGISTER_SIZE, FMA3
# I don't know how to query GHz;
# Your best bet would be to check your bios
# Alternatives are to look up your CPU model or `watch -n1 "cat /proc/cpuinfo | grep MHz"`
# Boosts and avx downclocking complicate it.
const GHz = 4.1 
const W64 = REGISTER_SIZE ÷ sizeof(Float64) # vector width
const FMA_RATIO = FMA3 ? 2 : 1
const INSTR_PER_CLOCK = 2 # I don't know how to query this, but true for most recent CPUs
const PEAK_DGFLOPS = GHz * W64 * FMA_RATIO * INSTR_PER_CLOCK

using DataFrames, VegaLite
function create_float_df(res, nbytes)
    df = DataFrame(
        Size = res.matrix_size,
        MaBLAS_24x9 = res.MaBLAS_24x9,
        MaBLAS_32x6 = res.MaBLAS_32x6,
        MaBLAS_40x5 = res.MaBLAS_40x5,
        Gaius = res.Gaius,
        PaddedMatrices = res.PaddedMatrices,
        OpenBLAS = res.OpenBLAS,
        MKL = res.MKL
    );
#    dfs = stack(df, [:Gaius, :PaddedMatrices, :OpenBLAS, :MKL], variable_name = :Library, value_name = :Time);
    dfs = stack(df, [:MaBLAS_24x9, :MaBLAS_32x6, :MaBLAS_40x5, :Gaius, :PaddedMatrices, :OpenBLAS, :MKL], variable_name = :Library, value_name = :Time);
    dfs.GFLOPS = gflops.(dfs.Size, dfs.Time);
    dfs.Percent_Peak = 100 .* dfs.GFLOPS .* (8 ÷ nbytes) ./ PEAK_DGFLOPS;
    dfs
end
function create_int_df(res, nbytes)
    df = DataFrame(
        Size = res.matrix_size,
        MaBLAS_24x9 = res.MaBLAS_24x9,
        MaBLAS_32x6 = res.MaBLAS_32x6,
        MaBLAS_40x5 = res.MaBLAS_40x5,
        Gaius = res.Gaius,
        PaddedMatrices = res.PaddedMatrices,
        GenericMatMul = res.OpenBLAS,
    );
    dfs = stack(df, [:MaBLAS_24x9, :MaBLAS_32x6, :MaBLAS_40x5, :Gaius, :PaddedMatrices, :GenericMatMul], variable_name = :Library, value_name = :Time);
    dfs.GFLOPS = gflops.(dfs.Size, dfs.Time);
 #   dfs.Percent_Peak = 100 .* dfs.GFLOPS .* (FMA_RATIO * 8 ÷ nbytes) ./ PEAK_DGFLOPS;
    dfs
end
create_df(res, ::Type{T}) where {T} = create_float_df(res, sizeof(T))
create_df(res, ::Type{T}) where {T<:Integer} = create_int_df(res, sizeof(T))

#using PaddedMatrices
#PICTURES = joinpath(pkgdir(PaddedMatrices), "docs", "src", "assets")
PICTURES = "/home/chriselrod/Pictures"


function plot(tf, ::Type{T} = Float64, PICTURES = "/home/chriselrod/Pictures") where {T}
    res = create_df(tf, T)
    plt = res |> @vlplot(
        :line, color = :Library,
       # x = {:Size, scale={type=:log}}, y = {:GFLOPS},#, scale={type=:log}},
        x = {:Size}, y = {:GFLOPS},#, scale={type=:log}},
        width = 2400, height = 600
    )
    save(joinpath(PICTURES, "gemm$(string(T)).png"), plt)
end



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

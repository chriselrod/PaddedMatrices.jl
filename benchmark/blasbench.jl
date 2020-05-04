
#using Gaius

using MaBLAS, StructArrays, LinearAlgebra, BenchmarkTools
using PaddedMatrices: jmul!
BLAS.set_num_threads(1); Base.Threads.nthreads()

function blocked_mul!(C, A, B)
    mc = 160
    dopack =  2size(C,1) > 5mc
    MaBLAS.mul!(C, A, B; packing=dopack, cache_params=(cache_m=mc, cache_k=400, cache_n=4080), kernel_params=(Val(16), Val(12)))
end
function blocked_mul40x5!(C, A, B)
    mc = 160
    dopack =  2size(C,1) > 5mc
    MaBLAS.mul!(C, A, B; packing=dopack, cache_params=(cache_m=mc, cache_k=400, cache_n=4080), kernel_params=(Val(40), Val(5)))
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

function runbench(::Type{T}, sizes = [2, 4, 8:5:256..., ( (16:63) .^ 2)...]) where {T}
    (StructVector ∘ map)(sizes) do sz
        n, k, m = sz, sz, sz
        C1 = zeros(T, n, m)
        C2 = zeros(T, n, m)
        C3 = zeros(T, n, m)
        C4 = zeros(T, n, m)
        C5 = zeros(T, n, m)
        A  = randa(T, n, k)
        B  = randa(T, k, m)
        BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.05

        opb = @elapsed mul!(C1, A, B)
        if 2opb < BenchmarkTools.DEFAULT_PARAMETERS.seconds
            opb = min(opb, @belapsed mul!($C1, $A, $B))         #samples=100
        end
        lvb = @elapsed blocked_mul!(C2, A, B)
        if 2lvb < BenchmarkTools.DEFAULT_PARAMETERS.seconds
            lvb = min(lvb, @belapsed blocked_mul!($C2, $A, $B)) #samples=100
        end
        @assert C1 ≈ C2
        mab = @elapsed blocked_mul40x5!(C2, A, B)
        if 2mab < BenchmarkTools.DEFAULT_PARAMETERS.seconds
            mab = min(mab, @belapsed blocked_mul40x5!($C2, $A, $B)) #samples=100
        end
        @assert C1 ≈ C2
        pmb = @elapsed jmul!(C3, A, B)
        if 2pmb < BenchmarkTools.DEFAULT_PARAMETERS.seconds
            pmb = min(pmb, @belapsed jmul!($C3, $A, $B))         #samples=100
        end
        @assert C1 ≈ C3
        if T <: Integer
            res = (matrix_size=sz, lvBLAS=lvb, OpenBLAS=opb, PaddedMatrices = pmb)
            @show res
        else
            mklb = @elapsed mklmul!(C4, A, B)
            if 2mklb < BenchmarkTools.DEFAULT_PARAMETERS.seconds
                mklb = min(mklb, @belapsed mklmul!($C4, $A, $B))         #samples=100
            end
            @assert C1 ≈ C4
            res = (matrix_size=sz, MaBLAS_16x12=lvb, MaBLAS_40x5=mab, OpenBLAS=opb, PaddedMatrices = pmb, MKL = mklb)
            @show res
        end
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
        MaBLAS_16x12 = res.MaBLAS_16x12,
        MaBLAS_40x5 = res.MaBLAS_40x5,
        PaddedMatrices = res.PaddedMatrices,
        OpenBLAS = res.OpenBLAS,
        MKL = res.MKL
    );
#    dfs = stack(df, [:Gaius, :PaddedMatrices, :OpenBLAS, :MKL], variable_name = :Library, value_name = :Time);
    dfs = stack(df, [:MaBLAS_16x12, :MaBLAS_40x5, :PaddedMatrices, :OpenBLAS, :MKL], variable_name = :Library, value_name = :Time);
    dfs.GFLOPS = gflops.(dfs.Size, dfs.Time);
    dfs.Percent_Peak = 100 .* dfs.GFLOPS .* (8 ÷ nbytes) ./ PEAK_DGFLOPS;
    dfs
end
function create_int_df(res, nbytes)
    df = DataFrame(
        Size = res.matrix_size,
        MaBLAS_16x12 = res.MaBLAS_16x12,
        MaBLAS_40x5 = res.MaBLAS_40x5,
        # Gaius = res.lvBLAS,
        PaddedMatrices = res.PaddedMatrices,
        GenericMatMul = res.OpenBLAS,
    );
    dfs = stack(df, [:MaBLAS_16x12, :MaBLAS_40x5, :PaddedMatrices, :GenericMatMul], variable_name = :Library, value_name = :Time);
#    dfs = stack(df, [:Gaius, :PaddedMatrices, :GenericMatMul], variable_name = :Library, value_name = :Time);
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
       x = {:Size, scale={type=:log}}, y = {:GFLOPS},#, scale={type=:log}},
        # x = {:Size}, y = {:GFLOPS},#, scale={type=:log}},
        width = 900, height = 600
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


using PaddedMatrices
include(joinpath(pkgdir(PaddedMatrices), "benchmark/blasbench.jl"))

const LIBBLASFEO = Libdl.dlopen("/home/chriselrod/Documents/libraries/blasfeo/lib/libblasfeo.so");
const DGEMM_BLASFEO = Libdl.dlsym(LIBBLASFEO, :blasfeo_dgemm)
const SGEMM_BLASFEO = Libdl.dlsym(LIBBLASFEO, :blasfeo_sgemm)


for (fm,T) ∈ [(:DGEMM_BLASFEO,Float64), (:SGEMM_BLASFEO, Float32)]
    @eval begin
        function gemmfeo!(C::AbstractMatrix{$T}, A::AbstractMatrix{$T}, B::AbstractMatrix{$T})
            transA = istransposed(A)
            transB = istransposed(B)
            M, N = size(C); K = size(B, 1) % Int32
            pA = parent(A); pB = parent(B)
            ldA = stride(pA, 2) % Int32
            ldB = stride(pB, 2) % Int32
            ldC = stride(C, 2) % Int32
            α = one($T)
            β = zero($T)
            ccall(
                $fm, Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{$T}, Ref{$T},
                 Ref{Int32}, Ref{$T}, Ref{Int32}, Ref{$T}, Ref{$T}, Ref{Int32}),
                transA, transB, (M % Int32), (N % Int32), K, α, pA, ldA, pB, ldB, β, C, ldC
            )
        end
    end
end


function runbenchfeo(::Type{T}, sizes = 2:300) where {T}
    (StructVector ∘ map)(sizes) do sz
        n, k, m = sz, sz, sz
        C1 = Matrix{T}(undef, n, m)
        C2 = similar(C1);
        C3 = similar(C1);
        C4 = similar(C1);
        A  = randa(T, n, k)
        B  = randa(T, k, m)

        jmlt = benchmark_fun!(PaddedMatrices.jmul!, C1, A, B, sz == first(sizes))
        opbt = benchmark_fun!(gemmopenblas!, C2, A, B, sz == first(sizes), C1)
        mklbt= benchmark_fun!(gemmmkl!, C3, A, B, sz == first(sizes), C1)
        bfeot= benchmark_fun!(gemmfeo!, C4, A, B, sz == first(sizes), C1)
        @show (matrix_size=sz, OpenBLAS=opbt, MKL=mklbt, PaddedMatrices=jmlt, BLASFEO=bfeot)
    end
end




using PaddedMatrices
include(joinpath(pkgdir(PaddedMatrices), "benchmark/blasbench.jl"))

const LIBBLASFEO = Libdl.dlopen("/home/chriselrod/Documents/libraries/blasfeo/lib/libblasfeo.so");
const DGEMM_BLASFEO = Libdl.dlsym(LIBBLASFEO, :blasfeo_dgemm)
const SGEMM_BLASFEO = Libdl.dlsym(LIBBLASFEO, :blasfeo_sgemm)

@inline nottransposed(A) = A
@inline nottransposed(A::Adjoint) = parent(A)
@inline nottransposed(A::Transpose) = parent(A)

for (fm,T) ∈ [(:DGEMM_BLASFEO,Float64), (:SGEMM_BLASFEO, Float32)]
    @eval begin
        function gemmfeo!(C::AbstractMatrix{$T}, A::AbstractMatrix{$T}, B::AbstractMatrix{$T})
            transA = istransposed(A)
            transB = istransposed(B)
            M, N = size(C); K = size(B, 1) % Int32
            pA = nottransposed(A); pB = nottransposed(B)
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
        res =  (matrix_size=sz, OpenBLAS=opbt, MKL=mklbt, PaddedMatrices=jmlt, BLASFEO=bfeot)
        @show res
    end
end

istransposed(::PaddedMatrices.AbstractStrideMatrix{<:Any,<:Any,<:Any,1}) = 'N'
istransposed(::PaddedMatrices.AbstractStrideMatrix{<:Any,<:Any,<:Any,1,1}) = 'N'
istransposed(::PaddedMatrices.AbstractStrideMatrix{<:Any,<:Any,<:Any,<:Any,1}) = 'T'
nottransposed(A::PaddedMatrices.AbstractStrideMatrix{<:Any,<:Any,<:Any,1}) = A
nottransposed(A::PaddedMatrices.AbstractStrideMatrix{<:Any,<:Any,<:Any,1,1}) = A
nottransposed(A::PaddedMatrices.AbstractStrideMatrix{<:Any,<:Any,<:Any,<:Any,1}) = A'
using Random
function runstaticbenchfeo(::Type{T}, sizes, f1 = identity, f2 = identity) where {T}
    (StructVector ∘ map)(sizes) do sz
        Apadded = f1(rand!(FixedSizeMatrix{sz,sz,T}(undef)))
        Bpadded = f2(rand!(FixedSizeMatrix{sz,sz,T}(undef)))
        Cpadded1 = FixedSizeMatrix{sz,sz,T}(undef)
        Cpadded2 = FixedSizeMatrix{sz,sz,T}(undef)
        A = f1(Array(nottransposed(Apadded)));
        B = f2(Array(nottransposed(Bpadded)));
        C = similar(A, size(A,1), size(B,2));
        pmt = benchmark_fun!(mul!, Cpadded1, Apadded, Bpadded, sz == first(sizes))
        fmt = benchmark_fun!(gemmfeo!, Cpadded2, Apadded, Bpadded, sz == first(sizes), Cpadded1)
        jmt = benchmark_fun!(PaddedMatrices.jmul!, C, A, B, sz == first(sizes), Cpadded1)
        res = (matrix_size=sz, SizedArray = pmt, BLASFEO=fmt, PaddedMatrices=jmt)
        @show res
    end
end



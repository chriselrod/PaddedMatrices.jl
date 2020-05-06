


using MaBLAS, BenchmarkTools

function matmul_params(::Type{T}, mᵣ, nᵣ, mcratio, kcratio, ncratio) where {T}
    L₁, L₂, L₃ = MaBLAS.VectorizationBase.CACHE_SIZE .÷ sizeof(T)
    kc = round(Int, (kcratio * L₁ / nᵣ))
    mc = round(Int, (mcratio * L₂ / (kc*mᵣ))) * mᵣ
    nc = round(Int, (ncratio * L₃ / (kc*nᵣ))) * nᵣ
    mc, kc, nc
end



function search()
    kernels = [
        (16,12),
        (16,14),
        (24,9),
        (32,6),
        (40,5)
    ]
    Kcratio = 0.4:0.125:0.9
    Mcratio = 0.4:0.125:0.9
    Ncratio = 0.4:0.125:0.9
    matmul_sizes = (6:20) .^ 3
    GFLOPS = fill(-Inf, length(Mcratio), length(Kcratio), length(Ncratio), length(kernels), length(matmul_sizes))
    params = Array{Tuple{Tuple{Int,Int},NTuple{3,Int}}}(undef, length(Mcratio), length(Kcratio), length(Ncratio), length(kernels))
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = timelim = 1e-2

    for (szi,sz) ∈ enumerate(matmul_sizes)
        C = Matrix{Float64}(undef, sz, sz);
        A = rand(sz, sz); B = rand(sz, sz);
        for (rri,(mr,nr)) ∈ enumerate(kernels), (nci,ncf) ∈ enumerate(Ncratio), (kci,kcf) ∈ enumerate(Kcratio), (mci,mcf) ∈ enumerate(Mcratio)
            mc, kc, nc = matmul_params(Float64, mr, nr, mcf, kcf, ncf)
            
            t1 = @elapsed MaBLAS.mul!(C, A, B; packing=true, cache_params=(cache_m=mc, cache_k=kc, cache_n=nc), kernel_params=(Val(mr), Val(nr)))
            t0 = if isone(szi) || t1 < 1e-3
                t2 = @belapsed MaBLAS.mul!($C, $A, $B; packing=true, cache_params=(cache_m=$mc, cache_k=$kc, cache_n=$nc), kernel_params=(Val($mr), Val($nr)))
                min(t1,t2)
            elseif 5t1 < timelim
                t2 = @elapsed MaBLAS.mul!(C, A, B; packing=true, cache_params=(cache_m=mc, cache_k=kc, cache_n=nc), kernel_params=(Val(mr), Val(nr)))
                min(t1,t2)
            else
                t1
            end
            GFLOPS[mci,kci,nci,rri,szi] = gflops = 2e-9*sz^3/t0
            isone(szi) && (params[mci,kci,nci,rri] = ((mr,nr), (mc,kc,nc)))
            @show sz, (mr,nr), (mc,kc,nc), gflops
        end
    end
    GFLOPS, params
end


#gflops, params = search()




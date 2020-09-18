
function maximum(::typeof(abs), A::AbstractStrideArray{S,T}) where {S,T}
    s = typemin(T)
    @avx for i ∈ eachindex(A)
        s = max(s, abs(A[i]))
    end
    s
end

function Base.vcat(A::AbstractStrideMatrix, B::AbstractStrideMatrix)
    sA1 = maybestaticsize(A,Val{1}())
    out = allocarray(promote_type(eltype(A), eltype(B)), (PaddedMatrices.vadd(sA1, maybestaticsize(B,Val{1}())) , maybestaticsize(A,Val{2}())))
    # GC.@preserve out A B begin
        # outP = PtrArray(outP); AP = PtrArray(A); BP = PtrArray(B);
    @avx for j ∈ axes(A,2), i ∈ axes(A,1)
        out[i,j] = A[i,j]
    end
    @avx for j ∈ axes(B,2), i ∈ axes(B,1)
        out[i + sA1,j] = B[i,j]
    end
    out
end



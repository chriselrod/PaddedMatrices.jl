
function maximum(::typeof(abs), A::AbstractStrideArray{S,T}) where {S,T}
    s = typemin(T)
    @avx for i ∈ eachindex(A)
        s = max(s, abs(A[i]))
    end
    s
end


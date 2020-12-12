

@inline Base.map!(f::F, A::AbstractStrideArray, args::Vararg{Any,K}) where {F,K} = vmap!(f, A, args...)
@inline Base.map(f::F, A::AbstractStrideArray, args::Vararg{Any,K}) where {F,K} = vmap(f, A, args...)
@inline Base.reduce(op::O, A::AbstractStrideArray) where {O} = vreduce(op, A)
@inline Base.mapreduce(f::F, op::O, A::AbstractStrideArray, args::Vararg{Any,K}) where {F, O, K} = vmapreduce(f, op, A, args...)
@inline Base.mapreduce(f::F, op::O, A::AbstractStrideArray) where {F, O} = vmapreduce(f, op, A)

for op ∈ (:+, :max, :min)
    @eval @inline Base.reduce(::typeof($op), A::AbstractStrideArray; dims = nothing) = vreduce($op, A, dims = dims)
end

function maximum(::typeof(abs), A::AbstractStrideArray{S,T}) where {S,T}
    s = typemin(T)
    @avx for i ∈ eachindex(A)
        s = max(s, abs(A[i]))
    end
    s
end

function Base.vcat(A::AbstractStrideMatrix, B::AbstractStrideMatrix)
    MA, NA = size(A)
    MB, NB = size(B)
    @assert NA == NB
    TC = promote_type(eltype(A), eltype(B))
    C = StrideArray{TC}(undef, (MA + MB, NA))
    # TODO: Actually handle offsets
    @assert offsets(A) === offsets(B)
    @assert offsets(A) === offsets(C)
    @avx for j ∈ axes(A,2), i ∈ axes(A,1)
        C[i,j] = A[i,j]
    end
    @avx for j ∈ axes(B,2), i ∈ axes(B,1)
        C[i + MA,j] = B[i,j]
    end
    C
end



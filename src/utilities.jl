

function determine_vectorloads(num_rows, ::Type{T} = Float64) where T
    W = VectorizationBase.pick_vector_width(num_rows, T)
    num_vectors = cld(num_rows, W)
    num_vectors, num_vectors * W
end

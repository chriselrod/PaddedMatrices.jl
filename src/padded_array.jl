struct PaddedArray{T,N} <: AbstractPaddedArray{T,N}
    data::Array{T,N}
    nvector_loads::Int
    size::NTuple{N,Int}
end
const PaddedVector{T} = PaddedArray{T,1}
const PaddedMatrix{T} = PaddedArray{T,2}
@generated function PaddedArray{T}(::UndefInitializer, S::NTuple{N}) where {T,N}
    quote
        nrow = S[1]
        W, Wshift = VectorizationBase.pick_vector_width_shift(T)
        TwoN = 2nrow
        if W < TwoN
            rem = (nrow & (W-1))
            padded_rows = rem == 0 ? nrow : nrow + W - rem

            nvector_loads = padded_rows >> Wshift

            data = Array{$T,$N}(undef,
                $(Expr(:tuple, padded_rows, [:(S[$n]) for n ∈ 2:N]...))
            )
            @nloops $(N-1) i j -> 1:S[j+1] begin
                @inbounds for i_0 ∈ padded_rows+1-W:padded_rows
                    ( @nref $N data n -> i_{n-1} ) = $(zero(T))
                end
            end
            return PaddedArray{$T,$N}(data, nvector_loads, S)
        end
        while W >= TwoN
            W >>= 1
        end
        rem = (nrow & (W-1))
        padded_rows = rem == 0 ? nrow : nrow + W - rem

        data = zeros($T, $(Expr(:tuple, padded_rows, [:(S[$n]) for n ∈ 2:N]...)) )
        PaddedArray{$T,$N}(data, 1, S)
    end
end
function Base.zeros(::Type{PaddedArray{T}}, S::NTuple{N}) where {T,N}
    W, Wshift = VectorizationBase.pick_vector_width_shift(S[1], T)
    rem = (nrow & (W-1))
    padded_rows = rem == 0 ? nrow : nrow + W - rem
    nvector_loads = padded_rows >> Wshift
    PaddedArray{T,N}(
        zeros(T, Base.setindex(S, padded_rows, 1)), nvector_loads, S
    )
end
function Base.ones(::Type{PaddedArray{T}}, S::NTuple{N}) where {T,N}
    W, Wshift = VectorizationBase.pick_vector_width_shift(S[1], T)
    rem = (nrow & (W-1))
    padded_rows = rem == 0 ? nrow : nrow + W - rem
    nvector_loads = padded_rows >> Wshift
    PaddedArray{T,N}(
        ones(T, Base.setindex(S, padded_rows, 1)), nvector_loads, S
    )
end
function Base.fill(::Type{<: PaddedArray}, S::NTuple{N}, v::T) where {T,N}
    W, Wshift = VectorizationBase.pick_vector_width_shift(S[1], T)
    rem = (nrow & (W-1))
    padded_rows = rem == 0 ? nrow : nrow + W - rem
    nvector_loads = padded_rows >> Wshift
    PaddedArray{T,N}(
        fill(v, Base.setindex(S, padded_rows, 1)), nvector_loads, S
    )
end

@inline Base.pointer(A::PaddedArray) = pointer(A.data)
@inline VectorizationBase.vectorizable(A::PaddedArray) = pointer(A.data)

Base.strides(A::PaddedArray) = strides(A.data)

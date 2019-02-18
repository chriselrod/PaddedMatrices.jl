function init_mpadded_array_quote(S, T)
    SV = S.parameters
    N = length(SV)

    nrow = SV[1]
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    TwoN = 2nrow
    if W < TwoN
        rem = (nrow & (W-1))
        padded_rows = rem == 0 ? nrow : nrow + W - rem
        L = padded_rows
        for n ∈ 2:N
            L *= SV[n]
        end
        return quote
            out = MStaticPaddedArray{$S,$T,$N,$padded_rows,$L}(undef)
            @nloops $(N-1) i j -> 1:S[j+1] begin
                @inbounds for i_0 ∈ $(padded_rows+1-W):$padded_rows
                    ( @nref $N out n -> i_{n-1} ) = zero($T)
                end
            end
            out
        end
    end
    while W >= TwoN
        W >>= 1
    end
    rem = (nrow & (W-1))
    padded_rows = rem == 0 ? nrow : nrow + W - rem
    L = padded_rows
    for n ∈ 2:N
        L *= SV[n]
    end
    quote
        out = MStaticPaddedArray{$S,$T,$N,$padded_rows,$L}(undef)
        for l ∈ 1:$L
            out[l] = zero($T)
        end
        out
    end
end


"""
Parameters are
S, the size tuple
T, the element type
N, the number of axis
R, the actual number of rows, where the excess R > S[1] are zero.
L, number of elements (including buffer zeros).
"""
mutable struct MStaticPaddedArray{S,T,N,P,L} <: AbstractStaticPaddedArray{S,T,N,P,L}
    data::NTuple{L,T}
    function MStaticPaddedArray{S,T,N,R,L}(::UndefInitializer) where {S,T,N,R,L}
        new()
    end
    @generated function MStaticPaddedArray{S,T}(::UndefInitializer) where {S,T}
        init_mpadded_array_quote(S, T)
    end
    @generated function MStaticPaddedArray{S,T,N}(::UndefInitializer) where {S,T,N}
        init_mpadded_array_quote(S, T)
    end
end
const MStaticPaddedVector{S,T,P,L} = MStaticPaddedArray{Tuple{S},T,1,P,L}
const MStaticPaddedMatrix{M,N,T,P,L} = MStaticPaddedArray{Tuple{M,N},T,2,P,L}

@generated function MStaticPaddedArray(::UndefInitializer, ::Val{S}, ::Type{T1}=Float64) where {S,T1}
    SD = Tuple{S...}
    init_mpadded_array_quote(SD, T)
end

# function zero_hidden!(A::SizedSIMDArray{SD}) where SD
#
# end

# Not type stable!
# function MStaticPaddedArray(A::AbstractArray{T,N}) where {T,N}
#     out = MStaticPaddedArray{Tuple{size(A)...},T,N}(undef)
#     copyto!(out, A)
#     out
# end

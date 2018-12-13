struct PaddedArray{T,N} <: AbstractPaddedArray{T,N}
    data::Array{T,N}
    nvector_loads::Int
    size::NTuple{N,Int}
    @generated function PaddedArray{T}(::UndefInitializer, S::NTuple{N}) where {T,N}
        quote
            nrow = S[1]
            nvector_loads, padded_rows = determine_vectorloads(nrow, T)

            data = Array{$T,$N}(undef,
                $(Expr(:tuple, padded_rows, [:(S[$n]) for n ∈ 2:N]...))
            )
            # We need to zero out the excess, so it doesn't interfere
            # with operations like summing the columns or matrix mul.
            # add @generated and quote the expression if you add this back.
            @nloops $(N-1) i j -> 1:S[j+1] begin
                for i_0 = nrow+1:padded_rows
                    ( @nref $N out n -> i_{n-1} ) = $(zero(T))
                end
            end
            new{$T,$N}(data, nvector_loads, S)
        end
    end
end

@inline Base.pointer(A::PaddedArray) = pointer(A.data)
@inline VectorizationBase.vectorizable(A::PaddedArray) = pointer(A.data)

Base.strides(A::PaddedArray) = strides(A.data)

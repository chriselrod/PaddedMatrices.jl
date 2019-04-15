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
    nrow = S[1]
    W, Wshift = VectorizationBase.pick_vector_width_shift(nrow, T)
    Wm1 = W - 1
    rem = nrow & Wm1
    padded_rows = (nrow + Wm1) & ~Wm1
    nvector_loads = padded_rows >> Wshift
    PaddedArray{T,N}(
        ones(T, Base.setindex(S, padded_rows, 1)), nvector_loads, S
    )
end
function Base.fill(::Type{<: PaddedArray}, S::NTuple{N}, v::T) where {T,N}
    nrow = S[1]
    W, Wshift = VectorizationBase.pick_vector_width_shift(nrow, T)
    Wm1 = W - 1
    rem = nrow & Wm1
    padded_rows = (nrow + Wm1) & ~Wm1
    nvector_loads = padded_rows >> Wshift
    PaddedArray{T,N}(
        fill(v, Base.setindex(S, padded_rows, 1)), nvector_loads, S
    )
end

Base.size(A::PaddedArray) = A.size
@inline Base.getindex(A::PaddedArray, I...) = Base.getindex(A.data, I...)
@inline Base.setindex!(A::PaddedArray, v, I...) = Base.setindex!(A.data, v, I...)

@inline Base.pointer(A::PaddedArray) = pointer(A.data)
@inline VectorizationBase.vectorizable(A::PaddedArray) = VectorizationBase.vpointer(pointer(A.data))

@inline Base.strides(A::PaddedArray) = strides(A.data)
@inline Base.stride(A::PaddedArray, n::Integer) = stride(A.data, n)


@generated function muladd!(D::PaddedMatrix{T}, a::PaddedVector{T}, B::PaddedMatrix{T}, c::PaddedVector{T}) where {T}
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    register_count = VectorizationBase.REGISTER_COUNT
    # repetitions = register_count ÷ 3
    repetitions = 4
    # rep_half = repetitions >> 1
    quote
        N = size(D,2)
        @boundscheck begin
            # Nd = N
            Pd = size(D,1)
            Nd = N
            Pb, Nb = size(B)
            pa = length(a)
            pc = length(c)
            Pd == Pb == pa == pc || ThrowBoundsError("Rows: Pd == Pb == pa == pc => $Pd == $Pb == $pa == $pc => false")
            Nd == Nb || ThrowBoundsError("Columns: Nd == Nb => $Nd == $Nb => false")
        end
        s = stride(D,2)
        regrem = s & $(W-1)
        nregrep = s >> $Wshift
        nregrepgroup, nregrepindiv = divrem(nregrep, $repetitions)
        vD = VectorizationBase.vectorizable(D)
        va = VectorizationBase.vectorizable(a)
        vB = VectorizationBase.vectorizable(B)
        vc = VectorizationBase.vectorizable(c)
        for n ∈ 0:nregrepgroup-1
            Base.Cartesian.@nexprs $repetitions r -> begin
                a_r = SIMDPirates.vload(Vec{$W,$T}, va + $W*(r-1) + n*$(repetitions*W) )
                c_r = SIMDPirates.vload(Vec{$W,$T}, vc + $W*(r-1) + n*$(repetitions*W) )
            end
            for col ∈ 0:N-1
                Base.Cartesian.@nexprs $repetitions r -> begin
                    SIMDPirates.vstore!(
                        vD + $W*(r-1) + n*$(repetitions*W) + s*col,
                        SIMDPirates.vmuladd(
                            SIMDPirates.vload(Vec{$W,$T}, vB + $W*(r-1) + n*$(repetitions*W) + s*col),
                            a_r, c_r
                        )
                    )
                end
            end
        end
        rem_base = nregrepgroup*$(repetitions*W)
        # if nregrepindiv >= $rep_half
        #     Base.Cartesian.@nexprs $rep_half r -> begin
        #         a_r = SIMDPirates.vload(Vec{$W,$T}, va + $W*(r-1) + rem_base )
        #         c_r = SIMDPirates.vload(Vec{$W,$T}, vc + $W*(r-1) + rem_base )
        #     end
        #     for col ∈ 0:N-1
        #         Base.Cartesian.@nexprs $rep_half r -> begin
        #             SIMDPirates.vstore!(
        #                 vD + $W*(r-1) + rem_base + s*col,
        #                 SIMDPirates.vmuladd(
        #                     SIMDPirates.vload(Vec{$W,$T}, vB + $W*(r-1) + rem_base + s*col),
        #                     a_r, c_r
        #                 )
        #             )
        #         end
        #     end
        #     rem_base += $(rep_half*W)
        #     nregrepindiv -= $rep_half
        #     # @show rem_base, nregrepindiv
        # end
        # @show rem_base, nregrepindiv
        for n ∈ 0:nregrepindiv-1
            a_r = SIMDPirates.vload(Vec{$W,$T}, va + $W*n + rem_base )
            c_r = SIMDPirates.vload(Vec{$W,$T}, vc + $W*n + rem_base )
            for col ∈ 0:N-1
                SIMDPirates.vstore!(
                    vD + $W*n + rem_base + s*col,
                    SIMDPirates.vmuladd(
                        SIMDPirates.vload(Vec{$W,$T}, vB+ $W*n + rem_base + s*col),
                        a_r, c_r
                    )
                )
            end
        end
        if regrem > 0
            rowoffset = s & $(~(W-1))
            for col ∈ 1:N, j ∈ rowoffset+1:s
                D[j,col] = muladd(a[j], B[j,col], c[j])
            end
        end
    end
end


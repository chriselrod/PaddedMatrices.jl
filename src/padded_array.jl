 abstract type AbstractDynamicPaddedArray{T,N} <: AbstractPaddedArray{T,N} end
struct PaddedArray{T,N} <: AbstractDynamicPaddedArray{T,N}
    data::Array{T,N}
#    nvector_loads::Int
#    stride::Int
    size::NTuple{N,Int}
end
struct DynamicPtrArray{T,N} <: AbstractDynamicPaddedArray{T,N}
    ptr::Ptr{T}
    stride::Int
    size::NTuple{N,Int}
end
const DynamicPaddedVector{T} = DynamicPaddedArray{T,1}
const DynamicPaddedMatrix{T} = DynamicPaddedArray{T,2}
const DynamicPtrVector{T} = DynamicPtrArray{T,1}
const DynamicPtrMatrix{T} = DynamicPtrArray{T,2}

#=
@generated function DynamicPaddedArray{T}(::UndefInitializer, S::NTuple{N}) where {T,N}#, ::Val{Z} = Val(true)) where {T,N,Z}
    quote
        nrow = S[1]
        W, Wshift = VectorizationBase.pick_vector_width_shift(T)
        TwoN = 2nrow
#        if W < TwoN
            Wm1 = W - 1
            rem = nrow & Wm1
            padded_rows = (nrow + Wm1) & ~Wm1

            nvector_loads = padded_rows >> Wshift

            data = Array{$T,$N}(undef,
                $(Expr(:tuple, :padded_rows, [:(S[$n]) for n ∈ 2:N]...))
            )
#            if $Z
#                @nloops $(N-1) i j -> 1:S[j+1] begin
#                    @inbounds for i_0 ∈ padded_rows+1-W:padded_rows
#                        ( @nref $N data n -> i_{n-1} ) = $(zero(T))
#                    end
#                end
#            end
            return DynamicPaddedArray{$T,$N}(data, nvector_loads, S)
#        end
#        while W >= TwoN
#            W >>= 1
#        end
#        Wm1 = W - 1
#        rem = nrow & Wm1
#        padded_rows = (nrow + Wm1) & ~Wm1
#
#        data = zeros($T, $(Expr(:tuple, :padded_rows, [:(S[$n]) for n ∈ 2:N]...)) )
#        DynamicPaddedArray{$T,$N}(data, 1, S)
    end
end
=#
function DynamicPaddedArray{T}(::UndefInitializer, S::NTuple{N}) where {T,N}#, ::Val{Z} = Val(true)) where {T,N,Z}
    nrow = S[1]
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    TwoN = 2nrow
    Wm1 = W - 1
    rem = nrow & Wm1
    padded_rows = (nrow + Wm1) & ~Wm1
#    nvector_loads = padded_rows >> Wshift
    padded_size = Base.setindex(S, padded_rows, 1)
    data = Array{T,N}(undef, padded_size)
    return DynamicPaddedArray{T,N}(data, S)
end
function DynamicPaddedArray{T}(::UndefInitializer, S::NTuple{N,Int}, padded_rows::Int) where {T,N}#, ::Val{Z} = Val(true)) where {T,N,Z}
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
#    nvector_loads = padded_rows >> Wshift
    padded_size = Base.setindex(S, padded_rows, 1)
    data = Array{T,N}(undef, padded_size)
    return DynamicPaddedArray{T,N}(data, S)
end

function DynamicPaddedArray(A::AbstractArray{T,N}) where {T,N}
    pA = DynamicPaddedArray{T}(undef, size(A))
    @inbounds for i ∈ CartesianIndices(A)
        pA[i] = A[i]
    end
    pA
end
# function DynamicPaddedArray{T where T,N}(A::AbstractArray{T,N}) where {T,N}
#     pA = DynamicPaddedArray{T}(undef, size(A))
#     @inbounds for i ∈ CartesianIndices(A)
#         pA[i] = A[i]
#     end
#     pA
# end
function Base.zeros(::Type{DynamicPaddedArray{T}}, S::NTuple{N}) where {T,N}
    nrow = S[1]
    W, Wshift = VectorizationBase.pick_vector_width_shift(S[1], T)
    rem = (nrow & (W-1))
    padded_rows = rem == 0 ? nrow : nrow + W - rem
 #   nvector_loads = padded_rows >> Wshift
    DynamicPaddedArray{T,N}(
        zeros(T, Base.setindex(S, padded_rows, 1)), S#nvector_loads, S
    )
end
function Base.ones(::Type{DynamicPaddedArray{T}}, S::NTuple{N}) where {T,N}
    nrow = S[1]
    W, Wshift = VectorizationBase.pick_vector_width_shift(nrow, T)
    Wm1 = W - 1
    rem = nrow & Wm1
    padded_rows = (nrow + Wm1) & ~Wm1
#    nvector_loads = padded_rows >> Wshift
    DynamicPaddedArray{T,N}(
        ones(T, Base.setindex(S, padded_rows, 1)), S#nvector_loads, S
    )
end
function Base.fill(::Type{<: DynamicPaddedArray}, S::NTuple{N}, v::T) where {T,N}
    nrow = S[1]
    W, Wshift = VectorizationBase.pick_vector_width_shift(nrow, T)
    Wm1 = W - 1
    rem = nrow & Wm1
    padded_rows = (nrow + Wm1) & ~Wm1
#    nvector_loads = padded_rows >> Wshift
    DynamicPaddedArray{T,N}(
        fill(v, Base.setindex(S, padded_rows, 1)), S#nvector_loads, S
    )
end

Base.size(A::AbstractDynamicPaddedArray) = A.size
@inline Base.getindex(A::DynamicPaddedArray, I...) = Base.getindex(A.data, I...)
@inline Base.setindex!(A::DynamicPaddedArray, v, I...) = Base.setindex!(A.data, v, I...)

@inline function Base.getindex(A::DynamicPtrArray{T,N}, i::Integer) where {T,N}
    VectorizationBase.load(A.ptr + (i-1) * sizeof(T))
end
@inline function Base.setindex!(A::DynamicPtrArray{T,N}, v::T, i::Integer) where {T,N}
    VectorizationBase.store!(A.ptr + (i-1) * sizeof(T), v)
end
@inline function Base.getindex(A::DynamicPtrVector{T}, i::Integer) where {T}
    VectorizationBase.load(A.ptr + (i-1) * sizeof(T))
end
@inline function Base.setindex!(A::DynamicPtrVector{T}, v::T, i::Integer) where {T}
    VectorizationBase.store!(A.ptr + (i-1) * sizeof(T), v)
end
@inline function Base.getindex(A::DynamicPtrArray{T,N}, I::Vararg{Integer,N}) where {T,N}
    i = sub2ind(A.size, I, A.stride)
    VectorizationBase.load(A.ptr + (i-1) * sizeof(T))
end
@inline function Base.setindex!(A::DynamicPtrArray{T,N}, v::T, I::Vararg{Integer,N}) where {T,N}
    i = sub2ind(A.size, I, A.stride)
    VectorizationBase.store!(A.ptr + (i-1) * sizeof(T), v)
end
@inline function Base.getindex(A::DynamicPtrArray{T,N}, I::Union{CartesianIndex{N},NTuple{N,Int}}) where {T,N}
    i = sub2ind(A.size, I, A.stride)
    VectorizationBase.load(A.ptr + (i-1) * sizeof(T))
end
@inline function Base.setindex!(A::DynamicPtrArray{T,N}, v::T, I::Union{CartesianIndex{N},NTuple{N,Int}}) where {T,N}
    i = sub2ind(A.size, I, A.stride)
    VectorizationBase.store!(A.ptr + (i-1) * sizeof(T), v)
end

@inline Base.pointer(A::DynamicPaddedArray) = pointer(A.data)
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::DynamicPaddedArray{T}) where {T} = pointer(A)
@inline Base.pointer(A::DynamicPtrArray) = A.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::DynamicPtrArray{T}) where {T} = A.ptr

@inline VectorizationBase.vectorizable(A::AbstractDynamicPaddedArray) = VectorizationBase.vpointer(pointer(A.data))



@inline Base.strides(A::DynamicPaddedArray) = strides(A.data)
@inline Base.stride(A::DynamicPaddedArray, n::Integer) = stride(A.data, n)
@generated function Base.strides(A::DynamicPtrArray{T,N}) where {T,N}
    N == 1 && return (1,)
    s = Expr(:tuple,1,:(A.stride))
    N == 2 && return s
    for n ∈ 2:$N-1
        push!(s.args, :(sizaA[$n]))
    end
    :(sizeA = A.size; @inbounds $s)
end
@inline function Base.stride(A::DynamicPtrArray, n::Integer)
    n == 1 && return 1
    n == 2 && return A.stride
    @boundscheck n > N && ThrowBoundsError("$n > $N")
    @inbounds A.size[n-1]
end


@inline function Base.copy(A::DynamicPaddedArray{T,N}) where {T,N}
    DynamicPaddedArray(
        copy(A.data), A.size
    )
end
@inline function Base.copy(A::DynamicPtrArray{T,N}) where {T,N}
    B = Array{T,N}(undef, Base.setindex(A.size, A.stride, 1))
    @inbounds @simd ivdep for i ∈ eachindex(B)
        B[i] = A[i]
    end
    DynamicPaddedArray(
        B, A.size
    )
end

@generated function muladd!(
    D::AbstractDynamicPaddedMatrix{T},
    a::AbstractDynamicPaddedVector{T},
    B::AbstractDynamicPaddedMatrix{T},
    c::AbstractDynamicPaddedVector{T}
) where {T}
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


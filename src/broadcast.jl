# First, we add templates for matrix multiplication
abstract type AbstractProdct{T,N} end
abstract type AbstractSizedProduct{M,P,K,T,N} <: AbstractProdct{T,N} end

function reduce_size(@nospecialize(S), ::Val{N}) where {N}
    s = fill!(MutableFixedSizePaddedVector{N,Int}(undef),1)
    SV = S.parameters
    j = SV[1]::Union{Int,DataType}
    if j isa Int
        for i ∈ eachindex(SV)
            s[i] = (SV[i])::Int
        end
    elseif j isa DataType
        for i ∈ eachindex(SV)
            s2 = reduce_size(SV[i]::DataType, Val(N))
            for n ∈ 1:N
                sₙ = s[n]
                s2ₙ = s2[n]
                if sₙ == 1
                    s[n] = s2[n]
                elseif (s2ₙ == sₙ) || s2ₙ == 1 # neither are 1
                    nothing
                else
                    throw("sₙ == $sₙ != s2ₙ == $s2ₙ")
                end
            end
        end
    end
    ConstantFixedSizePaddedVector(s)
end

function reduce_size(@nospecialize S)
    reduce_size!(Int[1], S)
end
function pick_size(s1::Int, s2::Int)
    if s1 == 1
        s3 = s2
    elseif (s1 == s2) || (s2 == 1)
        s3 = s1
    else
        throw("s1 == $s1 != s2 == $s2; they should be equal, or == 1")
    end
    s3
end

function reduce_size!(s::Vector{Int}, @nospecialize(S))
    SV = S.parameters
    length(SV) == 0 && return s
    j = SV[1]::Union{Int,DataType}
    if j isa Int
        N1 = length(SV)
        N2 = length(s)
        for n ∈ 1:min(N1,N2)
            s[n] = pick_size(s[n], SV[n])
        end
        for n ∈ min(N1,N2)+1:N1
            push!(s, SV[n])
        end
    elseif j isa DataType
        for i ∈ eachindex(SV)
            s2 = reduce_size!(s, (SV[i])::DataType)
        end
    end
    s
end

#reduce_size(S,Val(2))


@inline Base.ndims(::Type{<:AbstractProdct{T,N}}) where {T,N} = N
@inline Base.broadcastable(A::AbstractProdct) = A
function Base.show(io::IO, p::AbstractProdct)
    # We don't define a getindex method.
    println("Product of A, of size $(size(p.A)):\n", p.A, "\nand B, of size $(size(p.B)):\n", p.B)
end

struct SizedMatrixMatrixProduct{T,M,P,K,TA <: AbstractFixedSizePaddedMatrix{M,P,T}, TB <: AbstractFixedSizePaddedMatrix{P,K,T}} <: AbstractSizedProduct{M,P,K,T,2}
    A::TA
    B::TB
end
@inline function LinearAlgebra.:×(A::TA, B::TB) where {M,P,K,T,TA <: AbstractFixedSizePaddedMatrix{M,P,T}, TB <: AbstractFixedSizePaddedMatrix{P,K,T}}
    SizedMatrixMatrixProduct{T,M,P,K,TA,TB}(A,B)
end

@inline Base.size(::SizedMatrixMatrixProduct{T,M,P,K}) where {T,M,P,K} = (M,K)
@inline Base.axes(::SizedMatrixMatrixProduct{T,M,P,K}) where {T,M,P,K} = (Base.OneTo(M),Base.OneTo(K))

struct SizedVectorMatrixProduct{T,P,K,TA <: AbstractFixedSizePaddedVector{P,T}, TB <: AbstractFixedSizePaddedMatrix{P,K,T}} <: AbstractSizedProduct{1,P,K,T,2}
    A::TA
    B::TB
end
function LinearAlgebra.:×(A::TA, B::TB) where {P,K,T,TA <: AbstractFixedSizePaddedVector{P,T}, TB <: AbstractFixedSizePaddedMatrix{P,K,T}}
    SizedVectorMatrixProduct{T,P,K,TA,TB}(A,B)
end
@inline Base.size(::SizedVectorMatrixProduct{T,P,K}) where {T,P,K} = (1,K)
@inline Base.axes(::SizedVectorMatrixProduct{T,P,K}) where {T,P,K} = (Base.OneTo(1),Base.OneTo(K))


struct SizedMatrixVectorProduct{T,M,P,TA <: AbstractFixedSizePaddedMatrix{M,P,T}, TB <: AbstractFixedSizePaddedVector{P,T}} <: AbstractSizedProduct{M,P,1,T,1}
    A::TA
    B::TB
end
function LinearAlgebra.:×(A::TA, B::TB) where {M,P,T,TA <: AbstractFixedSizePaddedMatrix{M,P,T}, TB <: AbstractFixedSizePaddedVector{P,T}}
    SizedMatrixVectorProduct{T,M,P,TA,TB}(A,B)
end
@inline Base.size(::SizedMatrixVectorProduct{T,M}) where {T,M} = (M,)
@inline Base.axes(::SizedMatrixVectorProduct{T,M}) where {T,M} = (Base.OneTo(M),Base.OneTo(1))


struct MatrixMatrixProduct{T,TA <: AbstractPaddedMatrix{T}, TB <: AbstractPaddedMatrix{T}} <: AbstractProdct{T,2}
    A::TA
    B::TB
end
function LinearAlgebra.:×(A::TA, B::TB) where {T,TA <: AbstractPaddedMatrix{T}, TB <: AbstractPaddedMatrix{T}}
    MatrixMatrixProduct{T,TA,TB}(A,B)
end
@inline Base.size(A::MatrixMatrixProduct) = (size(A.A,1),size(A.B,2))
@inline Base.axes(A::MatrixMatrixProduct) = (Base.OneTo(size(A.A,1)),Base.OneTo(size(A.B,2)))

struct VectorMatrixProduct{T,TA <: AbstractPaddedVector{T}, TB <: AbstractPaddedMatrix{T}} <: AbstractProdct{T,2}
    A::TA
    B::TB
end
function LinearAlgebra.:×(A::TA, B::TB) where {T,TA <: AbstractPaddedVector{T}, TB <: AbstractPaddedMatrix{T}}
    VectorMatrixProduct{T,TA,TB}(A,B)
end
@inline Base.size(A::VectorMatrixProduct) = (1,size(A.B,2))
@inline Base.axes(A::VectorMatrixProduct) = (Base.OneTo(1),Base.OneTo(size(A.B,2)))

struct MatrixVectorProduct{T,TA <: AbstractPaddedMatrix{T}, TB <: AbstractPaddedVector{T}} <: AbstractProdct{T,1}
    A::TA
    B::TB
end
function LinearAlgebra.:×(A::TA, B::TB) where {T,TA <: AbstractPaddedMatrix{T}, TB <: AbstractPaddedVector{T}}
    MatrixVectorProduct{T,TA,TB}(A,B)
end
@inline Base.size(A::MatrixVectorProduct) = (size(A.A,1),)
@inline Base.axes(A::MatrixVectorProduct) = (Base.OneTo(size(A.A,1)),)

@generated function load_kernel(
                MMP::Union{<:SizedMatrixMatrixProduct{T},<:MatrixMatrixProduct{T}},
                i, j, ::Val{MP}, ::Val{mp}
            ) where {T,MP,mp}
    M, P = MP
    mk, pk = mp
    out = MutableFixedSizePaddedMatrix{mk,pk,T}(undef)
    # Should write new, simpler kernel function
    # that can take N as a symbol, and accepts arbitrary strides.
    quote

    end
end

@inline function Base.convert(::Type{A}, MMP::AbstractProdct) where {A <: AbstractArray}
    MMP.A * MMP.B
end
@inline function Base.copyto!(
                C::Union{<: AbstractDynamicPaddedArray{T},<:AbstractMutableFixedSizePaddedArray{T}},
                MMP::AbstractProdct{T}
            ) where {T}
    mul!(C, MMP.A, MMP.B)
end
@inline function Base.:+(C::AbstractPaddedArray, MMP::AbstractProdct)
    gemm(C, MMP.A, MMP.B)
end
@inline function Base.:+(MMP::AbstractProdct, C::AbstractPaddedArray)
    gemm(C, MMP.A, MMP.B)
end

struct KernelView{BC <: Base.Broadcast.Broadcasted}
    bc::BC
    m::Int
    k::Int
end
@inline function KernelView(bc::BC, m, k, ::Any, ::Any) where {BC <: Base.Broadcast.Broadcasted}
    KernelView{BC}(bc, m, k)
end
@inline function KernelView(bc::SizedMatrixMatrixProduct{T,M,P,K}, m, k, ::Val{Mk}, ::Val{Kk}) where {T,M,P,K,Mk,Kk}
    
end
@inline function Base.getindex(kv::KernelView, i, j)
    @inbounds kv.bc[kv.m + i, kv.k + j]
end

# @inline function Base.copyto!(
#             C::AbstractPaddedArray,
#             AB::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{2},Nothing,typeof(+),Tuple{Array{Float64,2},Foo{Array{Float64,2},Array{Float64,2}}}}
#         )

# end

# @enum ContainerType begin # First has highest priority
#     PaddedUnsized
#     MutableFixedSize
#     ConstantFixedSize
# end
@enum AccessPattern begin # Last has highest priority
    LinearIndexing
    BatchedColumnMajor # Batched pattern down rows, to reduce reloads of (column) vectors broadcasted with matrices
    CartesianIndexing # Standard Cartesian Indexing; if there is a row vector, we save on it's reloads.
    KernelBatches # Same pattern as matrix mul functions
end

abstract type AbstractPaddedMatrixStyle{S,T,A} <: Broadcast.BroadcastStyle end

struct FixedSizePaddedMatrixDefaultStyle{S,T,R,A} <: AbstractPaddedMatrixStyle{S,T,A} end

function pick_R(R1, R2)
    R3 = R1 == 1 ? typemax(R1) : R1
    R4 = R2 == 1 ? typemax(R2) : R2
#    R = min(R3, R4)
    if R3 != R4
        R = min(R3,R4)
        if (R3 == typemax(R1)) || (R2 == typemax(R2))
            access_pattern = LinearIndexing
        else
            access_pattern = CartesianIndexing
        end
    else
        R = R3
        access_pattern = LinearIndexing
    end
    R, access_pattern
end

Base.BroadcastStyle(::Type{<:AbstractFixedSizePaddedArray{S,T,N,R}}) where {S,T,N,R} = FixedSizePaddedMatrixDefaultStyle{S,T,R,LinearIndexing}()
Base.BroadcastStyle(::Type{LinearAlgebra.Adjoint{T,V}}) where {S,T,R,V<:AbstractFixedSizePaddedVector{S,T,R}} = FixedSizePaddedMatrixDefaultStyle{Tuple{1,S},T,1,CartesianIndexing}()
#Base.BroadcastStyle(::Type{LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedMatrix{M,N,T,R}}}) where {M,N,T,R} = FixedSizePaddedMatrixDefaultStyle{Tuple{N,M},T,R,LinearIndexing}()

@inline function Base.Broadcast.result_style(s1::FixedSizePaddedMatrixDefaultStyle, s2::FixedSizePaddedMatrixDefaultStyle)
    # s1, s2 is always the canonical order.
    Base.Broadcast.BroadcastStyle(s1, s2)
end



@generated function Base.Broadcast.combine_styles(
    s::Vararg{<:Union{AbstractFixedSizePaddedArray{S} where S,
                      <:LinearAlgebra.Adjoint{<:Any,<:AbstractFixedSizePaddedArray{S}} where S,
                      <:Base.Broadcast.Broadcasted{<:AbstractPaddedMatrixStyle},
                      BLAS.BlasFloat,Int32,Int64}, N}) where {N}

    is_padded_array = Vector{Bool}(undef, N)
 #   is_adj_padded_array = Vector{Bool}(undef, N)
    for n ∈ 1:N
        if s[n] <: AbstractFixedSizePaddedArray
            is_padded_array[n] = true
        elseif s[n] <: LinearAlgebra.Adjoint{<:Any,<:AbstractFixedSizePaddedArray}
            is_padded_array[n] = true
        elseif s[n] <: Base.Broadcast.Broadcasted{<:AbstractPaddedMatrixStyle}
            is_padded_array[n] = true
        else
            is_padded_array[n] = false
        end
#        is_adj_padded_array[n] = s[n] <: LinearAlgebra.Adjoint{<:Any,<:AbstractFixedSizePaddedArray}
    end
    any(is_padded_array) || return Base.Broadcast.DefaultArrayStyle{0}()
    
    A = LinearIndexing
    Svec = DataType[]
#    @show s
#    @show typeof(s)
#    throw("Bad!")
    if is_padded_array[1]
        bs = Base.Broadcast.BroadcastStyle(s[1])
        sₙ = typeof(bs).parameters
        push!(Svec, sₙ[1])
        T = sₙ[2]
        R = sₙ[3]
        A = sₙ[4]
    else#if s[1] <: Base.Broadcast.DefaultArrayStyle{0}()
        push!(Svec, Tuple{})
        T = s[1]
        R = typemax(Int)
        A = LinearIndexing
    end
    for n ∈ 2:N
        if is_padded_array[n]
            bs = Base.Broadcast.BroadcastStyle(s[n])
            sₙ = typeof(bs).parameters
#            sₙ = s[n].parameters
            push!(Svec, sₙ[1])
            T = promote_type(T, sₙ[2])
            R, access_pattern = pick_R(R, sₙ[3])
            A = max(A, sₙ[4], access_pattern)
        else
            push!(Svec, Tuple{})
            T = promote_type(T, s[2])
        end
    end
    ST = Tuple{Svec...}
    if (A == CartesianIndexing) || (A == KernelBatches)
        return FixedSizePaddedMatrixDefaultStyle{ST,T,R,A}()
    end
    sz = reduce_size(ST)
    nonscalarinds = (Svec .=== Tuple{}) .== false
    if A == LinearIndexing
        all_equal = true
        nonscalar = Svec[nonscalarinds]
        for n ∈ 2:length(nonscalar)
            all_equal &= (nonscalar[n] === nonscalar[n-1])
        end
        all_equal && return FixedSizePaddedMatrixDefaultStyle{ST,T,R,LinearIndexing}()
    end
    if length(sz) == 1
        return FixedSizePaddedMatrixDefaultStyle{ST,T,R,LinearIndexing}()
    else#if length(sz) > 2
        return FixedSizePaddedMatrixDefaultStyle{ST,T,R,CartesianIndexing}()
    end   
end
#@generated function Base.Broadcast.combine_styles(s::Vararg{FixedSizePaddedMatrixDefaultStyle,N}) where {N}
#
#    A = LinearIndexing
#    Svec = DataType[]
#    
#    sₙ = s[1].parameters
#    push!(Svec, sₙ[1])
#    T = Sₙ[1]
#    R = Sₙ[3]
#    A = Sₙ[4]
#    for n ∈ 2:N
#        sₙ = s[n].parameters
#        push!(Svec, sₙ[1])
#        T = promote_type(T, sₙ[2])
#        R, access_pattern = pick_R(R, sₙ[3])
#        A = max(A, sₙ[4], access_pattern)
#    end
#    S = Tuple{Svec...}
#    if (A == CartesianIndexing) || (A == KernelBatches)
#        return FixedSizePaddedMatrixDefaultStyle{S,T,R,A}()
#    end
#    sz = reduce_size(S)
#    nonscalarinds = (Svec .=== Tuple{}) .== false
#    if A == LinearIndexing
#        all_equal = true
#        nonscalar = Svec[nonscalarinds]
#        for n ∈ 2:length(nonscalar)
#            all_equal &= (nonscalar[n] === nonscalar[n-1])
#        end
#        all_equal && return FixedSizePaddedMatrixDefaultStyle{S,T,R,LinearIndexing}()
#    end
#    if length(sz) == 1
#        return FixedSizePaddedMatrixDefaultStyle{S,T,R,LinearIndexing}()
#    else#if length(sz) > 2
#        return FixedSizePaddedMatrixDefaultStyle{S,T,R,CartesianIndexing}()
#    end
#    
#end

@generated function Base.Broadcast.BroadcastStyle(
            style1::FixedSizePaddedMatrixDefaultStyle{S1,T1,R1,A1},
            style2::FixedSizePaddedMatrixDefaultStyle{S2,T2,R2,A2}
        ) where {S1,S2,T1,T2,R1,R2,A1,A2}

    T = promote_type(T1, T2)
    R, access_pattern = pick_R(R1, R2)
    if (A1 == KernelBatches) || (A2 == KernelBatches)
        return FixedSizePaddedMatrixDefaultStyle{Tuple{S1,S2},T,R,KernelBatches}()
    elseif (A1 == CartesianIndexing) || (A2 == CartesianIndexing) || (access_pattern == CartesianIndexing)
        return FixedSizePaddedMatrixDefaultStyle{Tuple{S1,S2},T,R,CartesianIndexing}()
    end
    sa1 = reduce_size(S1)
    sa2 = reduce_size(S2)
    l1 = length(sa1)
    l2 = length(sa2)
    if (((A1 == BatchedColumnMajor) || (A2 == BatchedColumnMajor)) && (max(l1,l2)>2))
        return FixedSizePaddedMatrixDefaultStyle{Tuple{S1,S2},T,R,CartesianIndexing}()
    end
    # 
    equal_lengths = l1 == l2
    if equal_lengths
        equal_dims = sa1 .== sa2
    elseif l1 < l2
        equal_dims = sa1 .== @view(sa2[1:l1])
    else
        equal_dims = @view(sa1[1:l2]) .== sa2
    end
    if all(equal_dims) && equal_lengths
        return FixedSizePaddedMatrixDefaultStyle{Tuple{S1,S2},T,R,max(A1,A2)}()
    elseif (max(length(sa1), length(sa2)) == 2) && equal_dims[1]
        return FixedSizePaddedMatrixDefaultStyle{Tuple{S1,S2},T,R,BatchedColumnMajor}()
    else
        return FixedSizePaddedMatrixDefaultStyle{Tuple{S1,S2},T,R,CartesianIndexing}()
    end
end
function Base.Broadcast.BroadcastStyle(
    style1::Base.Broadcast.BroadcastStyle,
    style2::FixedSizePaddedMatrixDefaultStyle{S,T,R,A}
) where {S,T,R,A}
    return style1
end


function Base.Broadcast.result_style(
    style1::FixedSizePaddedMatrixDefaultStyle{S,T,R,A},
    style2::Base.Broadcast.DefaultArrayStyle{0}
) where {S,T,R,A}
    return FixedSizePaddedMatrixDefaultStyle{Tuple{S,Tuple{}},T,R,A}()
end
function Base.Broadcast.result_style(
    style1::Base.Broadcast.DefaultArrayStyle{0},
    style2::FixedSizePaddedMatrixDefaultStyle{S,T,R,A}
) where {S,T,R,A}
    return FixedSizePaddedMatrixDefaultStyle{Tuple{Tuple{},S},T,R,A}()
end


# @generated function Base.BroadcastStyle(style1::AbstractPaddedMatrixStyle{S1,A1,C1}, style2::AbstractPaddedMatrixStyle{S2,A2,C2}) where {S1,A1,C1,S2,A2,C2}
#     A3 = max(A1, A2)
#     C3 = min(C1, C2)
#     if C3 == PaddedUnsized
#         S1N = C1 == PaddedUnsized ? S1 : length(S1.parameters)
#         S2N = C2 == PaddedUnsized ? S2 : length(S2.parameters)
#         S3 = max(S1N, S2N)
#     else
#         S1V = S1.parameters
#         S2V = S2.parameters
#         S1N = length(S1V)
#         S2N = length(S2V)
#         s3 = Int[]
#         for i ∈ 1:min(S1N,S2N)
#             if S1V[i] == S2V[i] || S2V[i] == 1
#                 push!(s3, S1V[i])
#             elseif S1V[i] == 1
#                 push!(s3, S2V[i])
#             else
#                 throw("Broadcast dimensions aren't alligned, $(S1V) vs $(S2V).")
#             end
#         end
#         sv_longer = S1N > S2N ? S1V : S2V
#         for i ∈ min(S1N,S2N)+1:max(S1N,S2N)
#             push!(s3, sv_longer[i])
#         end
#         S3 = Tuple{s3...}
#     end
#     :(PaddedMatrixStyle{$S3,$A3,$C3}())


# end
# Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{0}, style::AbstractPaddedMatrixStyle) = style


@generated function Base.similar(bc::Base.Broadcast.Broadcasted{FixedSizePaddedMatrixDefaultStyle{S,T,R,A}}) where {S,A,T,R}
    Salloc = reduce_size(S)
    #    N, R, L = calc_NPL(Salloc, T)
    N = length(Salloc)
    L = R
    for n ∈ 2:N
        L *= Salloc[n]
    end
    ST = Tuple{Salloc...}
    :(MutableFixedSizePaddedArray{$ST,$T,$N,$R,$L}(undef))
end
@generated function Base.similar(bc::Base.Broadcast.Broadcasted{FixedSizePaddedMatrixDefaultStyle{S,T1,R,A}}, mystack::Ptr{T2}) where {S,A,T1,T2,R}
    Salloc = reduce_size(S)
    #    N, R, L = calc_NPL(Salloc, T)
    N = length(Salloc)
    L = R
    for n ∈ 2:N
        L *= Salloc[n]
    end
    ST = Tuple{Salloc...}
    # allocates from mystack, and advances that pointer.
    if T1 == T2
        return :(PtrArray{$ST,$T1,$N,$R,$L,true}(mystack), mystack+$(L*sizeof(T)))
    else
        return :(PtrArray{$ST,$T1,$N,$R,$L,true}(Base.unsafe_convert(Ptr{$T1},mystack)), mystack+$(L*sizeof(T)))
    end
end

@inline extract(x::Base.RefValue) = x[]
@inline extract(x::Number) = x

## flattens a broadcast
function broadcast_index_expression(SB, N)
    SBV = SB.parameters
    inds = [gensym(Symbol(:n_,n)) for n ∈ 1:N]
    q = quote end
    preq = quote end
    assign_to = gensym(:assign)
    if length(SBV) > 0 && SBV[1] isa Int
        callexpr = quote end
        
        incorporate_cartesian_inds!(callexpr, preq, SBV,  inds, gensym(:arg), :(bc.args[1]))
        push!(q.args, Expr(:call, :setindex!, :out, callexpr, inds...))
        return inds, q, preq
    elseif length(SBV) == 0
        push!(q.args, :($assign_to = @inbounds  PaddedMatrices.extract($bcsym.args[1])))
        return ind, q, preq
    end
    broadcast_index_expression!(q, preq, SBV, inds, :bc, assign_to)
    push!(q.args, Expr(:call, :setindex!, :out, assign_to, inds...))
    inds, q, preq 
end
function incorporate_cartesian_inds!(callexpr, preq, SBV, inds, argsym, bcargsym)
            
    ind = Expr(:call, :getindex, argsym)
    push!(preq.args, :($argsym = @inbounds $bcargsym))
    for n ∈ 1:length(SBV)
        indₙ = (SBV[n])::Int
        if indₙ == 1
            push!(ind.args, 1)
        else
            push!(ind.args, inds[n])
        end
    end
    push!(callexpr.args, ind)

end
function broadcast_index_expression!(q, preq, SBV, inds, bcsym, assign)
    callexpr = Expr(:call, Expr(:., bcsym, :(:f)))
    for l ∈ eachindex(SBV)
        SBVₗ = ((SBV[l])::DataType).parameters
        argsym = gensym(:arg)
        if length(SBVₗ) == 0 # scalar argument
            push!(preq.args, :($argsym = @inbounds PaddedMatrices.extract($bcsym.args[$l])))
            push!(callexpr.args, argsym)
            continue
#        else

        end
        SBVₗ₁ = (SBVₗ[1])::Union{Int,DataType}
        if SBVₗ₁ isa Int # array argument
            incorporate_cartesian_inds!(callexpr, preq, SBVₗ, inds, argsym, :($bcsym.args[$l]))
        elseif SBVₗ₁ isa DataType # Another broadcastable
            push!(preq.args, :($argsym = @inbounds $bcsym.args[$l]))
            assign_to = gensym(:assign)
            broadcast_index_expression!(q, preq, SBVₗ, inds, argsym, assign_to)
            push!(callexpr.args, assign_to)
        end
    end
    push!(q.args, Expr(:(=), assign, callexpr))
    nothing
end
function broadcast_linearindex_expression(SB, N)
    ind = gensym(:ind)
    q = quote end
    preq = quote end
    assign_to = gensym(:assign)
    SBV = SB.parameters
    # @show length(SBV)
    if length(SBV) > 0 && SBV[1] isa Int
        # Then the broadcast contains only a single object
        callexpr = quote end
        argsym = gensym(:arg)
        push!(preq.args, :($argsym = @inbounds bc.args[1]))
        push!(callexpr.args, Expr(:call, :getindex, argsym, ind))
        push!(q.args, Expr(:call, :setindex!, :out, callexpr, ind))
        return ind, q, preq
    elseif length(SBV) == 0
        push!(q.args, :($assign_to = @inbounds  PaddedMatrices.extract(bc)))
        push!(q.args, Expr(:call, :setindex!, :out, assign_to, ind))
        return ind, q, preq
    end
    broadcast_linearindex_expression!(q, preq, SB.parameters, ind, :bc, assign_to)
    push!(q.args, Expr(:call, :setindex!, :out, assign_to, ind))
    ind, q, preq
end
function broadcast_linearindex_expression!(q, preq, SBV, ind, bcsym, assign)
    callexpr = Expr(:call, Expr(:., bcsym, :(:f)))
    for l ∈ eachindex(SBV)
        SBVₗ = ((SBV[l])::DataType).parameters
        argsym = gensym(:arg)
        if length(SBVₗ) == 0 # scalar argument
            push!(preq.args, :($argsym = @inbounds PaddedMatrices.extract($bcsym.args[$l])))
            push!(callexpr.args, argsym)
            continue
        else
            push!(preq.args, :($argsym = @inbounds $bcsym.args[$l]))
        end
        SBVₗ₁ = (SBVₗ[1])::Union{Int,DataType}
        if SBVₗ₁ isa Int # array argument
            push!(callexpr.args, Expr(:call, :getindex, argsym, ind))
        elseif SBVₗ₁ isa DataType
            assign_to = gensym(:assign)
            broadcast_linearindex_expression!(q, preq, SBVₗ, ind, argsym, assign_to)
            push!(callexpr.args, assign_to)
        end                           
    end
    push!(q.args, Expr(:(=), assign, callexpr))
    nothing
end

function materialize_quote(S, A, T, SB, N, P, R)
    #    @show S, A, T, SB, N, P
    R, A2 = pick_R(P, R)
    A = max(A, A2)
    if A == LinearIndexing
        ind, loop_body, pre_loop = broadcast_linearindex_expression(SB, N)
        L = R
        for n ∈ 2:N
            L *= S[n]
        end
        return quote
            $(Expr(:meta,:inline))
            $pre_loop
            LoopVectorization.@vectorize $T for $ind ∈ 1:$(L)
                $loop_body
            end
        end
    elseif A == BatchedColumnMajor
        @assert N == 2
        # Hvaven't actually implemented this as batched yet.
        # For now, will default to regular CartesianIndexing fallback.
        # A proper implementation would be able to estimate how many registers are left over,
        # and make sure those that shouldn't be reloaded are in fact not.
        # Ie, which arrays are column vectors vs matrices,
        # and are special functions like exp getting called, which will consume a lot of registers?
        inds, loop_body, pre_loop = broadcast_index_expression(SB, N)
#        M = S[1]
#        L = S[2]
#        Md, Mr = divrem(M, 
        loop = quote
            LoopVectorization.@vectorize $T for $(inds[1]) ∈ 1:$(S[1])
                $loop_body
            end
        end
        for n ∈ 2:N
            loop = quote
                for $(inds[n]) ∈ 1:$(S[n])
                    $loop
                end
            end
        end
        return quote
            $(Expr(:meta,:inline))
            $pre_loop
            $loop
        end
    elseif A == CartesianIndexing
        inds, loop_body, pre_loop = broadcast_index_expression(SB, N)
        loop = quote
#            for $(inds[1]) ∈ 1:$(S[1])
            LoopVectorization.@vectorize $T for $(inds[1]) ∈ 1:$(S[1])
                $loop_body
            end
        end
        for n ∈ 2:N
            loop = quote
                for $(inds[n]) ∈ 1:$(S[n])
                    $loop
                end
            end
        end
        return quote
            $(Expr(:meta,:inline))
            $pre_loop
            $loop
        end
    elseif A == KernelBatches
        throw("KernelBatches not yet implemented.")
    end
end

function materialize_quote(bc::Base.Broadcast.Broadcasted{FixedSizePaddedMatrixDefaultStyle{SB,T,R,A}}) where {A,T,SB,R}
    S = reduce_size(SB)
    prettify(materialize_quote(S, A, T, SB, length(S), R, R))
end


# FIXME:
# D .= A .+ B
# does not work when A is a matrix and B a 3d array
@generated function Base.Broadcast.materialize!(out::AbstractMutableFixedSizePaddedArray{S,T,N,P}, bc::Base.Broadcast.Broadcasted{FixedSizePaddedMatrixDefaultStyle{SB,T,R,A}}) where {S,A,T,SB,N,P,R}
    materialize_quote(S.parameters, A, T, SB, N, P, R)
end


@inline function Base.Broadcast.materialize(
    sp::StackPointer,
    bc::Base.Broadcast.Broadcasted{FixedSizePaddedMatrixDefaultStyle{S,T1,R,A}}
) where {S,A,T1,T2,R}
    sp, out = similar(sp, bc)
    Base.broadcast.materialize!(out, bc)
    sp, out
end
@inline function Base.Broadcast.materialize(
        bc::Base.Broadcast.Broadcasted{FixedSizePaddedMatrixDefaultStyle{S,T,R,A}}
    ) where {S,T,R,A}
#    ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
#    out = similar(bc, ElType)
    out = similar(bc)
    Base.Broadcast.materialize!(out, bc)
    out
end


#=
function batched_column_major_quote(S, A, C, T)

end

function kernel_batch_quote(S, A, C, T)
    N = length(S)
    # for now
    @assert N == 2
    M, P = S
    epr, m_k, p_k = pick_kernel_size(T, M, P)
    mrep, mrem = divrem(M, m_k)
    prep, prem = divrem(P, p_k)
    quote
        for p ∈ 0:prep-1
            for m ∈ 0:mrep-1

            end
        end
    end
end
=#
#=
@generated function Base.copyto!(
                        dest::AbstractMutableFixedSizePaddedArray{S1,T},
                        bc::Base.Broadcast.Broadcasted{PaddedMatrixStyle{S2,A,C}}
                    ) where {S1,S2,A,C,T}
    SV = S1.parameters
    if A == Agnostic || A == StandardColumnMajor
        return quote
            @inbounds @simd for I ∈ CartesianIndices($(Expr(:tuple, SV...)))
                dest[I] = bc[I]
            end
        end
    elseif A == BatchedColumnMajor
        return batched_column_major_quote(SV, A, C, T)
    elseif A == KernelBatches
        return kernel_bactch_quote(SV, A, C, T)
    end
end
=# 

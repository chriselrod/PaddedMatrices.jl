
abstract type AbstractStrideStyle{S,N} <: Base.Broadcast.AbstractArrayStyle{N} end
struct LinearStyle{S,N,X} <: AbstractStrideStyle{S,N} end
struct CartesianStyle{S,N} <: AbstractStrideStyle{S,N} end
Base.BroadcastStyle(::Type{A}) where {S,T,N,X,SN,XN,A<:AbstractStrideArray{S,T,N,X,SN,XN,false}} = LinearStyle{S,N,X}()
Base.BroadcastStyle(::Type{A}) where {S,T,N,X,SN,XN,A<:AbstractStrideArray{S,T,N,X,SN,XN,true}} = CartesianStyle{S,N}()
function reverse_simplevec(S, N = length(S))
    Srev = Expr(:curly, :Tuple)
    for n ∈ 1:N
        push!(Srev.args, S.parameters[N + 1 - n])
    end
    if N == 1
        N += 1
        insert!(Srev.args, 2, 1)
    end
    Srev, N
end
@generated function Base.BroadcastStyle(::Type{Adjoint{T,A}}) where {S,T,N,A<:AbstractStrideArray{S,T,N}}
    Srev, Nrev = reverse_simplevec(S, N)
    Expr(:call, Expr(:curly, :CartesianStyle, Srev, Nrev))
end
@generated function Base.BroadcastStyle(::Type{Transpose{T,A}}) where {S,T,N,A<:AbstractStrideArray{S,T,N}}
    Srev, Nrev = reverse_simplevec(S, N)
    Expr(:call, Expr(:curly, :CartesianStyle, Srev, Nrev))
end

const StrideArrayProduct = Union{
    LoopVectorization.Product{<:AbstractStrideArray},
    LoopVectorization.Product{<:Any,<:AbstractStrideArray},
    LoopVectorization.Product{<:AbstractStrideArray,<:AbstractStrideArray}
    # LoopVectorization.Product{Adjoint{<:Any,<:AbstractFixedSizeArray}},
    # LoopVectorization.Product{Transpose{<:Any,<:AbstractFixedSizeArray}},
    # LoopVectorization.Product{<:Any,Adjoint{<:Any,<:AbstractFixedSizeArray}},
    # LoopVectorization.Product{<:Any,Transpose{<:Any,<:AbstractFixedSizeArray}}
}

Base.BroadcastStyle(::Type{P}) where {M,K,A<:AbstractStrideMatrix{M,K},B<:AbstractStrideVector{K},P<:LoopVectorization.Product{A,B}} = CartesianStyle{Tuple{M},1}()
# Base.BroadcastStyle(::Type{P}) where {K,N,T,A<:AbstractStrideVector{K,T},B<:AbstractStrideMatrix{K,N},P<:LoopVectorization.Product{Adjoint{T,A},B}} = CartesianStyle{Tuple{1,N},2}()
# Base.BroadcastStyle(::Type{P}) where {K,N,T,A<:AbstractStrideVector{K,T},B<:AbstractStrideMatrix{K,N},P<:LoopVectorization.Product{Transpose{T,A},B}} = CartesianStyle{Tuple{1,N},2}()

Base.BroadcastStyle(::Type{P}) where {M,K,N,A<:AbstractStrideMatrix{M,K},B<:AbstractStrideMatrix{K,N},P<:LoopVectorization.Product{A,B}} = CartesianStyle{Tuple{M,N},2}()

# Base.BroadcastStyle(::Type{P}) where {M,K,N,TA,A<:AbstractStrideMatrix{K,M,TA},B<:AbstractStrideMatrix{K,N},P<:LoopVectorization.Product{Adjoint{TA,A},B}} = CartesianStyle{Tuple{M,N},2}()
# Base.BroadcastStyle(::Type{P}) where {M,K,N,TA,A<:AbstractStrideMatrix{K,M,TA},B<:AbstractStrideMatrix{K,N},P<:LoopVectorization.Product{Transpose{TA,A},B}} = CartesianStyle{Tuple{M,N},2}()

# Base.BroadcastStyle(::Type{P}) where {M,K,N,A<:AbstractStrideMatrix{M,K},TB,B<:AbstractStrideMatrix{N,K,TB},P<:LoopVectorization.Product{A,Adjoint{TB,B}}} = CartesianStyle{Tuple{M,N},2}()
# Base.BroadcastStyle(::Type{P}) where {M,K,N,A<:AbstractStrideMatrix{M,K},TB,B<:AbstractStrideMatrix{N,K,TB},P<:LoopVectorization.Product{A,Transpose{TB,B}}} = CartesianStyle{Tuple{M,N},2}()

# Base.BroadcastStyle(::Type{P}) where {M,K,N,TA,A<:AbstractStrideMatrix{K,M,TA},TB,B<:AbstractStrideMatrix{N,K,TB},P<:LoopVectorization.Product{Adjoint{TA,A},Adjoint{TB,B}}} = CartesianStyle{Tuple{M,N},2}()
# Base.BroadcastStyle(::Type{P}) where {M,K,N,TA,A<:AbstractStrideMatrix{K,M,TA},TB,B<:AbstractStrideMatrix{N,K,TB},P<:LoopVectorization.Product{Adjoint{TA,A},Transpose{TB,B}}} = CartesianStyle{Tuple{M,N},2}()
# Base.BroadcastStyle(::Type{P}) where {M,K,N,TA,A<:AbstractStrideMatrix{K,M,TA},TB,B<:AbstractStrideMatrix{N,K,TB},P<:LoopVectorization.Product{Transpose{TA,A},Adjoint{TB,B}}} = CartesianStyle{Tuple{M,N},2}()
# Base.BroadcastStyle(::Type{P}) where {M,K,N,TA,A<:AbstractStrideMatrix{K,M,TA},TB,B<:AbstractStrideMatrix{N,K,TB},P<:LoopVectorization.Product{Transpose{TA,A},Transpose{TB,B}}} = CartesianStyle{Tuple{M,N},2}()

@generated Base.BroadcastStyle(a::CartesianStyle{S,N1}, b::Base.Broadcast.DefaultArrayStyle{N2}) where {S,N1,N2} = N2 > N1 ? Base.Broadcast.Unknown() : :a
@generated Base.BroadcastStyle(a::LinearStyle{S,N1}, b::Base.Broadcast.DefaultArrayStyle{N2}) where {S,N1,N2} = N2 > N1 ? Base.Broadcast.Unknown() : CartesianStyle{S,N1}()
Base.BroadcastStyle(a::CartesianStyle{S,N}, b::AbstractStrideStyle{S,N}) where {S,N} = a
Base.BroadcastStyle(a::LinearStyle{S,N,X}, b::LinearStyle{S,N,X}) where {S,N,X} = a
Base.BroadcastStyle(a::LinearStyle{S,N}, b::LinearStyle{S,N}) where {S,N} = CartesianStyle{S,N}()
@generated function Base.BroadcastStyle(a::AbstractStrideStyle{S1,N1}, b::AbstractStrideStyle{S2,N2}) where {S1,S2,N1,N2}
    S = Expr(:curly, :Tuple)
#    @show N2, N1
    N2 > N1 && return Base.Broadcast.Unknown()
    # foundfirstdiff = false
    for n ∈ 1:N2#min(N1,N2)
        s1 = (S1.parameters[n])::Int
        s2 = (S2.parameters[n])::Int
        if s1 == s2
            push!(S.args, s1)
        elseif s2 == 1
            # foundfirstdiff = true
            push!(S.args, s1)
        elseif s1 == 1
            # foundfirstdiff || return Base.Broadcast.Unknown()
            # foundfirstdiff = true
            push!(S.args, s2)
        elseif s2 == -1
            push!(S.args, s1)
        elseif s1 == -1
            push!(S.args, s2)
        else
            throw("Mismatched sizes: $S1, $S2.")
        end
    end
    # if N2 > N1
    #     for n ∈ N1+1:N2
    #         push!(S.args, S2.parameters[n])
    #     end
    # else
    # if N1 > N2
    for n ∈ N2+1:N1
        push!(S.args, S1.parameters[n])
    end
    # end
    Expr(:call, Expr(:curly, :CartesianStyle, S, max(N1,N2)))
end

# function Base.similar(
    # ::Base.Broadcast.Broadcasted{FS}, ::Type{T}
# ) where {S,T<:Union{VectorizationBase.FloatingTypes,PaddedMatrices.VectorizationBase.IntTypes,PaddedMatrices.VectorizationBase.UIntTypes},N,FS<:AbstractStrideStyle{S,N}}
function Base.similar(
    bc::Base.Broadcast.Broadcasted{FS}, ::Type{T}
) where {S,T,N,FS<:AbstractStrideStyle{S,N}}
    if isfixed(S)
        FixedSizeArray{S,T}(undef)
    else
        StrideArray{S,T}(undef, size(bc))
    end
end
Base.similar(sp::StackPointer, ::Base.Broadcast.Broadcasted{FS}, ::Type{T}) where {S,T,FS<:AbstractStrideStyle{S,T},N} = PtrArray{S,T}(sp)


function add_single_element_array!(ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol, elementbytes)
    LoopVectorization.pushpreamble!(ls, Expr(:(=), Symbol("##", destname), Expr(:call, :first, bcname)))
    LoopVectorization.add_constant!(ls, destname, elementbytes)
end
function add_fs_array!(ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, indexes, S, X::Vector{Int}, elementbytes)
    ref = Symbol[]
    # aref = LoopVectorization.ArrayReference(bcname, ref)
    vptrbc = LoopVectorization.vptr(bcname)
    LoopVectorization.add_vptr!(ls, bcname, vptrbc, true, false) #TODO: is this necessary?
    offset = 0
    for (i,n) ∈ enumerate(indexes)
        s = (S.parameters[i])::Int
        stride = X[i]
        # (isone(n) & (stride != 1)) && pushfirst!(ref, LoopVectorization.DISCONTIGUOUS)
        if iszero(stride) || s == 1
            offset += 1
            bco = bcname
            bcname = Symbol(:_, bcname)
            v = Expr(:call, :view, bco)
            foreach(_ -> push!(v.args, :(:)), 1:i - offset)
            push!(v.args, Expr(:call, Expr(:curly, :Static, 1)))
            foreach(_ -> push!(v.args, :(:)), i+1:length(indexes))
            LoopVectorization.pushpreamble!(ls, Expr(:(=), bcname, v))
            # vptrbc = LoopVectorization.subset_vptr!(
            #     ls, vptrbc, i - offset, 1, loopsyms, fill(true, length(indexes))
            # )
        else
            push!(ref, loopsyms[n])
        end
    end
    if iszero(length(ref))
        return add_single_element_array!(ls, destname, bcname, elementbytes)
    end
    bctemp = Symbol(:_, bcname)
    mref = LoopVectorization.ArrayReferenceMeta(
        LoopVectorization.ArrayReference(bctemp, ref), fill(true, length(ref)), vptrbc
    )
    sp = sort_indices!(mref, X)
    if isnothing(sp)
        LoopVectorization.pushpreamble!(ls, Expr(:(=), bctemp,  bcname))
        # LoopVectorization.add_vptr!(ls, bcname, vptrbc, true, false)
    else
        ssp = Expr(:tuple); append!(ssp.args, sp)
        ssp = Expr(:call, Expr(:curly, :Static, ssp))
        LoopVectorization.pushpreamble!(ls, Expr(:(=), bctemp,  Expr(:call, :PermutedDimsArray, bcname, ssp)))
        # LoopVectorization.add_vptr!(ls, bctemp, vptrbc, true, false)
        # vptemp = gensym(vptrbc)
        # LoopVectorization.add_vptr!(ls, bcname, vptemp, true, false)
        # LoopVectorization.pushpreamble!(Expr(:(=), vptrbc,  Expr(:call, :PermutedDimsArray, vptemp, ssp)))
    end
    LoopVectorization.add_simple_load!(ls, destname, mref, mref.ref.indices, elementbytes)
end

function add_broadcast_adjoint_array!(
    ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, ::Type{A}, indices, elementbytes::Int = 8
) where {S, T, N, A <: AbstractStrideArray{S,T,N}}
    if N == 1
        if first(S.parameters)::Int == 1
            add_single_element_array!(ls, destname, bcname, sizeof(T))
        else
            ref = LoopVectorization.ArrayReference(bcname, Symbol[loopsyms[2]])
            LoopVectorization.add_load!( ls, destname, ref, sizeof(T) )
        end
    else
        add_fs_array!(ls, destname, bcname, loopsyms, indices, S, sizeof(T))
    end
end
function LoopVectorization.add_broadcast!(
    ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol,
    loopsyms::Vector{Symbol}, ::Type{Adjoint{T,A}}, elementbytes::Int = 8
) where {T, S, N, A <: AbstractStrideArray{S, T, N}}
    # @show @__LINE__, A
    add_broadcast_adjoint_array!( ls, destname, bcname, loopsyms, A, N:-1:1, sizeof(T) )
end
function LoopVectorization.add_broadcast!(
    ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol,
    loopsyms::Vector{Symbol}, ::Type{Transpose{T,A}}, elementbytes::Int = 8
) where {T, S, N, A <: AbstractStrideArray{S, T, N}}
    # @show @__LINE__, A
    add_broadcast_adjoint_array!( ls, destname, bcname, loopsyms, A, N:-1:1, sizeof(T) )
end
function LoopVectorization.add_broadcast!(
    ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol,
    loopsyms::Vector{Symbol}, ::Type{PermutedDimsArray{T,N,I1,I2,A}}, elementbytes::Int = 8
) where {T, S, N, I1, I2, A <: AbstractStrideArray{S, T, N}}
    @show @__LINE__, A
    add_broadcast_adjoint_array!( ls, destname, bcname, loopsyms, A, I2, sizeof(T) )
end

function sort_indices!(ar, Xv)
    li = ar.loopedindex; NN = length(li)
    all(n -> ((Xv[n+1]) % UInt) ≥ ((Xv[n]) % UInt), 1:NN-1) && return nothing    
    inds = LoopVectorization.getindices(ar)
    sp = sortperm(reinterpret(UInt,Xv), alg = Base.Sort.DEFAULT_STABLE)
    lib = copy(li); indsb = copy(inds)
    for i ∈ eachindex(li, inds)
        li[i] = lib[sp[i]]
        inds[i] = indsb[sp[i]]
    end
    isone(Xv[sp[1]]) || pushfirst!(inds, LoopVectorization.DISCONTIGUOUS)
    sp
end

function LoopVectorization.add_broadcast!(
    ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol,
    loopsyms::Vector{Symbol}, ::Type{A}, elementbytes::Int = 8
) where {T, S, X, N, A <: AbstractStrideArray{S, T, N, X}}
    # @show @__LINE__, A
    Xv = tointvec(X)
    NN = min(N,length(loopsyms))
    op = add_fs_array!(
        ls, destname, bcname, loopsyms, 1:NN, S, Xv, sizeof(T)
    )
    op
end


# function Base.Broadcast.materialize!(
@generated function Base.Broadcast.materialize!(
    dest::A, bc::BC
) where {S, T <: Union{Float32,Float64}, N, X, L, A <: AbstractStrideArray{S,T,N,X,L}, FS <: LinearStyle{S,N,X}, BC <: Base.Broadcast.Broadcasted{FS}}
    # we have an N dimensional loop.
    # need to construct the LoopSet
    loopsyms = [gensym(:n)]
    ls = LoopVectorization.LoopSet(:PaddedMatrices)
    itersym = first(loopsyms)
    if L == -1
        Lsym = gensym(:L)
        LoopVectorization.pushpreamble!(ls, Expr(:(=), Lsym, Expr(:call, :length, :dest)))
        LoopVectorization.add_loop!(ls, LoopVectorization.Loop(itersym, 1, Lsym))
    else
        LoopVectorization.add_loop!(ls, LoopVectorization.Loop(itersym, 1, L))
    end
    elementbytes = sizeof(T)
    LoopVectorization.add_broadcast!(ls, :dest, :bc, loopsyms, BC, elementbytes)
    LoopVectorization.add_simple_store!(ls, :dest, LoopVectorization.ArrayReference(:dest, loopsyms), elementbytes)
    resize!(ls.loop_order, LoopVectorization.num_loops(ls)) # num_loops may be greater than N, eg Product
    Expr(
        :block,
        Expr(:meta,:inline),
        ls.prepreamble,
        LoopVectorization.lower(ls),
        :dest
    )
    # ls
end
@generated function Base.Broadcast.materialize!(
# function Base.Broadcast.materialize!(
    dest::A, bc::BC
) where {S, T <: Union{Float32,Float64}, N, X, A <: AbstractStrideArray{S,T,N,X}, BC <: Union{Base.Broadcast.Broadcasted,StrideArrayProduct}}
    1+1
    # we have an N dimensional loop.
    # need to construct the LoopSet
    loopsyms = [gensym(:n) for n ∈ 1:N]
    ls = LoopVectorization.LoopSet(:PaddedMatrices)
    destref = LoopVectorization.ArrayReference(:_dest, copy(loopsyms))
    destmref = LoopVectorization.ArrayReferenceMeta(destref, fill(true, length(LoopVectorization.getindices(destref))))
    sp = sort_indices!(destmref, tointvec(X))
    for n ∈ 1:N
        itersym = loopsyms[n]#isnothing(sp) ? n : sp[n]]
        Sₙ = (S.parameters[n])::Int
        if Sₙ == -1
            Sₙsym = gensym(:Sₙ)
            LoopVectorization.pushpreamble!(ls, Expr(:(=), Sₙsym, Expr(:call, :size, :dest, n)))
            LoopVectorization.add_loop!(ls, LoopVectorization.Loop(itersym, 1, Sₙsym))
        else
            LoopVectorization.add_loop!(ls, LoopVectorization.Loop(itersym, 1, Sₙ))
        end
    end
    elementbytes = sizeof(T)
    # destadj = length(X.parameters) > 1 && last(X.parameters)::Int == 1
    # if destadj
    #     destsym = :dest′
    #     LoopVectorization.pushpreamble!(ls, Expr(:(=), destsym, Expr(:call, Expr(:(.), :LinearAlgebra, QuoteNode(:Transpose)), :dest)))
    # else
    # end
    LoopVectorization.add_broadcast!(ls, :destination, :bc, loopsyms, BC, elementbytes)
    if isnothing(sp)
        LoopVectorization.pushpreamble!(ls, Expr(:(=), :_dest, :dest))
    else
        ssp = Expr(:tuple); append!(ssp.args, sp)
        ssp = Expr(:call, Expr(:curly, :Static, ssp))
        LoopVectorization.pushpreamble!(ls, Expr(:(=), :_dest,  Expr(:call, :PermutedDimsArray, :dest, ssp)))
    end
    storeop = LoopVectorization.add_simple_store!(ls, :destination, destmref, sizeof(T))
    # destref = if destadj
    #     ref = LoopVectorization.ArrayReference(:dest′, reverse(loopsyms))
    # else
    #     if first(X.parameters)::Int != 1
    #         pushfirst!(LoopVectorization.getindices(ref), Symbol("##DISCONTIGUOUSSUBARRAY##"))
    #     end
    #     ref
    # end
    resize!(ls.loop_order, LoopVectorization.num_loops(ls)) # num_loops may be greater than N, eg Product
    # ls.vector_width[] = VectorizationBase.pick_vector_width(T)
    # return ls
    Expr(
        :block,
        ls.prepreamble,
        LoopVectorization.lower(ls, 0),
        :dest
    )
end

# @generated function Base.Broadcast.materialize!(
#     dest′::Union{Adjoint{T,A},Transpose{T,A}}, bc::BC
# ) where {S, T <: Union{Float32,Float64}, N, X, A <: AbstractStrideArray{S,T,N,X}, BC <: Union{Base.Broadcast.Broadcasted,StrideArrayProduct}}
#     # we have an N dimensional loop.
#     # need to construct the LoopSet
#     loopsyms = [gensym(:n) for n ∈ 1:N]
#     ls = LoopVectorization.LoopSet(:PaddedMatrices)
#     LoopVectorization.pushpreamble!(ls, Expr(:(=), :dest, Expr(:call, :parent, :dest′)))
#     for (n,itersym) ∈ enumerate(loopsyms)
#         LoopVectorization.add_loop!(ls, LoopVectorization.Loop(itersym, 1, (S.parameters[n])::Int))
#     end
#     elementbytes = sizeof(T)
#     LoopVectorization.add_broadcast!(ls, :dest, :bc, loopsyms, BC, elementbytes)
#     LoopVectorization.add_simple_store!(ls, :dest, LoopVectorization.ArrayReference(:dest, reverse(loopsyms)), elementbytes)
#     resize!(ls.loop_order, num_loops(ls)) # num_loops may be greater than N, eg Product
#     q = LoopVectorization.lower(ls)
#     push!(q.args, :dest′)
#     pushfirst!(q.args, Expr(:meta,:inline))
#     q
#     # ls
# end
@inline function Base.Broadcast.materialize(bc::Base.Broadcast.Broadcasted{S}) where {S <: CartesianStyle}
    ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
    Base.Broadcast.materialize!(similar(bc, ElType), bc)
end

LoopVectorization.vmaterialize(bc::Base.Broadcast.Broadcasted{<:AbstractStrideStyle}) = Base.Broadcast.materialize(bc)
LoopVectorization.vmaterialize!(dest, bc::Base.Broadcast.Broadcasted{<:AbstractStrideStyle}) = Base.Broadcast.materialize!(dest, bc)

LoopVectorization.vmaterialize(bc::StrideArrayProduct) = Base.Broadcast.materialize(bc)
LoopVectorization.vmaterialize!(dest, bc::StrideArrayProduct) = Base.Broadcast.materialize!(dest, bc)



abstract type AbstractFixedSizeStyle{S,N} <: Base.Broadcast.AbstractArrayStyle{N} end
struct FixedSizeLinearStyle{S,N,X} <: AbstractFixedSizeStyle{S,N} end
struct FixedSizeStyle{S,N} <: AbstractFixedSizeStyle{S,N} end
Base.BroadcastStyle(::Type{A}) where {S,T,N,X,A<:AbstractFixedSizeArray{S,T,N,X}} = FixedSizeLinearStyle{S,N,X}()
Base.BroadcastStyle(::Type{A}) where {S,T,N,X,L,A<:PtrArray{S,T,N,X,L,true}} = FixedSizeStyle{S,N}()
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
@generated function Base.BroadcastStyle(::Type{Adjoint{T,A}}) where {S,T,N,A<:AbstractFixedSizeArray{S,T,N}}
    Srev, Nrev = reverse_simplevec(S, N)
    Expr(:call, Expr(:curly, :FixedSizeStyle, Srev, Nrev))
end
@generated function Base.BroadcastStyle(::Type{Transpose{T,A}}) where {S,T,N,A<:AbstractFixedSizeArray{S,T,N}}
    Srev, Nrev = reverse_simplevec(S, N)
    Expr(:call, Expr(:curly, :FixedSizeStyle, Srev, Nrev))
end

const FixedSizeProduct = Union{
    LoopVectorization.Product{<:AbstractFixedSizeArray},
    LoopVectorization.Product{<:Any,<:AbstractFixedSizeArray},
    LoopVectorization.Product{Adjoint{<:Any,<:AbstractFixedSizeArray}},
    LoopVectorization.Product{Transpose{<:Any,<:AbstractFixedSizeArray}},
    LoopVectorization.Product{<:Any,Adjoint{<:Any,<:AbstractFixedSizeArray}},
    LoopVectorization.Product{<:Any,Transpose{<:Any,<:AbstractFixedSizeArray}}
}

Base.BroadcastStyle(::Type{P}) where {M,K,A<:AbstractFixedSizeMatrix{M,K},B<:AbstractFixedSizeVector{K},P<:LoopVectorization.Product{A,B}} = FixedSizeStyle{Tuple{M},1}()
Base.BroadcastStyle(::Type{P}) where {K,N,T,A<:AbstractFixedSizeVector{K,T},B<:AbstractFixedSizeMatrix{K,N},P<:LoopVectorization.Product{Adjoint{T,A},B}} = FixedSizeStyle{Tuple{1,N},2}()
Base.BroadcastStyle(::Type{P}) where {K,N,T,A<:AbstractFixedSizeVector{K,T},B<:AbstractFixedSizeMatrix{K,N},P<:LoopVectorization.Product{Transpose{T,A},B}} = FixedSizeStyle{Tuple{1,N},2}()

Base.BroadcastStyle(::Type{P}) where {M,K,N,A<:AbstractFixedSizeMatrix{M,K},B<:AbstractFixedSizeMatrix{K,N},P<:LoopVectorization.Product{A,B}} = FixedSizeStyle{Tuple{M,N},2}()

Base.BroadcastStyle(::Type{P}) where {M,K,N,TA,A<:AbstractFixedSizeMatrix{K,M,TA},B<:AbstractFixedSizeMatrix{K,N},P<:LoopVectorization.Product{Adjoint{TA,A},B}} = FixedSizeStyle{Tuple{M,N},2}()
Base.BroadcastStyle(::Type{P}) where {M,K,N,TA,A<:AbstractFixedSizeMatrix{K,M,TA},B<:AbstractFixedSizeMatrix{K,N},P<:LoopVectorization.Product{Transpose{TA,A},B}} = FixedSizeStyle{Tuple{M,N},2}()

Base.BroadcastStyle(::Type{P}) where {M,K,N,A<:AbstractFixedSizeMatrix{M,K},TB,B<:AbstractFixedSizeMatrix{N,K,TB},P<:LoopVectorization.Product{A,Adjoint{TB,B}}} = FixedSizeStyle{Tuple{M,N},2}()
Base.BroadcastStyle(::Type{P}) where {M,K,N,A<:AbstractFixedSizeMatrix{M,K},TB,B<:AbstractFixedSizeMatrix{N,K,TB},P<:LoopVectorization.Product{A,Transpose{TB,B}}} = FixedSizeStyle{Tuple{M,N},2}()

Base.BroadcastStyle(::Type{P}) where {M,K,N,TA,A<:AbstractFixedSizeMatrix{K,M,TA},TB,B<:AbstractFixedSizeMatrix{N,K,TB},P<:LoopVectorization.Product{Adjoint{TA,A},Adjoint{TB,B}}} = FixedSizeStyle{Tuple{M,N},2}()
Base.BroadcastStyle(::Type{P}) where {M,K,N,TA,A<:AbstractFixedSizeMatrix{K,M,TA},TB,B<:AbstractFixedSizeMatrix{N,K,TB},P<:LoopVectorization.Product{Adjoint{TA,A},Transpose{TB,B}}} = FixedSizeStyle{Tuple{M,N},2}()
Base.BroadcastStyle(::Type{P}) where {M,K,N,TA,A<:AbstractFixedSizeMatrix{K,M,TA},TB,B<:AbstractFixedSizeMatrix{N,K,TB},P<:LoopVectorization.Product{Transpose{TA,A},Adjoint{TB,B}}} = FixedSizeStyle{Tuple{M,N},2}()
Base.BroadcastStyle(::Type{P}) where {M,K,N,TA,A<:AbstractFixedSizeMatrix{K,M,TA},TB,B<:AbstractFixedSizeMatrix{N,K,TB},P<:LoopVectorization.Product{Transpose{TA,A},Transpose{TB,B}}} = FixedSizeStyle{Tuple{M,N},2}()

@generated Base.BroadcastStyle(a::FixedSizeStyle{S,N1}, b::Base.Broadcast.DefaultArrayStyle{N2}) where {S,N1,N2} = N2 > N1 ? Base.Broadcast.Unknown() : :a
@generated Base.BroadcastStyle(a::FixedSizeLinearStyle{S,N1}, b::Base.Broadcast.DefaultArrayStyle{N2}) where {S,N1,N2} = N2 > N1 ? Base.Broadcast.Unknown() : FixedSizeStyle{S,N1}()
Base.BroadcastStyle(a::FixedSizeStyle{S,N}, b::AbstractFixedSizeStyle{S,N}) where {S,N} = a
Base.BroadcastStyle(a::FixedSizeLinearStyle{S,N,X}, b::FixedSizeLinearStyle{S,N,X}) where {S,N,X} = a
Base.BroadcastStyle(a::FixedSizeLinearStyle{S,N}, b::FixedSizeLinearStyle{S,N}) where {S,N} = FixedSizeStyle{S,N}()
@generated function Base.BroadcastStyle(a::AbstractFixedSizeStyle{S1,N1}, b::AbstractFixedSizeStyle{S2,N2}) where {S1,S2,N1,N2}
    S = Expr(:curly, :Tuple)
    N2 > N1 && return Base.Broadcast.Unknown()
    foundfirstdiff = false
    for n ∈ 1:min(N1,N2)
        s1 = (S1.parameters[n])::Int
        s2 = (S2.parameters[n])::Int
        if s1 == s2
            push!(S.args, s1)
        elseif s2 == 1
            foundfirstdiff = true
            push!(S.args, s1)
        elseif s1 == 1
            foundfirstdiff || return Base.Broadcast.Unknown()
            foundfirstdiff = true
            push!(S.args, s2)
        else
            throw("Mismatched sizes: $S1, $S2.")
        end
    end
    if N2 > N1
        for n ∈ N1+1:N2
            push!(S.args, S2.parameters[n])
        end
    elseif N1 > N2
        for n ∈ N2+1:N1
            push!(S.args, S1.parameters[n])
        end
    end
    Expr(:call, Expr(:curly, :FixedSizeStyle, S, max(N1,N2)))
end

Base.similar(::Base.Broadcast.Broadcasted{FS}, ::Type{T}) where {S,T,N,FS<:AbstractFixedSizeStyle{S,N}} = FixedSizeArray{S,T,N}(undef)
Base.similar(sp::StackPointer, ::Base.Broadcast.Broadcasted{FS}, ::Type{T}) where {S,T,FS<:AbstractFixedSizeStyle{S,T},N} = PtrArray{S,T,N}(sp)

function add_single_element_array!(ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol, elementbytes)
    LoopVectorization.pushpreamble!(ls, Expr(:(=), Symbol("##", destname), Expr(:call, :first, bcname)))
    LoopVectorization.add_constant!(ls, destname, elementbytes)
end
function add_fs_array!(ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, indexes, S, elementbytes)
    ref = Symbol[]
    for (i,n) ∈ enumerate(indexes)
        s = (S.parameters[i])::Int
        s == 1 || push!(ref, loopsyms[n])
    end
    if length(ref) > 0
        LoopVectorization.add_simple_load!(ls, destname, LoopVectorization.ArrayReference(bcname, ref), elementbytes)
    else
        add_single_element_array!(ls, destname, bcname, elementbytes)
    end
end

function add_broadcast_adjoint_array!(
    ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, ::Type{A}, elementbytes::Int = 8
) where {S, T, N, A <: AbstractFixedSizeArray{S,T,N}}
    if N == 1
        if first(S.parameters)::Int == 1
            add_single_element_array!(ls, destname, bcname, sizeof(T))
        else
            ref = LoopVectorization.ArrayReference(bcname, Symbol[loopsyms[2]])
            LoopVectorization.add_load!( ls, destname, ref, sizeof(T) )
        end
    else
        add_fs_array!(ls, destname, bcname, loopsyms, N:-1:1, S, sizeof(T))
    end
end
function LoopVectorization.add_broadcast!(
    ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    ::Type{Adjoint{T,A}}, elementbytes::Int = 8
) where {T, S, A <: AbstractFixedSizeArray{S, T}}
    add_broadcast_adjoint_array!( ls, destname, bcname, loopsyms, A, sizeof(T) )
end
function LoopVectorization.add_broadcast!(
    ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    ::Type{Transpose{T,A}}, elementbytes::Int = 8
) where {T, S, A <: AbstractFixedSizeArray{S, T}}
    add_broadcast_adjoint_array!( ls, destname, bcname, loopsyms, A, sizeof(T) )
end
function LoopVectorization.add_broadcast!(
    ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    ::Type{A}, elementbytes::Int = 8
) where {T, S, N, A <: AbstractFixedSizeArray{S, T, N}}
    add_fs_array!(ls, destname, bcname, loopsyms, 1:min(N,length(loopsyms)), S, sizeof(T))
end


@generated function Base.Broadcast.materialize!(
    dest::A, bc::BC
) where {S, T <: Union{Float32,Float64}, N, X, L, A <: AbstractFixedSizeArray{S,T,N,X,L}, FS <: FixedSizeLinearStyle{S,N,X}, BC <: Base.Broadcast.Broadcasted{FS}}
    # we have an N dimensional loop.
    # need to construct the LoopSet
    loopsyms = [gensym(:n)]
    ls = LoopVectorization.LoopSet(:PaddedMatrices)
    for (n,itersym) ∈ enumerate(loopsyms)
        LoopVectorization.add_loop!(ls, LoopVectorization.Loop(itersym, 0, L))
    end
    elementbytes = sizeof(T)
    LoopVectorization.add_broadcast!(ls, :dest, :bc, loopsyms, BC, elementbytes)
    LoopVectorization.add_simple_store!(ls, :dest, LoopVectorization.ArrayReference(:dest, loopsyms), elementbytes)
    resize!(ls.loop_order, LoopVectorization.num_loops(ls)) # num_loops may be greater than N, eg Product
    q = LoopVectorization.lower(ls)
    push!(q.args, :dest)
    pushfirst!(q.args, Expr(:meta,:inline))
    q
    # ls
end
@generated function Base.Broadcast.materialize!(
    dest::A, bc::BC
) where {S, T <: Union{Float32,Float64}, N, X, A <: AbstractFixedSizeArray{S,T,N,X}, BC <: Union{Base.Broadcast.Broadcasted,FixedSizeProduct}}
    # we have an N dimensional loop.
    # need to construct the LoopSet
    loopsyms = [gensym(:n) for n ∈ 1:N]
    ls = LoopVectorization.LoopSet(:PaddedMatrices)
    for (n,itersym) ∈ enumerate(loopsyms)
        LoopVectorization.add_loop!(ls, LoopVectorization.Loop(itersym, 0, (S.parameters[n])::Int))
    end
    elementbytes = sizeof(T)
    LoopVectorization.add_broadcast!(ls, :dest, :bc, loopsyms, BC, elementbytes)
    LoopVectorization.add_simple_store!(ls, :dest, LoopVectorization.ArrayReference(:dest, loopsyms), elementbytes)
    resize!(ls.loop_order, LoopVectorization.num_loops(ls)) # num_loops may be greater than N, eg Product
    q = LoopVectorization.lower(ls)
    push!(q.args, :dest)
    pushfirst!(q.args, Expr(:meta,:inline))
    q
    # ls
end

@generated function Base.Broadcast.materialize!(
    dest′::Union{Adjoint{T,A},Transpose{T,A}}, bc::BC
) where {S, T <: Union{Float32,Float64}, N, X, A <: AbstractFixedSizeArray{S,T,N,X}, BC <: Union{Base.Broadcast.Broadcasted,FixedSizeProduct}}
    # we have an N dimensional loop.
    # need to construct the LoopSet
    loopsyms = [gensym(:n) for n ∈ 1:N]
    ls = LoopVectorization.LoopSet(:PaddedMatrices)
    pushpreamble!(ls, Expr(:(=), :dest, Expr(:call, :parent, :dest′)))
    for (n,itersym) ∈ enumerate(loopsyms)
        LoopVectorization.add_loop!(ls, LoopVectorization.Loop(itersym, 0, (S.parameters[n])::Int))
    end
    elementbytes = sizeof(T)
    LoopVectorization.add_broadcast!(ls, :dest, :bc, loopsyms, BC, elementbytes)
    LoopVectorization.add_simple_store!(ls, :dest, LoopVectorization.ArrayReference(:dest, reverse(loopsyms)), elementbytes)
    resize!(ls.loop_order, num_loops(ls)) # num_loops may be greater than N, eg Product
    q = LoopVectorization.lower(ls)
    push!(q.args, :dest′)
    pushfirst!(q.args, Expr(:meta,:inline))
    q
    # ls
end
@inline function Base.Broadcast.materialize(bc::Base.Broadcast.Broadcasted{S}) where {S <: FixedSizeStyle}
    ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
    Base.Broadcast.materialize!(similar(bc, ElType), bc)
end

LoopVectorization.vmaterialize(bc::Base.Broadcast.Broadcasted{<:AbstractFixedSizeStyle}) = Base.Broadcast.materialize(bc)
LoopVectorization.vmaterialize!(dest, bc::Base.Broadcast.Broadcasted{<:AbstractFixedSizeStyle}) = Base.Broadcast.materialize!(dest, bc)

LoopVectorization.vmaterialize(bc::FixedSizeProduct) = Base.Broadcast.materialize(bc)
LoopVectorization.vmaterialize!(dest, bc::FixedSizeProduct) = Base.Broadcast.materialize!(dest, bc)


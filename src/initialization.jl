
function calc_padding(nrow::Int, T)
    W = VectorizationBase.pick_vector_width(T)
    W > nrow ? VectorizationBase.nextpow2(nrow) : VectorizationBase.align(nrow, T)
end

static_expr(x) = Expr(:call, Expr(:curly, :Static, x))
function axis_expressions(x, s)
    out = Vector{Expr}(undef, length(x))
    n = 0
    for i ∈ eachindex(x)
        xᵢ = (x[i])::Int
        out[i] = xᵢ == -1 ? Expr(:ref, s, n += 1) : static_expr(xᵢ)
    end
    out
end

@inline new_view(A::PtrArray, sn, xn, org) = PtrArray(A.ptr, sn, xn, org)
@inline new_view(A::StrideArray, sn, xn, org) = StrideArray(new_view(A.ptr, sn, xn, org), A.data)
@inline new_view(A::FixedSizeArray, sn, xn, org) = FixedSizeArray(new_view(A.ptr, sn, xn, org), A.data)

function sizedefs(@nospecialize(s), sym)
    static = Expr(:tuple)
    dynamic = Expr(:tuple)
    intvec = Int[]
    sp = s.parameters
    for i ∈ eachindex(sp)
        spᵢ = sp[i]
        if spᵢ <: Static || spᵢ <: Val
            push!(intvec, spᵢ.parameters[1])
            
        else
            push!(intvec, -1)
            push!(dynamic.args, Expr(:ref, sym, i))
        end
    end
    append!(static.args, intvec)
    static, dynamic
end
@generated function PtrArray(ptr::Ptr{T}, s::Tuple{Vararg{<:Any,N}}, x::Tuple{Vararg{<:Any,N}}, ::Val{org}) where {T,N,org}
    S, sv, _ = sizedefs(s, :s)
    X, xv, _ = sizedefs(x, :x)
    C, B, O, D = org
    Expr(:block, Expr(:meta, :inline), :(PtrArray{$S,$T,$N,$X,$C,$B,$O,$D}(ptr, $sv, $xv)))
end
@generated function PtrArray(ptr::Ptr{T}, s::Tuple{Vararg{<:Any,N}}, x::Tuple{Vararg{<:Any,N}}) where {T,N}
    S, sv, _ = sizedefs(s, :s)
    X, xv, _ = sizedefs(x, :x)
    C, B, O, D = default_org(N)
    Expr(:block, Expr(:meta, :inline), :(PtrArray{$S,$T,$N,$X,$C,$B,$O,$D}(ptr, $sv, $xv)))
end
@generated function PtrArray(ptr::Ptr{T}, s::Tuple{Vararg{<:Any,N}}, ::Val{pad}) where {T,N, pad}
    S, sv, intvec = sizedefs(s, :s)
    C, B, O, D = default_org(N)
    D.args[1] = pad ? :padded : :dense
    any(isequal(-1), intvec) || return Expr(:block, Expr(:meta,:inline), :(PtrArray{$S}(ptr)))
    q, st, xt, X, L = partially_sized(intvec, pad, T)
    push!(q.args, :(PtrArray{$S,$T,$N,$X,$C,$B,$O,$D}(ptr, $st, $xt)))
    q
end
@generated function PtrArray{S}(ptr::Ptr{T}, ::Val{pad}) where {S,T,pad}
    N = length(S)
    q, st, xt, X, L = partially_sized(S, pad, T)
    C, B, O, D = default_org(N)
    D.args[1] = pad ? :padded : :dense
    Expr(:block, Expr(:meta,:inline), :(PtrArray{$S,$T,$N,$X,$C,$B,$O,$D}(ptr, $st, $xt)))
end
@inline PtrArray(ptr::Ptr, s::Tuple) = PtrArray(ptr, s, Val{false}())
@inline PtrArray{S}(ptr::Ptr) where {S} = PtrArray{S}(ptr, Val{false}())

StrideArray{S,T}(::UndefInitializer) where {S,T} = StrideArray{S,T}(undef, Val{false}())
pad_for_alignment(N, ::Type{T}) where {T} = N + VectorizationBase.pick_vector_width(N, T) - 1
unpadded_length(S, ::Type{T}) where {T} = pad_for_alignment(prod(S), T)
align_ptr(ptr::Ptr{T}, L) where {T} = align(ptr, VectorizationBase.pick_vector_width(L, T)*sizeof(T))
function StrideArray{S,T}(::UndefInitializer, ::Val{false}) where {S,T}
    L = unpadded_length(S, T)
    data = Vector{T}(undef, L)
    StrideArray(PtrArray{S}(align_ptr(pointer(data), L), Val{false}()), data)
end
padded_length(S, ::Type{T}) where {T} = pad_for_alignment(calc_padding(first(S), T) * prod(Base.tail(S)), T)
padded_length(S::NTuple{1}, ::Type{T}) where {T} = pad_for_alignment(calc_padding(first(S), T), T)
function StrideArray{S,T}(::UndefInitializer, ::Val{true}) where {S,T}
    data = Vector{T}(undef, padded_length(S, T))
    StrideArray(PtrArray{S}(align(pointer(data)), Val{true}()), data)
end

function partially_sized(sv, pad::Bool, ::Type{T}) where {T}
    st = Expr(:tuple)
    xt = Expr(:tuple)
    xv = Expr(:tuple)
    N = length(sv)
    L = 1
    Lpos = 1
    q = Expr(:block, Expr(:meta,:inline))
    for n ∈ 1:N
        push!(xv.args, L)
        if L == -1
            xn = Symbol(:stride_,n)
            calcstride = Expr(:call, :vmul, Expr(:ref,:s,n-1), Symbol(:stride_, n-1))
            if pad & (n == 2)
                calcstride = Expr(:call, :calc_padding, calcstride, T)
            end
            calcstride = Expr(:call, :vmul, sizeof(T), calcstride)
            push!(q.args, Expr(:(=), xn, calcstride))
            push!(xt.args, xn)
        end
        svₙ = sv[n]
        if svₙ == -1
            if L > 0
                push!(q.args, Expr(:(=), Symbol(:stride_, n), L))
            end
            L = -1
            push!(st.args, Expr(:ref, :s, n))
        else
            if pad & (n == 1)
                svₙ = calc_padding(svₙ, T)
            end
            Lpos *= svₙ
            if L != -1
                L *= svₙ
            end
        end
    end
    Lexpr = if L == -1
        Expr(:call, :*, Symbol(:stride_, N), Expr(:ref, :s, N))
    else
        Lpos
    end
    q, st, xt, xv, Lexpr
end

function StrideArray(A::AbstractArray{T}, ::Val{S}) where {T,S}
    StrideArray{S,T}(undef, size(A)) .= A
end
@generated function negative_one_tuple_val(::Val{N}) where {N}
    t = Expr(:tuple); foreach(_ -> push!(t.args, -1), Base.OneTo(N)); Expr(:call, Expr(:curly, :Val, t))
end
function StrideArray(A::AbstractArray{T,N}) where {T,N}
    StrideArray(A, negative_one_tupletype(Val{N}()))
end

StrideArray{T}(::UndefInitializer, s::Tuple) where {T} = StrideArray{T}(undef, s, Val{false}())
function StrideArray{T}(::UndefInitializer, s::Tuple, ::Val{pad}) where {T, pad}
    A = PtrArray(Base.unsafe_convert(Ptr{T}, C_NULL), s, Val{pad}())
    data = Vector{T}(undef, pad_for_alignment(length(A), T))
    StrideArray(similar(A, align(pointer(data))), data)
end
function StrideArray(data::Vector{T}, s::Tuple, ::Val{pad}) where {T, pad}
    A = PtrArray(pointer(data), s, Val{pad}())
    StrideArray(A, data)
end

function calc_NPL(S, pad::Bool, T)
    nrow = S[1]
    padded_rows = ((!pad) || isone(length(S))) ? nrow : calc_padding(nrow, T)
    N, X, L = calc_NXL(S, padded_rows)
    N, X, pad_for_alignment(L, T)
end
# @generated ValM1(::Val{N}) where {N} = Expr(:call, Expr(:curly, :Val, N - 1))
function calc_NXL(S::NTuple{N}, padded_rows::Int) where {N}
    L = Ref(1)
    X = ntuple(Val{N}()) do n
        sₙ = isone(n) ? padded_rows : S[n]
        Lold = L[]
        L[] *= sₙ
        isone(sₙ) ? 0 : Lold
    end
    N, X, L[]
end

@generated function FixedSizeArray{S,T}(::UndefInitializer, ::Val{pad}) where {S,T,pad}
    N, X, L = calc_NPL(S, pad, T)
    Expr(:block, Expr(:meta,:inline), :(FixedSizeArray{$S,$T,$N,$X,$L}(undef)))
end
@inline FixedSizeArray{S,T}(::UndefInitializer) where {S,T} = FixedSizeArray{S,T}(undef, Val{false}())

function tuplesortperm(O::NTuple{N}) where {N}
    P = ntuple(_ -> 0, Val(N))
    for n ∈ 1:N
        P = Base.setindex(P, n, O[n])
    end
    P
end

tuplerank(X::NTuple{N,Int}) where {N} = ntuple(n -> sum(X[n] .≥ X), Val(N))
@generated function FixedSizeArray{S,T,N,X}(::UndefInitializer) where {S,T,N,X}
    # X may not be monotonic!!!
    # Largest stride corresponds to last dimension.
    # So, find max stride index
    @assert N == length(X) == length(S)

    O = tuplerank(X)
    minind = argmin(X)
    C = isone(X[minind]) ? minind : -1
    B = 1
    maxind = argmax(X)
    D = Expr(:tuple)
    resize!(D.args, 4)
    # Then multiply stride and dimension of that index to get length.
    L = X[maxind] * S[maxind]
    L = pad_for_alignment(L, T)
    :(FixedSizeArray{$S,$T,$N,$X,$C,$B,$O,$D,0,0,$L}(undef))
end




function totupleexpr(x)
    ex = Expr(:tuple); append!(ex.args, x); ex
end

@generated function PtrArray(ptr::PackedStridedPointer{T}, sz::NTuple{N}) where {N,T}
    S = Expr(:tuple); foreach(_ -> push!(S.args, -1), Base.OneTo(N))
    X = Expr(:tuple); foreach(n -> push!(S.args, isone(n) ? n : -1), Base.OneTo(N))
    C, B, O, D = default_org(N)
    :(PtrArray{$S,$T,$N,$X,$N,$(N-1),$C,$B,$O,$D}(ptr.ptr, sz, ptr.strides))
end

@inline function PtrArray(A::AbstractStrideArray{S,T,N,X,C,B,O,D}) where {S,T,N,X,C,B,O,D}
    PtrArray{S,T,N,X,C,B,O,D}(pointer(A), size_tuple(A), stride_tuple(A))
end
function PtrArray(A::AbstractArray)
    PtrArray(stridedpointer(A), size(A))
end


# @inline function NoPadPtrView(θ::Ptr{T}, ::Type{Tuple{L}}) where {L,T}
#     PtrArray{Tuple{L},T,1,Tuple{1},0,0,true,L}(θ, tuple(), tuple())
# end
# @inline function Base.similar(A::PtrArray{S,T,N,X,SN,XN,V}, ptr::Ptr{T}) where {S,T,N,X,SN,XN,V}
#     PtrArray{S,T,N,X,SN,XN,V}(ptr, A.size, A.stride)
# end
# # @inline function NoPadPtrView{Tuple{M,N}}(θ::Ptr{T}) where {M,N,T}
#     # PtrArray{Tuple{M,N},T,1,Tuple{1,M},0,0,true}(θ)
# # end
# @generated function NoPadPtrView(θ::Ptr{T}, ::Type{S}) where {S,T}
#     X = Expr(:curly, :Tuple)
#     SV = S.parameters
#     N = length(SV)
#     quote
#         $(Expr(:meta,:inline))
#         PtrArray{$S,$T,$N,$X,0,0,true}(θ, tuple(), tuple())
#     end
# end
# @inline function NoPadPtrView(sp::StackPointer, ::Type{S}) where {S}
#     A = NoPadPtrView(pointer(sp))
#     sp + align(full_length(A)), A
# end
# @inline function NoPadFlatPtrView(sp::StackPointer, ::Type{S}) where {S}
#     A = NoPadPtrView(S, pointer(sp))
#     sp + align(full_length(A)), flatvector(A)
# end

# @inline function NoPadPtrViewMulti(θ::Ptr{T}, ::Type{Tuple{L}}, ::Val{M}) where {L,T,M}
#     PtrArray{Tuple{L,M},T,2}(θ)
# end
# @generated function NoPadPtrViewMulti(θ::Ptr{T}, ::Type{S}, ::Val{M}) where {S,T,M}
#     X = Expr(:curly, :Tuple)
#     SV = S.parameters
#     L = 1
#     N = length(SV)
#     for n ∈ 1:N
#         push!(X.parameters, L)
#         L *= (SV[n])::Int
#     end
#     N += 1
#     push!(S.parameters, M)
#     push!(X.parameters, L)
#     L *= M
#     quote
#         $(Expr(:meta,:inline))
#         PtrArray{$S,$T,$N,$X,0,0,true,$L}(θ, tuple(), tuple())
#     end
# end
# @inline function NoPadPtrViewMulti(sp::StackPointer, ::Type{S}, ::Val{M}) where {S,M}
#     A = NoPadPtrViewMulti(S, pointer(sp), Val{M}())
#     sp + align(full_length(A)), A
# end
# @inline function NoPadFlatPtrViewMulti(sp::StackPointer, ::Type{Tuple{L}}, ::Val{M}) where {L,M}
#     A = PtrArray{Tuple{L,M},T,2}(pointer(θ))
#     sp + align(full_length(A)), A
# end
# @generated function NoPadFlatPtrViewMulti(sp::StackPointer, ::Type{S}, ::Val{M}) where {S,M}
#     Expr(
#         :block, Expr(:meta,:inline),
#         Expr(
#             :call, :NoPadPtrViewMulti,
#             Expr(:curly, :Tuple, simple_vec_prod(S.parameters)),
#             :sp, Expr(:call, Expr(:curly, :Val, M))
#         )
#     )
# end

allocarray(::Type{T}, s::NTuple{N,Int}) where {T,N} = Array{T}(undef, s)
allocarray(::Type{T}, s::Tuple) where {T} = StrideArray{T}(undef, s)

function Base.similar(::AbstractFixedSizeArray{S,Told,N,X}, ::Type{T}) where {S,T,N,X,Told}
    FixedSizeArray{S,T,N,X}(undef)
end
function Base.similar(A::AbstractStrideArray{S}, ::Type{T}) where {S,T}
    StrideArray{T}(undef, maybestaticsize(A))
end
@generated function size_permute_tuples(A::AbstractStrideArray{S,T,N,X}) where {S,T,N,X}
    sp = sortperm(reinterpret(UInt,tointvec(X)), alg = Base.Sort.DEFAULT_STABLE)
    Si = tointvec(S)
    Sv = axis_expressions(Si, :(size_tuple(A)))
    Svp = Sv[sp]
    perm = Vector{Int}(undef, N)
    for n in 1:N
        perm[sp[n]] = n
    end
    Expr(:tuple, totupleexpr(Svp), Expr(:call, Expr(:curly, :Static, totupleexpr(perm))))
end
@inline similarlypermuted(A::AbstractStrideArray{S,T}) where {S,T} = similarlypermuted(A, T)
@inline function similarlypermuted(A::AbstractStrideArray{S,Told,N,X,0}, ::Type{T}) where {S,Told,T,N,X}
    s, p = size_permute_tuples(A)
    PermutedDimsArray(FixedSizeArray{T}(undef, s), p)
end
@inline function similarlypermuted(A::AbstractStrideArray, ::Type{T}) where {T}
    s, p = size_permute_tuples(A)
    PermutedDimsArray(StrideArray{T}(undef, s), p)
end
@inline function (sp::StackPointer)(::typeof(similarlypermuted), A::AbstractStrideArray, ::Type{T}) where {T}
    s, p = size_permute_tuples(A)
    sp, B = PtrArray{T}(sp, s)
    sp, PermutedDimsArray(B, p)
end

@inline (sp::StackPointer)(::typeof(similarlypermuted), A) = similarlypermuted(sp, A)
@inline (sp::StackPointer)(::typeof(similarlypermuted), A::AbstractStrideArray{<:Any,T}, ::Type{T}) where {T} = similarlypermuted(sp, A, T)



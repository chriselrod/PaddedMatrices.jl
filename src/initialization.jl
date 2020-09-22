
function ctuple(n)
    Expr(:curly, :Tuple, n...) 
end
function calc_padding(nrow::Int, T)
    W = VectorizationBase.pick_vector_width(T)
    W > nrow ? VectorizationBase.nextpow2(nrow) : VectorizationBase.align(nrow, T)
end

axis_expressions(x::Core.SimpleVector, s) = axis_expressions(x.parameters, s)
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

@inline new_view(A::PtrArray{S,T,N,X,SN,XN,V}, sn, xn) where {S,T,N,X,SN,XN,V} = PtrArray(A.ptr, sn, xn, Val{V}())
@inline new_view(A::StrideArray, sn, xn) = StrideArray(new_view(A.ptr, sn, xn), A.data)
@inline new_view(A::FixedSizeArray, sn, xn) = FixedSizeArray(new_view(A.ptr, sn, xn), A.data)

@generated function StrideArray{S,T}(::UndefInitializer, ::Val{pad} = Val{false}()) where {S,T,pad}
    sv = tointvec(S)
    N = length(sv)
    firstsv = first(sv)::Int
    L = pad ? calc_padding(firstsv, T) : firstsv
    xv = similar(sv)
    xv[1] = 1
    for n ∈ 2:N
        svₙ = sv[n]
        xv[n] = L
        L *= svₙ
    end
    W = VectorizationBase.pick_vector_width(T)
    quote
        parent = Vector{$T}(undef, $L + $(W-1))
        StrideArray{$S,$T,$N,$(ctuple(xv)),0,0,false}(align(pointer(parent)), tuple(), tuple(), parent)
    end
end

function partially_sized(sv, pad::Bool, ::Type{T}) where {T}
    st = Expr(:tuple)
    xt = Expr(:tuple)
    xv = similar(sv)
    N = length(sv)
    L = 1
    Lpos = 1
    lastsize_known::Bool = false
    q = Expr(:block, Expr(:meta,:inline))
    for n ∈ 1:N
        xv[n] = L
        if L == -1
            xn = Symbol(:stride_,n)
            calcstride = if lastsize_known
                Expr(:call, :vmul, sv[n-1], Symbol(:stride_, n-1))
            else
                Expr(:call, :vmul, Symbol(:size_,n-1), Symbol(:stride_, n-1))
            end
            if pad & (n == 2)
                calcstride = Expr(:call, :calc_padding, calcstride, T)
            end
            calcstride = Expr(:call, :vmul, sizeof(T), calcstride)
            push!(q.args, Expr(:(=), xn, calcstride))
            push!(xt.args, xn)
        end
        svₙ = sv[n]
        lastsize_known = svₙ != -1
        if lastsize_known
            if pad & (n == 1)
                svₙ = calc_padding(svₙ, T)
            end
            Lpos *= svₙ
            if L != -1
                L *= svₙ
            end
        else
            if L > 0
                push!(q.args, Expr(:(=), Symbol(:stride_, n), L))
            end
            L = -1
            sizesym = Symbol(:size_, n)
            push!(q.args, Expr(:(=), sizesym, Expr(:ref, :s, n)))
            push!(st.args, sizesym)
        end
    end
    Lexpr = if L == -1
        Expr(:call, :*, Symbol(:stride_, N), Expr(:ref, :s, N))
    else
        Lpos
    end
    q, st, xt, xv, Lexpr
end

@generated function StrideArray{S,T}(
# function StrideArray{S,T}(
    ::UndefInitializer, s::NTuple{N,<:Integer}, ::Val{pad}
) where {S,T,N,pad}
    sv = tointvec(S)
    @assert N == length(sv)
    any(isequal(-1), sv) || return Expr(:block, Expr(:meta,:inline), :(StrideArray{$S,$T}(::UndefInitializer)))
    q, st, xt, xv, L = partially_sized(sv, pad, T)
    SN = length(st.args); XN = length(xt.args)
    # W = VectorizationBase.pick_vector_width(T)
    push!(q.args, :(parent = Vector{$T}(undef, $L + $(pick_vector_width(T) - 1))))
    push!(q.args, :(StrideArray{$S,$T,$N,$(ctuple(xv)),$SN,$XN,false}(align(pointer(parent)), $st, $xt, parent)))
    q
end
@inline StrideArray{S,T}(::UndefInitializer, s::NTuple{N,<:Integer}) where {S,T,N} = StrideArray{S,T}(undef, s, Val{false}())
function StrideArray(A::AbstractArray{T}, ::Type{S}) where {T,S<:Tuple}
    StrideArray{S,T}(undef, size(A)) .= A
end
@generated function negative_one_tupletype(::Val{N}) where {N}
    Expr(:curly, :Tuple, (-1 for _ ∈ 1:N)...)
end
function StrideArray(A::AbstractArray{T,N}) where {T,N}
    StrideArray(A, negative_one_tupletype(Val{N}()))
end

function tointvecdt(ssv)
    N = length(ssv)::Int
    sv = Vector{Int}(undef, N)
    for n ∈ 1:N
        ssvₙ = ssv[n]
        sv[n] = ssvₙ <: Static ? VectorizationBase.unwrap(ssvₙ) : -1
    end
    sv
end
@generated function StrideArray{T}(::UndefInitializer, s::Tuple, ::Val{pad}) where {T,pad}
    sv = tointvecdt(s.parameters)
    S = Tuple{sv...}
    N = length(sv)
    any(s -> s == -1, sv) || return Expr(:block, Expr(:meta,:inline), :(StrideArray{$S,$T}(undef)))
    q, st, xt, xv, L = partially_sized(sv, pad, T)
    SN = length(st.args); XN = length(xt.args)
    push!(q.args, :(parent = Vector{$T}(undef, $L + $(VectorizationBase.pick_vector_width(T) - 1))))
    push!(q.args, :(StrideArray{$S,$T,$N,$(ctuple(xv)),$SN,$XN,false}(align(pointer(parent)), $st, $xt, parent)))
    q    
end
@inline StrideArray{T}(::UndefInitializer, s::Tuple) where {T} = StrideArray{T}(undef, s, Val{false}())
@inline StrideMatrix{T}(::UndefInitializer, s::Tuple{Vararg{<:Any,2}}) where {T} = StrideArray{T}(undef, s, Val{false}())

@generated function StrideArray(parent::Vector{T}, s::Tuple, ::Val{pad}) where {T,pad}
    sv = tointvecdt(s.parameters)
    N = length(sv)
    S = Expr(:curly, :Tuple); foreach(s -> push!(S.args,s), sv)
    if !any(s -> s == -1, sv)
        X = Expr(:curly, :Tuple, 1)
        x = 1
        for s in @view(sv[1:end-1])
            push!(X.args, (x *= s))
        end
        # Expr(:block, Expr(:meta,:inline), :(StrideArray{$S,$T,$N,$X,0,0,false}(align(pointer(parent)), (), (), parent)))
        Expr(:block, Expr(:meta,:inline), :(out = StrideArray{$S,$T,$N,$X,0,0,false}(align(pointer(parent)), (), (), parent)), :(@assert length(out) ≥ length(parent)), :out)
    end
    q, st, xt, xv, L = partially_sized(sv, pad, T)
    SN = length(st.args); XN = length(xt.args)
    # push!(q.args, :(parent = Vector{$T}(undef, $L)))
    push!(q.args, :(StrideArray{$S,$T,$N,$(ctuple(xv)),$SN,$XN,false}(align(pointer(parent)), $st, $xt, parent)))
    q    
end
@inline StrideArray(data::Vector{T}, s::Tuple) where {T} = StrideArray(data, s, Val{false}())
@inline StrideMatrix(data::Vector{T}, s::Tuple{Vararg{<:Any,2}}) where {T} = StrideArray(data, s, Val{false}())


function calc_NPL(SV::Core.SimpleVector, pad::Bool, T)
    nrow = (SV[1])::Int
    padded_rows = ((!pad) || isone(length(SV))) ? nrow : calc_padding(nrow, T)
    calc_NXL(SV, T, padded_rows)
end
function calc_NXL(SV::Core.SimpleVector, T, padded_rows::Int)
    L = padded_rows
    N = length(SV)
    X = Int[ (SV[1])::Int == 1 ? 0 : 1 ]
    for n ∈ 2:N
        svn = (SV[n])::Int
        push!(X, svn == 1 ? 0 : L)
        L *= svn
    end
    # LA = VectorizationBase.align(L,T)
    N, Tuple{X...}, L#A
end

function maybeincreaseL(L::Int, ::Type{T}) where {T}
    # Wm1 = VectorizationBase.pick_vector_width(L, T) - 1
    Wm1 = VectorizationBase.pick_vector_width(T) - 1
    return L + Wm1
    # if ALIGN_ALL_FS_ARRAYS || (L + Wm1) * sizeof(T) < 512
    #     L + Wm1
    # elseif L * sizeof(T) ≥ 512
    #     L
    # else
    #     512 ÷ sizeof(T)
    # end
end
@generated function FixedSizeArray{S,T}(::UndefInitializer, ::Val{pad}) where {S,T,pad}
    N, X, L = calc_NPL(S.parameters, pad, T)
    L = maybeincreaseL(L, T)
    Expr(:block, Expr(:meta,:inline), :(FixedSizeArray{$S,$T,$N,$X,$L}(undef)))
end
@inline FixedSizeArray{S,T}(::UndefInitializer) where {S,T} = FixedSizeArray{S,T}(undef, Val{false}())
@generated function FixedSizeArray{S,T,N,X}(::UndefInitializer) where {S,T,N,X}
    # X may not be monotonic!!!
    # Largest stride corresponds to last dimension.
    # So, find max stride index
    Xv = X.parameters
    @assert N == length(X.parameters) == length(S.parameters)
    Xint = Vector{Int}(undef, N)
    for n in 1:N
        Xint[n] = (Xv[n])::Int
    end
    maxind = argmax(Xint)
    # Then multiply stride and dimension of that index to get length.
    L = Xint[maxind] * (S.parameters[maxind])::Int
    L = maybeincreaseL(L, T)
    :(FixedSizeArray{$S,$T,$N,$X,$L}(undef))
end
@generated function FixedSizeVector{M,T}(::UndefInitializer) where {M,T}
    L = maybeincreaseL(M, T)
    :(FixedSizeArray{Tuple{$M},$T,1,Tuple{1},$L}(undef))
end
@generated function FixedSizeMatrix{M,N,T}(::UndefInitializer, ::Val{pad}) where {M,N,T,pad}
    X = pad ? calc_padding(M, T) : M
    L = maybeincreaseL(X*N, T)
    Expr(:block, Expr(:meta,:inline), :(FixedSizeArray{Tuple{$M,$N},$T,2,Tuple{1,$X},$L}(undef)))
end
@inline FixedSizeMatrix{M,N,T}(::UndefInitializer) where {M,N,T} = FixedSizeMatrix{M,N,T}(undef, Val{false}())
@generated function FixedSizeMatrix{M,N,T,X}(::UndefInitializer) where {M,N,T,X}
    @assert X ≥ M
    L = maybeincreaseL(X * N, T)
    :(FixedSizeArray{Tuple{$M,$N},$T,2,Tuple{1,$X},$L}(undef))
end

@generated function isfixed(::Type{T}) where {T<:Tuple}
    for i ∈ eachindex(T.parameters)
        T.parameters[i] === -1 && return false
    end
    true
end

@generated function PtrArray{S}(ptr::Ptr{T}, ::Val{pad}) where {S,T,pad}
    N, X, L = calc_NPL(S.parameters, T)
    Expr(:block, Expr(:meta,:inline), :(PtrArray{$S,$T,$N,$X,0,0,false}(ptr)))
end
@inline PtrArray{S}(ptr::Ptr{T}) where {S,T} = PtrArray{S}(ptr, Val{false}())
@generated function PtrArray{S}(ptr::Ptr{T}, s::NTuple{N,<:Integer}, ::Val{pad}) where {S,T,N,pad}
    sv = tointvec(S)
    @assert N == length(sv)
    any(isequal(-1), sv) || return Expr(:block, Expr(:meta,:inline), :(PtrArray{$S}(ptr)))
    q, st, xt, xv, L = partially_sized(sv, pad, T)
    SN = length(st.args); XN = length(xt.args)
    push!(q.args, :(PtrArray{$S,$T,$N,$(ctuple(xv)),$SN,$XN,false}(ptr, $st, $xt)))
    q
end
@inline PtrArray{S}(ptr::Ptr{T}, s::NTuple{N,<:Integer}) where {S,T,N} = PtrArray{S}(ptr, s, Val{false}())
function toctuple(s)
    sv = Int[]
    sp = s.parameters
    for i ∈ eachindex(sp)
        if sp[i] <: Static
            push!(sv, sp[i].parameters[1])
        else
            push!(sv, -1)
        end
    end
    S = Expr(:curly, :Tuple)
    append!(S.args, sv)
    S, sv
end
@inline PtrArray(ptr::Ptr{T}, s::Tuple{Vararg{<:Any,N}}, x::Tuple{Vararg{<:Any,N}}) where {T,N} = PtrArray(ptr, s, x, Val{false}())
function totupleexpr(x)
    ex = Expr(:tuple); append!(ex.args, x); ex
end
function sizedefs(s, sym)
    cT = Expr(:curly, :Tuple)
    t = Expr(:tuple)
    sp = s.parameters
    for i ∈ eachindex(sp)
        if sp[i] <: Static
            push!(cT.args, sp[i].parameters[1])
        else
            push!(cT.args, -1)
            push!(t.args, Expr(:ref, sym, i))
        end
    end
    cT, t
end
@generated function PtrArray(ptr::Ptr{T}, s::Tuple{Vararg{<:Any,N}}, x::Tuple{Vararg{<:Any,N}}, ::Val{V}) where {T,N,V}
    S, sv = sizedefs(s, :s)
    X, xv = sizedefs(x, :x)
    SN = length(sv.args); XN = length(xv.args)
    Expr(:block, Expr(:meta, :inline), :(PtrArray{$S,$T,$N,$X,$SN,$XN,$V}(ptr, $sv, $xv)))
end
@generated function PtrArray(ptr::Ptr{T}, s::Tuple{Vararg{<:Any,N}}, ::Val{pad}) where {T,N, pad}
    S, sv = toctuple(s)
    any(isequal(-1), sv) || return Expr(:block, Expr(:meta,:inline), :(PtrArray{$S}(ptr)))
    q, st, xt, xv, L = partially_sized(sv, pad, T)
    SN = length(st.args); XN = length(xt.args)
    push!(q.args, :(PtrArray{$S,$T,$N,$(ctuple(xv)),$SN,$XN,false}(ptr, $st, $xt)))
    q
end
@inline PtrArray(ptr::Ptr{T}, s::Tuple{Vararg{<:Any,N}}) where {T,N} = PtrArray(ptr, s, Val{false}())
@inline PtrArray{S,T,N}(ptr::Ptr{T}) where {S,T,N} = PtrArray{S}(ptr)
# @inline PtrArray{S,T,N,X,SN,XN}(ptr) where {S,T,N,X,SN,XN} = PtrArray{S,T,X,0,0,true}(ptr)
# @inline PtrArray{S,T,N,X,SN,XN,V}(ptr) where {S,T,N,X,SN,XN,V} = PtrArray{S,T,X,0,0,V}(ptr)
@inline PtrArray{S,T,N,X}(ptr, st::NTuple{SN,Int}, xt::NTuple{XN,Int}) where {S,T,N,X,SN,XN} = PtrArray{S,T,N,X,SN,XN,false}(ptr, st, xt)
@inline FixedSizeArray{S,T,N,X}(ptr, st::NTuple{SN,Int}, xt::NTuple{XN,Int}, d::Base.RefValue{NTuple{L,T}}) where {S,T,N,X,L,SN,XN} = FixedSizeArray{S,T,N,X,L,SN,XN,false}(ptr, st, xt,d)
@inline StrideArray{S,T,N,X}(ptr, st::NTuple{SN,Int}, xt::NTuple{XN,Int},data) where {S,T,N,X,SN,XN} = StrideArray{S,T,N,X,SN,XN,false}(ptr, st, xt, data)
@inline PtrArray{S,T,N,X}(ptr) where {S,T,N,X} = PtrArray{S,T,N,X,0,0,true}(ptr, tuple(), tuple())
@inline PtrArray{S,T,N,X,0}(ptr) where {S,T,N,X} = PtrArray{S,T,N,X,0,0,true}(ptr, tuple(), tuple())
@inline PtrArray{S,T,N,X,0,0}(ptr) where {S,T,N,X} = PtrArray{S,T,N,X,0,0,true}(ptr, tuple(), tuple())
# @generated PtrArray{S,T,N,X,0,0,V}(ptr) where {S,T,N,X,V} = Expr(:block, Expr(:meta,:inline), :(PtrArray{$S,$T,$X,0,0,$V}(ptr, tuple(), tuple())))
# @inline PtrArray{S,T,N,X,0,0}(ptr, ::NTuple{0}, ::NTuple{0}) where {S,T,N,X} = PtrArray{S,T,X,0,0,true}(ptr, tuple(), tuple())
# @generated PtrArray{S,T,N,X,0,0,V}(ptr, ::NTuple{0}, ::NTuple{0}) where {S,T,N,X,V} = Expr(:block, Expr(:meta,:inline), :(PtrArray{$S,$T,$X,0,0,$V,$(last(S.parameters)::Int*last(X.parameters)::Int)}(ptr, tuple(), tuple())))
# @generated PtrArray{S,T,N,X}(ptr, ::NTuple{0}, ::NTuple{0}) where {S,T,N,X} = Expr(:block, Expr(:meta,:inline), :(PtrArray{$S,$T,$X,0,0,true,$(last(S.parameters)::Int*last(X.parameters)::Int)}(ptr, tuple(), tuple())))
# @generated PtrArray{S,T,N,X,V}(ptr, ::NTuple{0}, ::NTuple{0}) where {S,T,N,X,V} = Expr(:block, Expr(:meta,:inline), :(PtrArray{$S,$T,$X,0,0,$V,$(last(S.parameters)::Int*last(X.parameters)::Int)}(ptr, tuple(), tuple())))
@generated function PtrArray(ptr::PackedStridedPointer{T}, sz::NTuple{N}) where {N,T}
    S = Tuple{(-1 for _ ∈ 1:N)...}
    X = Tuple{1,(-1 for _ ∈ 2:N)...}
    :(PtrArray{$S,$T,$N,$X,$N,$(N-1),true}(ptr.ptr, sz, ptr.strides))
end

@inline function PtrArray(A::AbstractStrideArray{S,T,N,X,SN,XN,V}) where {S,T,N,X,SN,XN,V}
    PtrArray{S,T,N,X,SN,XN,V}(pointer(A), size_tuple(A), stride_tuple(A))
end
@inline function PtrArray(A::AbstractFixedSizeArray{S,T,N,X,V}) where {S,T,N,X,V}
    PtrArray{S,T,N,X,0,0,V}(pointer(A), tuple(), tuple())
end
function PtrArray(A::AbstractArray)
    PtrArray(stridedpointer(A), size(A))
end
PtrVector{M}(ptr::Ptr{T}) where {M,T} = PtrVector{M,T,1,0,0,false}(ptr, tuple(), tuple())
function PtrMatrix{M,N}(A::PackedStridedPointer{T,1}) where {M, N, T}
    PtrArray{Tuple{M,N},T,2,Tuple{1,-1},0,1,false}(pointer(A), tuple(), A.strides)
end
function PtrMatrix{-1,N}(A::PackedStridedPointer{T,1}, nrows::Integer) where {N, T}
    PtrArray{Tuple{-1, N}, T, 2, Tuple{1,-1}, 1, 1, false}(pointer(A), (nrows,), A.strides)
end
function PtrMatrix{-1,N}(A::PackedStridedPointer{T,1}, ::Static{M}) where {M, N, T}
    PtrArray{Tuple{M, N}, T, 2, Tuple{1,-1}, 0, 1, false}(pointer(A), tuple(), A.strides)
end
function PtrMatrix{M,-1}(A::PackedStridedPointer{T,1}, ncols::Integer) where {M, T}
    PtrArray{Tuple{M, -1}, T, 2, Tuple{1,-1}, 1, 1, false}(pointer(A), (ncols,), A.strides)
end
function PtrMatrix{M,-1}(A::PackedStridedPointer{T,1}, ::Static{N}) where {M, N, T}
    PtrArray{Tuple{M, N}, T, 2, Tuple{1,-1}, 0, 1, false}(pointer(A), tuple(), A.strides)
end
function PtrMatrix(A::PackedStridedPointer{T,1}, nrows::Integer, ncols::Integer) where {T}
    PtrArray{Tuple{-1, -1}, T, 2, Tuple{1,-1}, 2, 1, false}(pointer(A), (nrows,ncols), A.strides)
end
function PtrMatrix(A::PackedStridedPointer{T,1}, ::Static{M}, ncols::Integer) where {T, M}
    PtrArray{Tuple{M, -1}, T, 2, Tuple{1,-1}, 1, 1, false}(pointer(A), (ncols,), A.strides)
end
function PtrMatrix(A::PackedStridedPointer{T,1}, nrows::Integer, ::Static{N}) where {T, N}
    PtrArray{Tuple{-1, N}, T, 2, Tuple{1,-1}, 1, 1, false}(pointer(A), (nrows,), A.strides)
end
function PtrMatrix(A::PackedStridedPointer{T,1}, ::Static{M}, ::Static{N}) where {T, M, N}
    PtrArray{Tuple{M,N}, T, 2, Tuple{1,-1}, 0, 1, false}(pointer(A), tuple(), A.strides)
end
function PtrArray{Tuple{-1,M,-1},T,3}(A::PackedStridedPointer{T,1}, sz::Tuple{Int,Int}) where {M,T}
    Ax = first(A.strides)
    PtrArray{Tuple{-1,M,-1}, T, 3, Tuple{1,-1,-1}, 2, 2, false}(pointer(A), sz, (Ax,Ax*M))
end
function PtrArray{Tuple{M,-1,-1},T,3}(A::PackedStridedPointer{T,1}, sz::Tuple{Int,Int}) where {M,T}
    PtrArray{Tuple{M,-1,-1}, T, 3, Tuple{1,M,-1}, 2, 1, false}(pointer(A), sz, A.strides)
end

function PtrMatrix{M,N}(A::RowMajorStridedPointer{T,1}) where {M, N, T}
    PtrArray{Tuple{M,N},T,2,Tuple{-1,1},0,1,false}(pointer(A), tuple(), A.strides)
end
function PtrMatrix{-1,N}(A::RowMajorStridedPointer{T,1}, nrows::Integer) where {N, T}
    PtrArray{Tuple{-1, N}, T, 2, Tuple{-1,1}, 1, 1, false}(pointer(A), (nrows,), A.strides)
end
function PtrMatrix{-1,N}(A::RowMajorStridedPointer{T,1}, ::Static{M}) where {M, N, T}
    PtrArray{Tuple{M, N}, T, 2, Tuple{-1,1}, 0, 1, false}(pointer(A), tuple(), A.strides)
end
function PtrMatrix{M,-1}(A::RowMajorStridedPointer{T,1}, ncols::Integer) where {M, T}
    PtrArray{Tuple{M, -1}, T, 2, Tuple{-1,1,}, 1, 1, false}(pointer(A), (ncols,), A.strides)
end
function PtrMatrix{M,-1}(A::RowMajorStridedPointer{T,1}, ::Static{N}) where {M, N, T}
    PtrArray{Tuple{M, N}, T, 2, Tuple{-1,1}, 0, 1, false}(pointer(A), tuple(), A.strides)
end
function PtrMatrix(A::RowMajorStridedPointer{T,1}, nrows::Integer, ncols::Integer) where {T}
    PtrArray{Tuple{-1, -1}, T, 2, Tuple{-1,1}, 2, 1, false}(pointer(A), (nrows,ncols), A.strides)
end
function PtrArray(A::RowMajorStridedPointer{T,1}, (nrows, ncols)) where {T}
    PtrArray{Tuple{-1, -1}, T, 2, Tuple{-1,1}, 2, 1, false}(pointer(A), (nrows,ncols), A.strides)
end
function PtrMatrix(A::RowMajorStridedPointer{T,1}, ::Static{M}, ncols::Integer) where {T, M}
    PtrArray{Tuple{M, -1}, T, 2, Tuple{-1,1}, 1, 1, false}(pointer(A), (ncols,), A.strides)
end
function PtrMatrix(A::RowMajorStridedPointer{T,1}, nrows::Integer, ::Static{N}) where {T, N}
    PtrArray{Tuple{-1, N}, T, 2, Tuple{-1,1}, 1, 1, false}(pointer(A), (nrows,), A.strides)
end
function PtrMatrix(A::RowMajorStridedPointer{T,1}, ::Static{M}, ::Static{N}) where {T, M, N}
    PtrArray{Tuple{M,N}, T, 2, Tuple{-1,1}, 0, 1, false}(pointer(A), tuple(), A.strides)
end
function PtrArray{Tuple{-1,M,-1},T,3}(A::RowMajorStridedPointer{T,1}, sz::Tuple{Int,Int}) where {M,T}
    PtrArray{Tuple{-1,M,-1}, T, 3, Tuple{-1,1,M}, 2, 1, false}(pointer(A), sz, A.strides)
end
function PtrArray{Tuple{M,-1,-1},T,3}(A::RowMajorStridedPointer{T,1}, sz::Tuple{Int,Int}) where {M,T}
    Ax = A.strides[1]; #xl = last(sz)
    PtrArray{Tuple{M,-1,-1}, T, 3, Tuple{-1,-1,1}, 2, 2, false}(pointer(A), sz, (Ax, Ax*M))
end


function PtrMatrix{M,N}(A::StaticStridedPointer{T,X}) where {M, N, T, X}
    PtrArray{Tuple{M,N},T,2,X,0,0,false}(pointer(A), tuple(), tuple())
end
function PtrMatrix{-1,N}(A::StaticStridedPointer{T,X}, nrows::Integer) where {N, T, X}
    PtrArray{Tuple{-1, N}, T, 2, X, 1, 0, false}(pointer(A), (nrows,), tuple())
end
function PtrMatrix{-1,N}(A::StaticStridedPointer{T,X}, ::Static{M}) where {M, N, T, X}
    PtrArray{Tuple{M, N}, T, 2, X, 0, 0, false}(pointer(A), tuple(), tuple())
end
function PtrMatrix{M,-1}(A::StaticStridedPointer{T,X}, ncols::Integer) where {M, T, X}
    PtrArray{Tuple{M, -1}, T, 2, X, 1, 0, false}(pointer(A), (ncols,), tuple())
end
function PtrMatrix{M,-1}(A::StaticStridedPointer{T,X}, ::Static{N}) where {M, N, T, X}
    PtrArray{Tuple{M, N}, T, 2, X, 0, 0, false}(pointer(A), tuple(), tuple())
end
function PtrMatrix(A::StaticStridedPointer{T,X}, nrows::Integer, ncols::Integer) where {T, X}
    PtrArray{Tuple{-1, -1}, T, 2, X, 2, 0, false}(pointer(A), (nrows,ncols), tuple())
end
function PtrMatrix(A::StaticStridedPointer{T,X}, ::Static{M}, ncols::Integer) where {T, M, X}
    PtrArray{Tuple{M, -1}, T, 2, X, 1, 0, false}(pointer(A), (ncols,), tuple())
end
function PtrMatrix(A::StaticStridedPointer{T,X}, nrows::Integer, ::Static{N}) where {T, N, X}
    PtrArray{Tuple{-1, N}, T, 2, X, 1, 0, false}(pointer(A), (nrows,), tuple())
end
function PtrMatrix(A::StaticStridedPointer{T,X}, ::Static{M}, ::Static{N}) where {T, M, N, X}
    PtrArray{Tuple{M,N}, T, 2, X, 0, 0, false}(pointer(A), tuple(), tuple())
end
                     
function PtrMatrix{M,N,Tb,X}(A::Ptr{Ta}) where {M, N, Ta, Tb, X}
    PtrArray{Tuple{M, N}, Tb, 2, Tuple{1,X}, 0, 0, false}(Base.unsafe_convert(Ptr{Tb}, A), tuple(), tuple())
end
function PtrMatrix{-1,N,Tb,X}(A::Ptr{Ta}, nrows::Integer) where {N, Ta, Tb, X}
    PtrArray{Tuple{-1, N}, Tb, 2, Tuple{1,X}, 1, 0, false}(Base.unsafe_convert(Ptr{Tb},A), (nrows,), tuple())
end
function PtrMatrix{-1,N,Tb,X}(A::Ptr{Ta}, ::Static{M}) where {M, N, Ta, Tb, X}
    PtrArray{Tuple{M, N}, Tb, 2, Tuple{1,X}, 0, 0, false}(Base.unsafe_convert(Ptr{Tb},A), tuple(), tuple())
end
function PtrMatrix{M,-1,Tb,X}(A::Ptr{Ta}, ncols::Integer) where {M, Ta, Tb, X}
    PtrArray{Tuple{M, -1}, Tb, 2, Tuple{1,X}, 1, 0, false}(Base.unsafe_convert(Ptr{Tb},A), (ncols,), tuple())
end
function PtrMatrix{M,-1,Tb,X}(A::Ptr{Ta}, ::Static{N}) where {M, N, Ta, Tb, X}
    PtrArray{Tuple{M, N}, Tb, 2, Tuple{1,X}, 0, 0, false}(Base.unsafe_convert(Ptr{Tb},A), tuple(), tuple())
end
function PtrMatrix{-1,-1,Tb,X}(A::Ptr{Ta}, nrows::Integer, ncols::Integer) where {Ta, Tb, X}
    PtrArray{Tuple{-1, -1}, Tb, 2, Tuple{1,X}, 2, 0, false}(Base.unsafe_convert(Ptr{Tb},A), (nrows,ncols), tuple())
end
function PtrMatrix{-1,-1,Tb,X}(A::Ptr{Ta}, nrows::Integer, ::Static{N}) where {N, Ta, Tb, X}
    PtrArray{Tuple{-1, N}, Tb, 2, Tuple{1,X}, 1, 0, false}(Base.unsafe_convert(Ptr{Tb},A), (nrows,), tuple())
end
function PtrMatrix{-1,-1,Tb,X}(A::Ptr{Ta}, ::Static{M}, ncols::Integer) where {M, Ta, Tb, X}
    PtrArray{Tuple{M, -1}, Tb, 2, Tuple{1,X}, 1, 0, false}(Base.unsafe_convert(Ptr{Tb},A), (ncols,), tuple())
end
function PtrMatrix{-1,-1,Tb,X}(A::Ptr{Ta}, ::Static{M}, ::Static{N}) where {M, N, Ta, Tb, X}
    PtrArray{Tuple{M, M}, Tb, 2, Tuple{1,X}, 0, 0, false}(Base.unsafe_convert(Ptr{Tb},A), tuple(), tuple())
end
function PtrMatrix{M,N}(A::Ptr{T}) where {M, N, T}
    PtrArray{Tuple{M, N}, T, 2, Tuple{1,M}, 0, 0, false}(A, tuple(), tuple())
end
function PtrMatrix{M,N,X}(A::Ptr{T}) where {M, N, T, X}
    PtrArray{Tuple{M, N}, T, 2, Tuple{1,X}, 0, 0, false}(A, tuple(), tuple())
end
function PtrMatrix{-1,N,X}(A::Ptr{T}, nrows::Integer) where {N, T, X}
    PtrArray{Tuple{-1, N}, T, 2, Tuple{1,X}, 1, 0, false}(A, (nrows,), tuple())
end
function PtrMatrix{-1,N,X}(A::Ptr{T}, ::Static{M}) where {M, N, T, X}
    PtrArray{Tuple{M, N}, T, 2, Tuple{1,X}, 0, 0, false}(A, tuple(), tuple())
end
function PtrMatrix{M,-1,X}(A::Ptr{T}, ncols::Integer) where {M, T, X}
    PtrArray{Tuple{M, -1}, T, 2, Tuple{1,X}, 1, 0, false}(A, (ncols,), tuple())
end
function PtrMatrix{M,-1,X}(A::Ptr{T}, ::Static{N}) where {M, N, T, X}
    PtrArray{Tuple{M, N}, T, 2, Tuple{1,X}, 0, 0, false}(A, tuple(), tuple())
end
function PtrMatrix{X}(A::Ptr{T}, nrows::Integer, ncols::Integer) where {T, X}
    PtrArray{Tuple{-1, -1}, T, 2, Tuple{1,X}, 2, 0, false}(A, (nrows,ncols), tuple())
end
function PtrMatrix{X}(A::Ptr{T}, nrows::Integer, ::Static{N}) where {N, T, X}
    PtrArray{Tuple{-1, N}, T, 2, Tuple{1,X}, 1, 0, false}(A, (nrows,), tuple())
end
function PtrMatrix{X}(A::Ptr{T}, ::Static{M}, ncols::Integer) where {M, T, X}
    PtrArray{Tuple{M, -1}, T, 2, Tuple{1,X}, 1, 0, false}(A, (ncols,), tuple())
end
function PtrMatrix{X}(A::Ptr{T}, ::Static{M}, ::Static{N}) where {M, N, T, X}
    PtrArray{Tuple{M, N}, T, 2, Tuple{1,X}, 0, 0, false}(A, tuple(), tuple())
end
function PtrMatrix(A::Ptr{T}, nrows::Integer, ncols::Integer, X) where {T}
    PtrArray{Tuple{-1, -1}, T, 2, Tuple{1,-1}, 2, 1, false}(A, (nrows,ncols), (X,))
end
function PtrMatrix(A::Ptr{T}, nrows::Integer, ::Static{N}, X) where {N, T}
    PtrArray{Tuple{-1, N}, T, 2, Tuple{1,-1}, 1, 1, false}(A, (nrows,), (X,))
end
function PtrMatrix(A::Ptr{T}, ::Static{M}, ncols::Integer, X) where {M, T}
    PtrArray{Tuple{M, -1}, T, 2, Tuple{1,-1}, 1, 1, false}(A, (ncols,), (X,))
end
function PtrMatrix(A::Ptr{T}, ::Static{M}, ::Static{N}, X) where {M, N, T}
    PtrArray{Tuple{M, N}, T, 2, Tuple{1,-1}, 0, 1, false}(A, tuple(), (X,))
end



@inline function NoPadPtrView(θ::Ptr{T}, ::Type{Tuple{L}}) where {L,T}
    PtrArray{Tuple{L},T,1,Tuple{1},0,0,true,L}(θ, tuple(), tuple())
end
@inline function Base.similar(A::PtrArray{S,T,N,X,SN,XN,V}, ptr::Ptr{T}) where {S,T,N,X,SN,XN,V}
    PtrArray{S,T,N,X,SN,XN,V}(ptr, A.size, A.stride)
end
# @inline function NoPadPtrView{Tuple{M,N}}(θ::Ptr{T}) where {M,N,T}
    # PtrArray{Tuple{M,N},T,1,Tuple{1,M},0,0,true}(θ)
# end
@generated function NoPadPtrView(θ::Ptr{T}, ::Type{S}) where {S,T}
    X = Expr(:curly, :Tuple)
    SV = S.parameters
    N = length(SV)
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$X,0,0,true}(θ, tuple(), tuple())
    end
end
@inline function NoPadPtrView(sp::StackPointer, ::Type{S}) where {S}
    A = NoPadPtrView(pointer(sp))
    sp + align(full_length(A)), A
end
@inline function NoPadFlatPtrView(sp::StackPointer, ::Type{S}) where {S}
    A = NoPadPtrView(S, pointer(sp))
    sp + align(full_length(A)), flatvector(A)
end

@inline function NoPadPtrViewMulti(θ::Ptr{T}, ::Type{Tuple{L}}, ::Val{M}) where {L,T,M}
    PtrArray{Tuple{L,M},T,2}(θ)
end
@generated function NoPadPtrViewMulti(θ::Ptr{T}, ::Type{S}, ::Val{M}) where {S,T,M}
    X = Expr(:curly, :Tuple)
    SV = S.parameters
    L = 1
    N = length(SV)
    for n ∈ 1:N
        push!(X.parameters, L)
        L *= (SV[n])::Int
    end
    N += 1
    push!(S.parameters, M)
    push!(X.parameters, L)
    L *= M
    quote
        $(Expr(:meta,:inline))
        PtrArray{$S,$T,$N,$X,0,0,true,$L}(θ, tuple(), tuple())
    end
end
@inline function NoPadPtrViewMulti(sp::StackPointer, ::Type{S}, ::Val{M}) where {S,M}
    A = NoPadPtrViewMulti(S, pointer(sp), Val{M}())
    sp + align(full_length(A)), A
end
@inline function NoPadFlatPtrViewMulti(sp::StackPointer, ::Type{Tuple{L}}, ::Val{M}) where {L,M}
    A = PtrArray{Tuple{L,M},T,2}(pointer(θ))
    sp + align(full_length(A)), A
end
@generated function NoPadFlatPtrViewMulti(sp::StackPointer, ::Type{S}, ::Val{M}) where {S,M}
    Expr(
        :block, Expr(:meta,:inline),
        Expr(
            :call, :NoPadPtrViewMulti,
            Expr(:curly, :Tuple, simple_vec_prod(S.parameters)),
            :sp, Expr(:call, Expr(:curly, :Val, M))
        )
    )
end

function Base.similar(::AbstractFixedSizeArray{S,Told,N,X}, ::Type{T}) where {S,T,N,X,Told}
    FixedSizeArray{S,T,N,X}(undef)
end
function Base.similar(A::AbstractStrideArray{S}, ::Type{T}) where {S,T}
    StrideArray{S,T}(undef, size(A))
end
@generated function similarlypermuted(A::AbstractStrideArray{S,Told,N,X}, ::Type{T} = Told) where {S,Told,T,N,X}
    sp = sortperm(reinterpret(UInt,tointvec(X)), alg = Base.Sort.DEFAULT_STABLE)
    Si = tointvec(S)
    Sv = axis_expressions(Si, :(size_tuple(A)))
    Svp = Sv[sp]
    perm = Vector{Int}(undef, N)
    for n in 1:N
        perm[sp[n]] = n
    end
    arrtyp = any(isequal(-1), Si) ? :StrideArray : :FixedSizeArray
    quote
        $(Expr(:meta,:inline))
        PermutedDimsArray($arrtyp{$T}(undef, $(totupleexpr(Svp))), Static{$(totupleexpr(perm))}())
    end
end

allocarray(::Type{T}, s::NTuple{N,Int}) where {T,N} = Array{T}(undef, s)
allocarray(::Type{T}, s::Tuple) where {T} = StrideArray{T}(undef, s)


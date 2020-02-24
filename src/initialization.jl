
function ctuple(n)
    Expr(:curly, :Tuple, n...) 
end
function calc_padding(nrow::Int, T)
    W = VectorizationBase.pick_vector_width(T)
    W > nrow ? VectorizationBase.nextpow2(nrow) : VectorizationBase.align(nrow, T)
end

@generated function StrideArray{S,T}(::UndefInitializer) where {S,T}
    sv = tointvec(S)
    N = length(sv)
    L = calc_padding(first(sv), T)
    xv = similar(sv)
    xv[1] = 1
    for n ∈ 2:N
        svₙ = sv[n]
        xv[n] = L
        L *= svₙ
    end
    W = VectorizationBase.pick_vector_width(T)
    :(StrideArray{$S,$T,$N,$(ctuple(xv)),0,0,false,$L}(SIMDPirates.valloc($T, $L, $W), tuple(), tuple()))
end

function partially_sized(sv)
    st = Expr(:tuple)
    xt = Expr(:tuple)
    xv = similar(sv)
    N = length(sv)
    L = 1
    q = quote end
    for n ∈ 1:N
        xv[n] = L
        if L == -1
            xn = Symbol(:stride_,n)
            push!(q.args, Expr(:(=), xn, Symbol(:stride_, n-1)))
            push!(xt, xn)
        end
        svₙ = sv[n]
        if svₙ == -1
            if L > 0
                push!(q.args, Expr(:(=), Symbol(:stride_, n), L))
            end
            L = -1
            push!(st, Expr(:ref, :s, n))
        elseif L != -1
            L *= svₙ
        end
    end
    st, xt
end

@generated function StrideArray{S,T}(::UndefInitializer, s::NTuple{N,<:Integer}) where {S,T,N}
    sv = tointvec(S)
    @assert N == length(sv)
    any(s -> s == -1, sv) || return :(StrideArray{$S,$T}(::UndefInitializer))
    st, xt = partially_sized(sv)
    SN = length(st.args); XN = length(xt.args)
    W = VectorizationBase.pick_vector_width(T)
    push!(q.args, :(StrideArray{$S,$T,$N,$(ctuple(xv)),$SN,$XN,false,-1}(SIMDPirates.valloc($T, $L, $W), $st, $xt)))
    q
end
function tointvecdt(s)
    ssv = s.paramters
    N = length(ssv)
    sv = Vector{Int}(undef, N)
    for n ∈ 1:N
        ssvₙ = ssv[n]
        sv[n] = ssvₙ <: Static ? VectorizationBase.unwrap(ssvₙ) : -1
    end
    sv
end
@generated function StrideArray{T}(::UndefInitializer, s::Tuple) where {T}
    sv = tointvecdt(s)
    N = length(sv)
    S = Tuple{sv...}
    any(s -> s == -1, sv) || return :(StrideArray{$S,$T}(::UndefInitializer))
    st, xt = partially_sized(sv)
    SN = length(st.args); XN = length(xt.args)
    W = VectorizationBase.pick_vector_width(T)
    push!(q.args, :(StrideArray{$S,$T,$N,$(ctuple(xv)),$SN,$XN,false,-1}(SIMDPirates.valloc($T, $L, $W), $st, $xt)))
    q    
end


function calc_NPL(SV::Core.SimpleVector, T)
    nrow = (SV[1])::Int
    padded_rows = calc_padding(nrow, T)
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
    LA = VectorizationBase.align(L,T)
    N, Tuple{X...}, LA
end

@generated function FixedSizeArray{S,T}(::UndefInitializer) where {S,T}
    N, X, L = calc_NPL(S.parameters, T)
    :(FixedSizeArray{$S,$T,$N,$X,$L}(undef))
end

@generated function PtrArray{S}(ptr::Ptr{T}) where {S,T}
    N, X, L = calc_NPL(S.parameters, T)
    :(PtrArray{$S,$T,$N,$X,0,0,false,$L}(ptr))
end
@generated function PtrArray{S}(ptr::Ptr{T}, s::NTuple{N,<:Integer}) where {S,T,N}
    sv = tointvec(S)
    @assert N == length(sv)
    any(s -> s == -1, sv) || return :(PtrArray{$S}(ptr))
    st, xt = partially_sized(sv)
    SN = length(st.args); XN = length(xt.args)
    push!(q.args, :(StrideArray{$S,$T,$N,$(ctuple(xv)),$SN,$XN,false,-1}(ptr, $st, $xt)))
    q
end
@inline PtrArray{S,T,N}(ptr) where {S,T,N} = PtrArray{S,T}(ptr)
@inline PtrArray{S,T,N,X,SN,XN}(ptr) where {S,T,N,X,SN,XN} = PtrArray{S,T,X,0,0,true,-1}(ptr)
@inline PtrArray{S,T,N,X,SN,XN,V}(ptr) where {S,T,N,X,SN,XN,V} = PtrArray{S,T,X,0,0,V,-1}(ptr)
@generated PtrArray{S,T,N,X,0,0}(ptr) where {S,T,N,X} = Expr(:block, Expr(:meta,:inline), :(PtrArray{$S,$T,$X,0,0,true,$(last(S.parameters)::Int*last(X.parameters)::Int)}(ptr, tuple(), tuple())))
@generated PtrArray{S,T,N,X,0,0,V}(ptr) where {S,T,N,X,V} = Expr(:block, Expr(:meta,:inline), :(PtrArray{$S,$T,$X,0,0,$V,$(last(S.parameters)::Int*last(X.parameters)::Int)}(ptr, tuple(), tuple())))
@generated PtrArray{S,T,N,X,0,0}(ptr, ::NTuple{0}, ::NTuple{0}) where {S,T,N,X} = Expr(:block, Expr(:meta,:inline), :(PtrArray{$S,$T,$X,0,0,true,$(last(S.parameters)::Int*last(X.parameters)::Int)}(ptr, tuple(), tuple())))
@generated PtrArray{S,T,N,X,0,0,V}(ptr, ::NTuple{0}, ::NTuple{0}) where {S,T,N,X,V} = Expr(:block, Expr(:meta,:inline), :(PtrArray{$S,$T,$X,0,0,$V,$(last(S.parameters)::Int*last(X.parameters)::Int)}(ptr, tuple(), tuple())))
@generated PtrArray{S,T,N,X}(ptr, ::NTuple{0}, ::NTuple{0}) where {S,T,N,X} = Expr(:block, Expr(:meta,:inline), :(PtrArray{$S,$T,$X,0,0,true,$(last(S.parameters)::Int*last(X.parameters)::Int)}(ptr, tuple(), tuple())))
@generated PtrArray{S,T,N,X,V}(ptr, ::NTuple{0}, ::NTuple{0}) where {S,T,N,X,V} = Expr(:block, Expr(:meta,:inline), :(PtrArray{$S,$T,$X,0,0,$V,$(last(S.parameters)::Int*last(X.parameters)::Int)}(ptr, tuple(), tuple())))
@generated function PtrArray(ptr::Ptr{T}, s::Tuple) where {T}
    sv = tointvecdt(s)
    N = length(sv)
    S = Tuple{sv...}
    any(s -> s == -1, sv) || return :(PtrArray{$S}(ptr))
    st, xt = partially_sized(sv)
    SN = length(st.args); XN = length(xt.args)
    push!(q.args, :(StrideArray{$S,$T,$N,$(ctuple(xv)),$SN,$XN,false,-1}(ptr, $st, $xt)))
    q
end

@inline function PtrArray(A::AbstractStrideArray{S,T,N,X,SN,XN,V,L}) where {S,T,N,X,SN,XN,V,L}
    PtrAray{S,T,N,X,SN,XN,V,L}(pointer(A), A.size, A.stride)
end
@inline function PtrArray(A::AbstractFixedSizeArray{S,T,N,X,V,L}) where {S,T,N,X,V,L}
    PtrAray{S,T,N,X,0,0,V,L}(pointer(A), tuple(), tuple())
end

@inline function NoPadPtrView(θ::Ptr{T}, ::Type{Tuple{L}}) where {L,T}
    PtrArray{Tuple{L},T,1,Tuple{1},0,0,true,L}(θ, tuple(), tuple())
end
# @inline function NoPadPtrView{Tuple{M,N}}(θ::Ptr{T}) where {M,N,T}
    # PtrArray{Tuple{M,N},T,1,Tuple{1,M},0,0,true}(θ)
# end
@generated function NoPadPtrView(θ::Ptr{T}, ::Type{S}) where {S,T}
    X = Expr(:curly, :Tuple)
    SV = S.parameters
    L = 1
    N = length(SV)
    for n ∈ 1:N
        push!(X.parameters, L)
        L *= (SV[n])::Int
    end
    quote
        $(Expr(:meta,:inline))
        PtrAray{$S,$T,$N,$X,0,0,true,$L}(θ, tuple(), tuple())
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
        PtrAray{$S,$T,$N,$X,0,0,true,$L}(θ, tuple(), tuple())
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


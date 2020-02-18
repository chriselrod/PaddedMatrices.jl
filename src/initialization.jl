
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


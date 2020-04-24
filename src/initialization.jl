
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
    :(StrideArray{$S,$T,$N,$(ctuple(xv)),0,0,false}(SIMDPirates.valloc($T, $L, $W), tuple(), tuple()))
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
    any(isequal(-1), sv) || return :(StrideArray{$S,$T}(::UndefInitializer))
    st, xt = partially_sized(sv)
    SN = length(st.args); XN = length(xt.args)
    W = VectorizationBase.pick_vector_width(T)
    push!(q.args, :(StrideArray{$S,$T,$N,$(ctuple(xv)),$SN,$XN,false}(SIMDPirates.valloc($T, $L, $W), $st, $xt)))
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
    push!(q.args, :(StrideArray{$S,$T,$N,$(ctuple(xv)),$SN,$XN,false}(SIMDPirates.valloc($T, $L, $W), $st, $xt)))
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
function FixedSizeVector{M,T}(::UndefInitializer) where {M,T}
    FixedSizeArray{Tuple{M},T,1,Tuple{1},M}(undef)
end
@generated function FixedSizeMatrix{M,N,T}(::UndefInitializer) where {M,N,T}
    X = calc_padding(M, T)
    :(FixedSizeArray{Tuple{$M,$N},$T,2,Tuple{1,$X},$(X*N)}(undef))
end
@generated function FixedSizeMatrix{M,N,T,X}(::UndefInitializer) where {M,N,T,X}
    @assert X ≥ M
    :(FixedSizeArray{Tuple{$M,$N},$T,2,Tuple{1,$X},$(X*N)}(undef))
end

@generated function PtrArray{S}(ptr::Ptr{T}) where {S,T}
    N, X, L = calc_NPL(S.parameters, T)
    :(PtrArray{$S,$T,$N,$X,0,0,false}(ptr))
end
@generated function PtrArray{S}(ptr::Ptr{T}, s::NTuple{N,<:Integer}) where {S,T,N}
    sv = tointvec(S)
    @assert N == length(sv)
    any(s -> s == -1, sv) || return :(PtrArray{$S}(ptr))
    st, xt = partially_sized(sv)
    SN = length(st.args); XN = length(xt.args)
    push!(q.args, :(PtrArray{$S,$T,$N,$(ctuple(xv)),$SN,$XN,false}(ptr, $st, $xt)))
    q
end
@inline PtrArray{S,T,N}(ptr::Ptr{T}) where {S,T,N} = PtrArray{S}(ptr)
# @inline PtrArray{S,T,N,X,SN,XN}(ptr) where {S,T,N,X,SN,XN} = PtrArray{S,T,X,0,0,true}(ptr)
# @inline PtrArray{S,T,N,X,SN,XN,V}(ptr) where {S,T,N,X,SN,XN,V} = PtrArray{S,T,X,0,0,V}(ptr)
@inline PtrArray{S,T,N,X}(ptr) where {S,T,N,X} = PtrArray{S,T,X,0,0,true}(ptr, tuple(), tuple())
@inline PtrArray{S,T,N,X,0}(ptr) where {S,T,N,X} = PtrArray{S,T,X,0,0,true}(ptr, tuple(), tuple())
@inline PtrArray{S,T,N,X,0,0}(ptr) where {S,T,N,X} = PtrArray{S,T,X,0,0,true}(ptr, tuple(), tuple())
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
    PtrArray{S,T,N,X,SN,XN,V}(pointer(A), A.size, A.stride)
end
@inline function PtrArray(A::AbstractFixedSizeArray{S,T,N,X,V}) where {S,T,N,X,V}
    PtrArray{S,T,N,X,0,0,V}(pointer(A), tuple(), tuple())
end
function PtrArray(A::AbstractArray)
    PtrArray(stridepointer(A), size(A))
end
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
function PtrMatrix{-1,-1}(A::PackedStridedPointer{T,1}, nrows::Integer, ncols::Integer) where {T}
    PtrArray{Tuple{-1, -1}, T, 2, Tuple{1,-1}, 2, 1, false}(pointer(A), (nrows,ncols), A.strides)
end
function PtrMatrix{-1,-1}(A::PackedStridedPointer{T,1}, ::Static{M}, ncols::Integer) where {T, M}
    PtrArray{Tuple{M, -1}, T, 2, Tuple{1,-1}, 1, 1, false}(pointer(A), (ncols,), A.strides)
end
function PtrMatrix{-1,-1}(A::PackedStridedPointer{T,1}, nrows::Integer, ::Static{N}) where {T, N}
    PtrArray{Tuple{-1, N}, T, 2, Tuple{1,-1}, 1, 1, false}(pointer(A), (nrows,), A.strides)
end
function PtrMatrix{-1,-1}(A::PackedStridedPointer{T,1}, ::Static{M}, ::Static{N}) where {T, M, N}
    PtrArray{Tuple{M,N}, T, 2, Tuple{1,-1}, 0, 1, false}(pointer(A), tuple(), A.strides)
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
function PtrMatrix{-1,-1}(A::StaticStridedPointer{T,X}, nrows::Integer, ncols::Integer) where {T, X}
    PtrArray{Tuple{-1, -1}, T, 2, X, 2, 0, false}(pointer(A), (nrows,ncols), tuple())
end
function PtrMatrix{-1,-1}(A::StaticStridedPointer{T,X}, ::Static{M}, ncols::Integer) where {T, M, X}
    PtrArray{Tuple{M, -1}, T, 2, X, 1, 0, false}(pointer(A), (ncols,), tuple())
end
function PtrMatrix{-1,-1}(A::StaticStridedPointer{T,X}, nrows::Integer, ::Static{N}) where {T, N, X}
    PtrArray{Tuple{-1, N}, T, 2, X, 1, 0, false}(pointer(A), (nrows,), tuple())
end
function PtrMatrix{-1,-1}(A::StaticStridedPointer{T,X}, ::Static{M}, ::Static{N}) where {T, M, N, X}
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
function PtrMatrix{-1,-1,X}(A::Ptr{T}, nrows::Integer, ncols::Integer) where {T, X}
    PtrArray{Tuple{-1, -1}, T, 2, Tuple{1,X}, 2, 0, false}(A, (nrows,ncols), tuple())
end
function PtrMatrix{-1,-1,X}(A::Ptr{T}, nrows::Integer, ::Static{N}) where {N, T, X}
    PtrArray{Tuple{-1, N}, T, 2, Tuple{1,X}, 1, 0, false}(A, (nrows,), tuple())
end
function PtrMatrix{-1,-1,X}(A::Ptr{T}, ::Static{M}, ncols::Integer) where {M, T, X}
    PtrArray{Tuple{M, -1}, T, 2, Tuple{1,X}, 1, 0, false}(A, (ncols,), tuple())
end
function PtrMatrix{-1,-1,X}(A::Ptr{T}, ::Static{M}, ::Static{N}) where {M, N, T, X}
    PtrArray{Tuple{M, N}, T, 2, Tuple{1,X}, 0, 0, false}(A, tuple(), tuple())
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


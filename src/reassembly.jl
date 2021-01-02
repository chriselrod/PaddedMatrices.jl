

function stridedpointer_dissasembly(
    sp::StridedPointer{T,2,C,B,R,X,Tuple{Zero,Zero}}
) where {T,C,B,R,X<:Tuple{Integer,Integer}}
    x1, x2 = sp.strd
    xcombined = ((x1 % UInt) << 32) | (x2 % UInt)
    reinterpret(UInt, sp.p), xcombined
    # reinterpret(UInt, sp.p), (x1 % UInt), (x2 % UInt)
end
function stridedpointer_dissasembly(
    sp::StridedPointer{T,2,C,B,R,X,Tuple{Zero,Zero}}
) where {T,C,B,R,X<:Tuple{Integer,StaticInt}}
    reinterpret(UInt, sp.p), first(sp.strd) % UInt64
    # reinterpret(UInt, sp.p), first(sp.strd) % UInt64, zero(UInt64)
end
function stridedpointer_dissasembly(
    sp::StridedPointer{T,2,C,B,R,X,Tuple{Zero,Zero}}
) where {T,C,B,R,X<:Tuple{StaticInt,Integer}}
    reinterpret(UInt, sp.p), last(sp.strd) % UInt64
    # reinterpret(UInt, sp.p), zero(UInt64), last(sp.strd) % UInt64
end
function stridedpointer_dissasembly(
    sp::StridedPointer{T,2,C,B,R,X,Tuple{Zero,Zero}}
) where {T,C,B,R,X<:Tuple{StaticInt,StaticInt}}
    x1, x2 = sp.strd
    reinterpret(UInt, sp.p), zero(UInt64)
end

function combine(
    ::Type{T}, _::UInt
) where {M,N,T <: Tuple{StaticInt{N},StaticInt{M}}}
    StaticInt{N}(), StaticInt{M}()
end
function combine(
    ::Type{T}, x::UInt
) where {M,I<:Integer,T <: Tuple{I,StaticInt{M}}}
    x % I, StaticInt{M}()
end
function combine(
    ::Type{T}, x::UInt
) where {N,I<:Integer,T <: Tuple{StaticInt{N},I}}
    StaticInt{N}(), x % I
end
function combine(
    ::Type{T}, x::UInt
) where {I1<:Integer,I2<:Integer,T <: Tuple{I1,I2}}
    x1 = x >>> 32
    x2 = x & 0x00000000ffffffff
    (x1 % I1, x2 % I2)
end

function stridedpointer_reasembly(
    ::Type{StridedPointer{T,2,C,B,R,X,Tuple{Zero,Zero}}},
    p::UInt, strd::UInt
) where {T,C,B,R,X}
    StridedPointer{T,2,C,B,R}(reinterpret(Ptr{T}, p), combine(X, strd), (Zero(),Zero()))
end

if UInt === UInt64
    function dissassemble(C, A, B)
        pc, xc = stridedpointer_dissasembly(C)
        pa, xa = stridedpointer_dissasembly(A)
        pb, xb = stridedpointer_dissasembly(B)
        (pc, xc, pa, xa, pb, xb)
    end
    function reassemble(
        ::Type{TC},::Type{TA}, ::Type{TB},
        data::Tuple{UInt64,UInt64,UInt64,UInt,UInt,UInt}
    ) where {TC,TA,TB}
        (pc, xc, pa, xa, pb, xb) = data
        C = stridedpointer_reasembly(pc, xc)
        A = stridedpointer_reasembly(pa, xa)
        B = stridedpointer_reasembly(pb, xb)
        C, A, B
    end
else # make it smaller by rearranging
    function dissassemble(C, A, B)
        pc, xc = stridedpointer_dissasembly(C)
        pa, xa = stridedpointer_dissasembly(A)
        pb, xb = stridedpointer_dissasembly(B)
        (xc, xa, xb, pc, pa, pb)
    end
    function reassemble(
        ::Type{TC},::Type{TA}, ::Type{TB},
        data::Tuple{UInt64,UInt64,UInt64,UInt,UInt,UInt}
    ) where {TC,TA,TB}
        (xc, xa, xb, pc, pa, pb) = data
        C = stridedpointer_reasembly(pc, xc)
        A = stridedpointer_reasembly(pa, xa)
        B = stridedpointer_reasembly(pb, xb)
        C, A, B
    end
end

ucoef(x::UInt64) = x
ucoef(x::Integer) = x % UInt64
ucoef(x::Float64) = reinterpret(UInt64, x)
ucoef(x::Float32) = reinterpret(UInt32, x) % UInt64
ucoef(x::Float16) = reinterpret(UInt16, x) % UInt64

coef(::Type{StaticInt{M}}, _::UInt64) where {M} = StaticInt{M}()
coef(::Type{UInt64}, x::UInt64) = x
coef(::Type{T}, x::UInt64) where {T <: Integer} = x % T
function coef(::Type{T}, x::UInt64) where {T <: Float64}
    reinterpret(T, x)
end
function coef(::Type{T}, x::UInt64) where {T <: Float32}
    reinterpret(T, x % UInt32)
end
function coef(::Type{T}, x::UInt64) where {T <: Float16}
    reinterpret(T, x % UInt16)
end

coef(::Type{StaticInt{M}}, _::UInt32) where {M} = StaticInt{M}()
coef(::Type{UInt32}, x::UInt32) = x
coef(::Type{T}, x::UInt32) where {T <: Integer} = x % T





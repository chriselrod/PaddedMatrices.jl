
const BCACHE_COUNT = something(VectorizationBase.CACHE_COUNT[3], 1);
const BSIZE = Int(something(cache_size(Float64, Val(3)), 393216));

const BCACHE_LOCK = Atomic{UInt}(zero(UInt))

struct BCache{T<:Union{UInt,Nothing}}
    p::Ptr{Float64}
    i::T
end
BCache(i::Integer) = BCache(pointer(BCACHE)+8BSIZE*i, i % UInt)
BCache(::Nothing) = BCache(pointer(BCACHE), nothing)

@inline Base.pointer(b::BCache) = b.p
@inline Base.unsafe_convert(::Type{Ptr{T}}, b::BCache) where {T} = Base.unsafe_convert(Ptr{T}, b.p)

function _use_bcache()
    while true
        x = one(UInt)
        if BCACHE_COUNT > 1
            for i ∈ 0:BCACHE_COUNT-1
                if iszero(atomic_or!(BCACHE_LOCK, x) & x) # we've now set the flag, `i` was free
                    return BCache(i)
                end
                x <<= one(UInt)
            end
        else
            if iszero(atomic_or!(BCACHE_LOCK, x) & x) # we've now set the flag, `i` was free
                return BCache(nothing)
            end
        end
        pause()
    end
end
_free_bcache!(b::BCache{UInt}) = (atomic_xor!(BCACHE_LOCK, one(UInt) << b.i); nothing)
_free_bcache!(b::BCache{Nothing}) = reseet_bcache_lock!()#atomic_xor!(BCACHE_LOCK, one(UInt))

"""
  reset_bcache_lock!()

Currently not using try/finally in matmul routine, despite locking.
So if it errors for some reason, you may need to manually call `reset_bcache_lock!()`.
"""
reseet_bcache_lock!() = (BCACHE_LOCK[] = zero(UInt); nothing)


let ityp = "i$(8sizeof(UInt))"
    @eval begin
        @inline function _atomic_load(ptr::Ptr{UInt})
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %v = load atomic volatile $(ityp), $(ityp)* %p acquire, align $(Base.gc_alignment(UInt))
              ret $(ityp) %v
            """), UInt, Tuple{Ptr{UInt}}, ptr)
        end
        @inline function _atomic_store!(ptr::Ptr{UInt}, x::UInt)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              store atomic volatile $(ityp) %1, $(ityp)* %p release, align $(Base.gc_alignment(UInt))
              ret void
            """), Cvoid, Tuple{Ptr{UInt}, UInt}, ptr, x)
        end
        @inline function _atomic_cas_cmp!(ptr::Ptr{UInt}, cmp::UInt, newval::UInt)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %c = cmpxchg volatile i64* %p, i64 %1, i64 %2 acq_rel acquire
              %bit = extractvalue { i64, i1 } %c, 1
              %bool = zext i1 %bit to i8
              ret i8 %bool
            """), Bool, Tuple{Ptr{UInt}, UInt, UInt}, ptr, cmp, newval)
        end
        @inline function _atomic_add!(ptr::Ptr{UInt}, x::UInt)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %v = atomicrmw volatile add $(ityp)* %p, $(ityp) %1 acq_rel
              ret $(ityp) %v
            """), UInt, Tuple{Ptr{UInt}, UInt}, ptr, x)
        end
        @inline function _atomic_or!(ptr::Ptr{UInt}, x::UInt)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %v = atomicrmw volatile or $(ityp)* %p, $(ityp) %1 acq_rel
              ret $(ityp) %v
            """), UInt, Tuple{Ptr{UInt}, UInt}, ptr, x)
        end
        @inline function _atomic_xor!(ptr::Ptr{UInt}, x::UInt)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %v = atomicrmw volatile xor $(ityp)* %p, $(ityp) %1 acq_rel
              ret $(ityp) %v
            """), UInt, Tuple{Ptr{UInt}, UInt}, ptr, x)
        end
        @inline function _atomic_max!(ptr::Ptr{UInt}, x::UInt)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %v = atomicrmw volatile umax $(ityp)* %p, $(ityp) %1 acq_rel
              ret $(ityp) %v
            """), UInt, Tuple{Ptr{UInt}, UInt}, ptr, x)
        end
        @inline function _atomic_min!(ptr::Ptr{UInt}, x::UInt)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %v = atomicrmw volatile umin $(ityp)* %p, $(ityp) %1 acq_rel
              ret $(ityp) %v
            """), UInt, Tuple{Ptr{UInt}, UInt}, ptr, x)
        end
    end
end



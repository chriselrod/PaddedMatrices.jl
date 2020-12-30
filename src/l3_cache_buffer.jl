
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
_free_bcache!(b::BCache{UInt}) = atomic_xor!(BCACHE_LOCK, one(UInt) << b.i)
_free_bcache!(b::BCache{Nothing}) = reseet_bcache_lock!()#atomic_xor!(BCACHE_LOCK, one(UInt))

"""
  reset_bcache_lock!()

Currently not using try/finally in matmul routine, despite locking.
So if it errors for some reason, you may need to manually call `reset_bcache_lock!()`.
"""
reseet_bcache_lock!() = BCACHE_LOCK[] = zero(UInt)


let ityp = "i$(8sizeof(UInt))"
    @inline function _atomic_load(ptr::Ptr{UInt})
        Base.llvmcall("""
          %p = inttoptr $(ityp) %0 to $(ityp)*
          %v = load atomic $(ityp), $(ityp)* %p acquire, align $(Base.gc_alignment(UInt))
          ret $(ityp) %v
        """, UInt, Tuple{Ptr{UInt}}, ptr)
    end
    @inline function _atomic_store!(ptr::Ptr{UInt}, x::UInt)
        Base.llvmcall("""
          %p = inttoptr $(ityp) %0 to $(ityp)*
          store atomic $(ityp) %1, $(ityp)* %p release, align $(Base.gc_alignment(UInt))
          ret void
        """, UInt, Tuple{Ptr{UInt}, UInt}, ptr, x)
    end
    @inline function _atomic_or!(ptr::Ptr{UInt}, x::UInt)
        Base.llvmcall("""
          %p = inttoptr $(ityp) %0 to $(ityp)*
          %v = = atomicrmw or $(ityp)* %2, $(ityp) %p acq_rel
          ret $(ityp) %v
        """, UInt, Tuple{Ptr{UInt}, UInt}, ptr, x)
    end
    @inline function _atomic_xor!(ptr::Ptr{UInt}, x::UInt)
        Base.llvmcall("""
          %p = inttoptr $(ityp) %0 to $(ityp)*
          %v = = atomicrmw xor $(ityp)* %2, $(ityp) %p acq_rel
          ret $(ityp) %v
        """, UInt, Tuple{Ptr{UInt}, UInt}, ptr, x)
    end
end


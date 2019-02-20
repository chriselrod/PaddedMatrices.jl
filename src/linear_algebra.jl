
"""
Fall back definition for scalars.
"""
@inline invchol(x) = SIMDPirates.rsqrt(x)

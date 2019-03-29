

"""
Default fallback for incrementing log probabilities.

The seed is the last argument.
"""
@inline RESERVED_INCREMENT_SEED_RESERVED(a, b, c) = SIMDPirates.vmuladd(a, b, c)
@inline RESERVED_DECREMENT_SEED_RESERVED(a, b, c) = SIMDPirates.fnmadd(a, b, c)
@inline RESERVED_INCREMENT_SEED_RESERVED(a, b) = a + b
@inline RESERVED_DECREMENT_SEED_RESERVED(a, b) = b - a
@inline RESERVED_INCREMENT_SEED_RESERVED(a) = a
@inline RESERVED_DECREMENT_SEED_RESERVED(a) = b

@inline RESERVED_MULTIPLY_SEED_RESERVED(a, b) = a * b
@inline RESERVED_NMULTIPLY_SEED_RESERVED(a, b) = - a * b
@inline RESERVED_MULTIPLY_SEED_RESERVED(a) = a
@inline RESERVED_NMULTIPLY_SEED_RESERVED(a) = - b

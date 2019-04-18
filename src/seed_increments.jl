

"""
Default fallback for incrementing log probabilities.

The seed is the last argument.
"""
@inline function RESERVED_INCREMENT_SEED_RESERVED(a, b, c)
    # @show typeof(a), typeof(b), typeof(c)
    # @show size(a), size(b), size(c)
    out = SIMDPirates.vmuladd(a, b, c)
    # @assert !(isa(out, Array) || isa(out, LinearAlgebra.Adjoint{<:Any,<:Array}))
    out
end
@inline function RESERVED_DECREMENT_SEED_RESERVED(a, b, c)
    out = SIMDPirates.fnmadd(a, b, c)
    # @assert !(isa(out, Array) || isa(out, LinearAlgebra.Adjoint{<:Any,<:Array}))
    out
end

@inline function RESERVED_INCREMENT_SEED_RESERVED(a, b)
    # println("Increment two")
    # @show typeof(a), typeof(b)
    # @show size(a), size(b)
    out = a + b
    # @assert !(isa(out, Array) || isa(out, LinearAlgebra.Adjoint{<:Any,<:Array}))
    out
end

@inline function RESERVED_DECREMENT_SEED_RESERVED(a, b)
    out = b - a
    # @assert !(isa(out, Array) || isa(out, LinearAlgebra.Adjoint{<:Any,<:Array}))
    out
end

@inline RESERVED_INCREMENT_SEED_RESERVED(a) = a
@inline RESERVED_DECREMENT_SEED_RESERVED(a) = -a


ussize(a) = size(a)
ussize(::UniformScaling) = ()
@inline function RESERVED_MULTIPLY_SEED_RESERVED(a, b)
    # println("Multiply two")
    # @show typeof(a), typeof(b)
    # @show ussize(a), ussize(b)
    out = a * b
    # @assert !(isa(out, Array) || isa(out, LinearAlgebra.Adjoint{<:Any,<:Array}))
    out
end

@inline function RESERVED_NMULTIPLY_SEED_RESERVED(a, b)
    out = - a * b
    # @assert !(isa(out, Array) || isa(out, LinearAlgebra.Adjoint{<:Any,<:Array}))
    out
end

@inline RESERVED_MULTIPLY_SEED_RESERVED(a) = a
@inline function RESERVED_NMULTIPLY_SEED_RESERVED(a)
    out = - a
    # @assert !(isa(out, Array) || isa(out, LinearAlgebra.Adjoint{<:Any,<:Array}))
    out
end

bc1 = Base.broadcasted(-, pt, pt');
bc2 = Base.broadcasted(abs, bc1);
bc3 = Base.broadcasted(log, ρ);
bc4 = Base.broadcasted(*, bc2, bc3);
bc5 = Base.broadcasted(exp, bc4);


AR = MutableFixedSizeMatrix{30,30,Float64,30}(undef);
t = cumsum(@Mutable rand(30));
pt = PtrVector{30,Float64,30}(pointer(t));
ρ = 0.6


AR .= exp.( abs.( pt .- pt' ) .* log( ρ ) )

@. AR = ρ .^ abs( pt - pt' )



# julia> B[:,1] .= A
# ERROR: MethodError: no method matching reduce_size(::Array{DataType,1})
# Closest candidates are:
  # reduce_size(::Core.SimpleVector) at /home/chriselrod/.julia/dev/PaddedMatrices/src/broadcast.jl:36

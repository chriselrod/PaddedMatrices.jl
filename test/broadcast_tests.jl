


AR = MutableFixedSizePaddedMatrix{30,30,Float64,30}(undef);
t = cumsum(@Mutable rand(30));
pt = PtrVector{30,Float64,30}(pointer(t));



@. AR = exp( abs( pt - pt' ) * log( ρ ) )

@. AR = ρ .^ abs( pt - pt' )

struct StackPointer
    ptr::Ptr{Cvoid}
end
@inline Base.pointer(s::StackPointer) = s.ptr
@inline Base.pointer(s::StackPointer, ::Type{T}) where {T} = Base.unsafe_convert(Ptr{T}, s.ptr)

# These are for "fuzzing" offsets; answers shouldn't change for SPO ≥ 0, so if they do, there's a bug!
#const SPO = Ref{Int}(800);
#@inline Base.:+(sp::StackPointer, i::Integer) = StackPointer(sp.ptr + i + SPO[])
#@inline Base.:+(i::Integer, sp::StackPointer) = StackPointer(sp.ptr + i + SPO[])

@inline Base.:+(sp::StackPointer, i::Integer) = StackPointer(sp.ptr + i)
@inline Base.:+(i::Integer, sp::StackPointer) = StackPointer(sp.ptr + i)
@inline Base.:-(sp::StackPointer, i::Integer) = StackPointer(sp.ptr - i)

# (Module, function) pairs supported by StackPointer.
#const STACK_POINTER_SUPPORTED_MODMETHODS = Set{Tuple{Symbol,Symbol}}()
const STACK_POINTER_SUPPORTED_METHODS = Set{Symbol}()

macro support_stack_pointer(mod, func)
    esc(quote
#        push!(PaddedMatrices.STACK_POINTER_SUPPORTED_MODMETHODS, ($(QuoteNode(mod)),$(QuoteNode(func))))
        push!(PaddedMatrices.STACK_POINTER_SUPPORTED_METHODS, $(QuoteNode(func)))
        @inline $mod.$func(sp::PaddedMatrices.StackPointer, args...) = (sp, $mod.$func(args...))
    end)
end
macro support_stack_pointer(func)
    # Could use @__MODULE__
    esc(quote
#        push!(PaddedMatrices.STACK_POINTER_SUPPORTED_MODMETHODS, ($(QuoteNode(mod)),$(QuoteNode(func))))
        push!(PaddedMatrices.STACK_POINTER_SUPPORTED_METHODS, $(QuoteNode(func)))
        @inline $func(sp::PaddedMatrices.StackPointer, args...) = (sp, $func(args...))
    end)
end

function ∂getindex end
function ∂materialize end
#function ∂mul end
#function ∂add end
#function ∂muladd end

@support_stack_pointer Base getindex
@support_stack_pointer Base.Broadcast materialize
@support_stack_pointer Base (*)
@support_stack_pointer Base (+)
@support_stack_pointer Base (-)
@support_stack_pointer Base similar
@support_stack_pointer Base copy

#@support_stack_pointer SIMDPirates vmul
#@support_stack_pointer SIMDPirates vadd
#@support_stack_pointer SIMDPirates vsub

@support_stack_pointer ∂getindex
@support_stack_pointer ∂materialize
#@support_stack_pointer ∂mul
#@support_stack_pointer ∂add
#@support_stack_pointer ∂muladd

@support_stack_pointer vexp

function stack_pointer_pass(expr, stacksym, blacklist = nothing)
    if blacklist == nothing
        whitelist = STACK_POINTER_SUPPORTED_METHODS
    else
        whitelist = setdiff(STACK_POINTER_SUPPORTED_METHODS, blacklist)
    end
    verbose = false
    postwalk(expr) do ex
        if @capture(ex, B_ = mod_.func_(args__)) && func ∈ whitelist
            if verbose
                return :(($stacksym, $B) = $mod.$func($stacksym, $(args...)); println($(string(func))); @show pointer($stacksym); @show (reinterpret(Int, pointer($stacksym)) - reinterpret(Int, pointer(ProbabilityModels.STACK_POINTER)))/8)
            else
                return :(($stacksym, $B) = $mod.$func($stacksym, $(args...)))
            end
        elseif @capture(ex, B_ = func_(args__)) && func ∈ whitelist
            ##            if func ∈ whitelist
            if verbose
                return :(($stacksym, $B) = $func($stacksym, $(args...)); println($(string(func))); @show pointer($stacksym); @show (reinterpret(Int, pointer($stacksym)) - reinterpret(Int, pointer(ProbabilityModels.STACK_POINTER)))/8)
            else
                return :(($stacksym, $B) = $func($stacksym, $(args...)))
            end
##            elseif func isa GlobalRef && func.name ∈ whitelist
##                return :(($stacksym, $B) = $(func.name)($stacksym, $(args...)))
##            end
        elseif @capture(ex, B_ = mod_.func_{T__}(args__)) && func ∈ whitelist
            if verbose
                return :(($stacksym, $B) = $mod.$func{$(T...)}($stacksym, $(args...)); println($(string(func))); @show pointer($stacksym); @show (reinterpret(Int, pointer($stacksym)) - reinterpret(Int, pointer(ProbabilityModels.STACK_POINTER)))/8)
            else
                return :(($stacksym, $B) = $mod.$func{$(T...)}($stacksym, $(args...)))
            end
        elseif @capture(ex, B_ = func_{T__}(args__)) && func ∈ whitelist
            if verbose
                return :(($stacksym, $B) = $func{$(T...)}($stacksym, $(args...)); println($(string(func))); @show pointer($stacksym); @show (reinterpret(Int, pointer($stacksym)) - reinterpret(Int, pointer(ProbabilityModels.STACK_POINTER)))/8)
            else
                return :(($stacksym, $B) = $func{$(T...)}($stacksym, $(args...)))
            end
        else
            return ex
        end
    end
end


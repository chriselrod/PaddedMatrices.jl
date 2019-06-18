struct StackPointer
    ptr::Ptr{Cvoid}
end
@inline Base.pointer(s::StackPointer) = s.ptr
@inline Base.pointer(s::StackPointer, ::Type{T}) where {T} = Base.unsafe_convert(Ptr{T}, s.ptr)

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

@support_stack_pointer Base getindex
@support_stack_pointer Base materialize
@support_stack_pointer Base (*)
@support_stack_pointer Base (+)
@support_stack_pointer Base similar
@support_stack_pointer Base copy

function stack_pointer_pass(expr, stacksym, blacklist = nothing)
    if blacklist == nothing
        whitelist = STACK_POINTER_SUPPORTED_METHODS
    else
        whitelist = setdiff(STACK_POINTER_SUPPORTED_METHODS, blacklist)
    end
    postwalk(expr) do ex
        if @capture(ex, B_ = mod_.func_(args__)) && func ∈ whitelist
            return :(($stacksym, $B) = $mod.$func($stacksym, $(args...)))
        elseif @capture(ex, B_ = func_(args__)) && func ∈ whitelist
            return :(($stacksym, $B) = $func($stacksym, $(args...)))
        elseif @capture(ex, B_ = mod_.func_{T__}(args__)) && func ∈ whitelist
            return :(($stacksym, $B) = $mod.$func{$(T...)}($stacksym, $(args...)))
        elseif @capture(ex, B_ = func_{T__}(args__)) && func ∈ whitelist
            return :(($stacksym, $B) = $func{$(T...)}($stacksym, $(args...)))
        else
            return ex
        end
    end
end


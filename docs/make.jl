using PaddedMatrices
using Documenter

makedocs(;
    modules=[PaddedMatrices],
    authors="Chris Elrod",
    repo="https://github.com/chriselrod/PaddedMatrices.jl/blob/{commit}{path}#L{line}",
    sitename="PaddedMatrices.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chriselrod.github.io/PaddedMatrices.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Architecture Benchmarks" => [
            "arches/cascadelake.md",
            "arches/tigerlake.md",
            "arches/haswell.md"
        ],
        "Random Number Generation" => "rng.md",
        "Broadcasting" => "broadcasting.md",
        "Stack Allocattion" => "stack_allocation.md"        
    ],
    strict=false,
)

deploydocs(;
    repo="github.com/chriselrod/PaddedMatrices.jl",
)

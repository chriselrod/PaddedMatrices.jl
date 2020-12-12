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
    ],
    strict=false,
)

deploydocs(;
    repo="github.com/chriselrod/PaddedMatrices.jl",
)

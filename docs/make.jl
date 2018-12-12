using Documenter, PaddedMatrices

makedocs(;
    modules=[PaddedMatrices],
    format=:html,
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/chriselrod/PaddedMatrices.jl/blob/{commit}{path}#L{line}",
    sitename="PaddedMatrices.jl",
    authors="Chris Elrod",
    assets=[],
)

deploydocs(;
    repo="github.com/chriselrod/PaddedMatrices.jl",
    target="build",
    julia="1.0",
    deps=nothing,
    make=nothing,
)

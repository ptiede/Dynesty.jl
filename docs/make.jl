using Dynesty
using Documenter

DocMeta.setdocmeta!(Dynesty, :DocTestSetup, :(using Dynesty); recursive=true)

makedocs(;
    modules=[Dynesty],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    repo="https://github.com/ptiede/Dynesty.jl/blob/{commit}{path}#{line}",
    sitename="Dynesty.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ptiede.github.io/Dynesty.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ptiede/Dynesty.jl",
    devbranch="main",
)

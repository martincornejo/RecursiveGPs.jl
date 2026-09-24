using Documenter, RecursiveGPs, Literate
import LowLevelParticleFilters  # needed so @docs can resolve LLPF.state / LLPF.covariance

# Raster output keeps the built pages under Documenter's size threshold; vector
# figures with thousands of points blow past it.
using CairoMakie
CairoMakie.activate!(type = "png")

# ---------------------------------------------------------------------------
# Generate tutorial markdown files from the example scripts in examples/.
# The output .md files land in docs/src/tutorials/ at build time and are
# .gitignore'd.
#
# DocumenterFlavor produces @example blocks, which Documenter executes, so the
# figures and printed output appear in the docs and a broken example fails the
# build. This requires the example dependencies in docs/Project.toml.
# ---------------------------------------------------------------------------
examples_dir = joinpath(@__DIR__, "..", "examples")
tutorial_out = joinpath(@__DIR__, "src", "tutorials")

for jl_file in [
        "basic_rgp.jl",
        "combined_rgp.jl",
        "hyperparameter_tuning.jl",
        "friction_learning.jl",
    ]
    Literate.markdown(
        joinpath(examples_dir, jl_file),
        tutorial_out;
        flavor = Literate.DocumenterFlavor(),
    )
end

makedocs(
    sitename = "RecursiveGPs.jl",
    modules = [RecursiveGPs],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    checkdocs = :exports,   # only warn about exported symbols missing from docs
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Mathematical Background" => "math_background.md",
        "Tutorials" => [
            "Basic RGP" => "tutorials/basic_rgp.md",
            "Multi-Component RGPs" => "tutorials/combined_rgp.md",
            "Hyperparameter Tuning" => "tutorials/hyperparameter_tuning.md",
            "Learning Missing Physics" => "tutorials/friction_learning.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/martincornejo/RecursiveGPs.jl.git",
)

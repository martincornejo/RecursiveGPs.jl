using Documenter, RecursiveGPs, Literate
import LowLevelParticleFilters  # needed so @docs can resolve LLPF.state / LLPF.covariance

# ---------------------------------------------------------------------------
# Generate tutorial markdown files directly from the example scripts in
# examples/.  The output .md files land in docs/src/tutorials/ at build time
# and are .gitignore'd.
#
# CommonMarkFlavor produces plain ```julia blocks (not @example blocks), so
# Documenter treats them as static code — no extra package deps needed in docs/.
# ---------------------------------------------------------------------------
examples_dir = joinpath(@__DIR__, "..", "examples")
tutorial_out = joinpath(@__DIR__, "src", "tutorials")

for (jl_file, _) in [
        "basic_rgp.jl" => "basic_rgp.md",
        "combined_rgp.jl" => "combined_rgp.md",
        "hyperparameter_tuning.jl" => "hyperparameter_tuning.md",
    ]
    Literate.markdown(
        joinpath(examples_dir, jl_file),
        tutorial_out;
        flavor = Literate.CommonMarkFlavor(),
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
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/martincornejo/RecursiveGPs.jl.git",
)

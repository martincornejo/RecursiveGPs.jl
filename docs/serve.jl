# Local documentation preview with live reload.
#
#     julia --project=docs docs/serve.jl
#
# Serves the docs at http://localhost:8000 and rebuilds whenever a file in
# docs/src/ or examples/ changes. docs/src/tutorials/ holds the markdown that
# make.jl generates from examples/, so it is skipped; otherwise every build
# would trigger the next one.

using LiveServer

servedocs(;
    include_dirs = [joinpath(@__DIR__, "..", "examples")],
    skip_dirs = [joinpath(@__DIR__, "src", "tutorials")],
)

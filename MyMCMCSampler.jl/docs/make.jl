push!(LOAD_PATH, dirname(dirname(@__DIR__)))
using Documenter, MyMCMCSampler
makedocs(sitename="MyMCMCSampler Documentation")
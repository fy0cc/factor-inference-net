# takes in a folder with all sample results, create a dataset in 

using Distributed

##
@everywhere using Pkg
@everywhere Pkg.activate(pwd())
Pkg.instantiate()
@everywhere using JLD2, FileIO
@everywhere using StatsBase
@everywhere using MyMCMCSampler

using PyCall
using Distributions
using DataFrames
using ArgParse
using ProgressMeter
using Base.Iterators: enumerate

py"""
import deepdish as dd
def savefile(filepath,data):
    dd.io.save(filepath,data)
"""
##
sa = ArgParseSettings()
@add_arg_table sa begin
    "--indir"
        help = "input folder containing all MCMC samples"
        arg_type = String
        default = "data/third/continuous/10small/samples"
    "--outdir"
        help = "folder to store dataset file"
        arg_type = String
        default = "data/tmp/"
    "--statistics"
        help = "which statistis to calculate"
        arg_type = String
        default = "standardized_moments"
    "--order"
        help = "how many moments to use"
        arg_type = Int
        default = 4
end

args = parse_args(ARGS, sa)
for i in procs()
    @spawnat i args |> fetch
end
indir = args["indir"]
outdir = args["outdir"]
@everywhere order = args["order"]

##
# get path for all graphs
filenames = readdir(indir)
graph_paths = [ indir*"/"*filename for filename in filenames]
@show "$(length(graph_paths)) graphs to be loaded"

# load samples and calculate statistics
if args["statistics"] == "standardized_moments"
    if order <= 2
        @everywhere stats = vcat(
            [mean],
            [x->abs(moment(x,i)).^(1/i).*sign(moment(x,i)) for i in 2:order]
        )
    else
        @everywhere stats = vcat(
            [mean, x->moment(x,2)],
            [x->abs(moment(x,i)).^(1/i) ./ (moment(x,2).^(0.5)) .* sign(moment(x,i)) for i in 3:order]
        )
    end
elseif args["statistics"] == "central_moments"
    @everywhere stats = vcat([mean], [x->abs(moment(x,i)).^(1/i) .* sign(moment(x,i)) for i in 2:order]) 
elseif args["statistics"] == "moments"
    @everywhere stats = [x-> abs(mean(x.^i)).^(1/i).*sign(mean(x.^i)) for i in 1:order];
end


@everywhere function load_and_compute(path)
    data = load(path)
    g = data["graph"]
    samples = data["samples"]  # (nsample, nvar, nchain)
    samples = vcat([samples[:,:,i] for i in 1:size(samples,3)]...) # (nsamples, nvar)
    @assert ndims(samples) == 2
    dict = dist2dict(g)
    dict["beta"] = data["beta"]
    dict["alpha"] = data["alpha"]
    dict["p"] = data["p"]
    dict["k2"] = data["k2"]
    dict["k3"] = data["k3"]
    nmoments = length(stats)
    nvar = size(samples,2)
    marginals = zeros(nvar, nmoments)
    for i in 1:nvar, j in 1:nmoments
        marginals[i,j]  = stats[j](samples[:,i])
    end
    return (dict, marginals)
end

results = @showprogress pmap(graph_paths) do path
    load_and_compute(path)
end

# show statistics
M = vcat([item[2] for item in results]...)
for i in 1:order
    println("------------ Statistics $i ------------")
    println(summarystats(M[:,i]))
end

## save to file
if  !isdir(outdir)
    println("$(outdir) doesn't exist, creating")
    mkpath(outdir)
end
outpath = joinpath(outdir, "data.dd")
@show "Saving dataset to $outpath"
py"savefile"(outpath, results)
@show "File saved"

using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(pwd())
Pkg.instantiate()
using MyMCMCSampler
using StatsBase
using MCMCDiagnostics
using Mamba
using StanSample
using ArgParse
using JLD2, FileIO
using ProgressMeter
ProjDir = @__DIR__
tmpdir= ProjDir * "/tmp"

sa = ArgParseSettings()
@add_arg_table sa begin
    "--outdir"
        help = "folder to store dataset file"
        arg_type = String
        default = "data/tmp/"
    "--n"
        help = "number of variable"
        arg_type = Int
        default = 10
    "--ndata"
        help = "number of different graphs per hyper-parameter"
        arg_type = Int
        default = 1
    "--beta"
        help = "inverse temperature"
        arg_type = Float64
        default = 0.3
    "--alpha"
        help = "relative strength of the 4th order term"
        arg_type = Float64
        default = 1.0
    "--scale"
        help = "scaling of singleton and interaction coefficients"
        arg_type = Float64
        default = 1.0
end

args = parse_args(ARGS, sa)
println(args)
n = args["n"]
ndata = args["ndata"]
outdir = args["outdir"]

stan_program = "
data {
    int N; // number of variables
    int M2;  // number of pariwise potentials
    int M3; // number of third-order potentials
    vector[N] s[3];  // singleton potentials, up to third order
    vector[M2] p; // pairwise potentials
    vector[M3] t;  //  third-order potentials
    int pid[2,M2]; // indices for pairwise potentials
    int tid[3,M3]; // indices for third-order potentials
    real alpha;  // coeff for 4th order base measure
    real beta;  // inverse temp
}
parameters {
    vector[N] x;  // variables
}


model{
    // singleton potentials
    target += - beta * ( s[1].*x + s[2].*x.*x + s[3].*x.*x.*x ) ;
    // pairwise potentials
    target += - beta * (p .* x[pid[1]] .* x[pid[2]]);
    // third-order potentials
    target += - beta * (t .* x[tid[1]] .* x[tid[2]] .* x[tid[3]]); 
    // base measure
    // target += - beta * (alpha * pow(sum(x.*x),2)); 
    target += - beta * alpha * sum(pow(x,4)); 
}

";



# ks = range(2.0,n,length=30)
# p0s = range(0.0,1.0,length=30)
k2range = Uniform(2.0,n-1)
k3range = Uniform(2.0, floor((n-1)*(n-2)/6))
prange = Uniform(0.0, 1.0)

# alpha =  0.2
# beta = 0.05
# beta = 0.3
beta0 = args["beta"]
alpha0 = args["alpha"]
scale = args["scale"]
beta = beta0 
alpha = alpha0 
gamma = 1/ scale
gamma1 = 1.0

counter = 0
prog = Progress(ndata)
# for k in ks,p0 in p0s, i in 1:ndata
for i in 1:ndata
    k2 = rand(k2range)
    k3 = rand(k3range)
    p0 = rand(prange)
    global counter
    counter += 1
    valid = false
    sm = nothing
    sm0 = nothing
    data = nothing
    data0 = nothing
    g = nothing
    while !valid
        # g = random_ws_factor_graph(n,k,p0;β=3.0,degrees=[2,3])
        # g = random_ws_factor_graph(n,k2,k3,p0;β=1.0, degrees=[2,3],singleton_dist=:randn)  # old, used for training set
        g = random_ws_factor_graph(n,k2,k3,p0;γ=gamma,γ1=gamma1, degrees=[2,3],singleton_dist=:randn ) 
        idx2 = findall(x->x==2,g.edge_type)
        idx3 = findall(x->x==3,g.edge_type)
        M2 = length(idx2)
        M3 = length(idx3)
        # s = permutedims(hcat(g.node_attr...),[2,1])
        s = hcat(g.node_attr...)  # (3,N)
        # s = [s[i,:] for i in 1:size(s,1)]
        p = vcat(g.edge_attr[idx2]...) |> Array{Float64}
        t = vcat(g.edge_attr[idx3]...) |> Array{Float64}
        pid = hcat(g.edge_index[idx2]...) |> Array{Int}
        tid = hcat(g.edge_index[idx3]...) |> Array{Int}
        stan_data = Dict(
            "N" => n,
            "M2" =>M2,
            "M3" =>M3,
            "s" => s,
            "p" =>p,
            "t"=>t,
            "pid"=>pid,
            "tid"=>tid,
            "alpha"=>alpha,
            "beta"=>beta,
        )
        # baseline model without all the interactions

        stan_data_singleton = Dict(
            "N" => n,
            "M2" =>M2,
            "M3" =>M3,
            "s" => s,
            "p" =>zero(p),
            "t"=>zero(t),
            "pid"=>pid,
            "tid"=>tid,
            "alpha"=>alpha,
            "beta"=>beta,
        )
        sm = SampleModel("test", stan_program, [8],
          method=StanSample.Sample(
                save_warmup=false, num_warmup=10000,
                num_samples=10000, 
                thin=4, 
                adapt=StanSample.Adapt(delta=0.85),
                algorithm=StanSample.Hmc(StanSample.Nuts(max_depth=30))
                ),
          tmpdir=tmpdir
        );

        sm0 = SampleModel("test0", stan_program, [8],
          method=StanSample.Sample(save_warmup=false, num_warmup=5000,
          num_samples=5000, thin=4, adapt=StanSample.Adapt(delta=0.9)),
          tmpdir=tmpdir
        );


        try
            rc = stan_sample(sm, data=stan_data);
            rc0 = stan_sample(sm0, data=stan_data_singleton); 
            chains = read_samples(sm; output_format=:mambachains,mpsrf=true, transform=true)
            gel = gelmandiag(chains)
            if all(gel.value[:,2,1] .< 1.3)
                valid=true
                data = chains.value
                chains0 = read_samples(sm0; output_format=:mambachains,mpsrf=true, transform=true)
                data0 = chains0.value
            end
        catch e
            if isa(e,LoadError)
                println("Process failed, skipping")
            end
            valid = false
        end
    end
    # save graph to file
    outdict = Dict(
        "graph" => g,
        "samples" => data,
        "samples0" => data0,
        "p" => p0,
        "k2" => k2,
        "k3" => k3,
        "alpha" => alpha,
        "beta" => beta,
        "gamma" =>gamma,
        "gamma1" =>gamma1,
        "n" => n
    )
    outfilepath = "$outdir/$(counter).jld2"
    save(outfilepath, outdict)
    next!(prog)
end
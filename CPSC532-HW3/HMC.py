import torch as tc
import json
from copy import deepcopy
from tqdm import tqdm

from graph_based_sampling import evaluate_graph, graph
from evaluation_based_sampling import Env, Eval, standard_env
from utils import log_sample

def split_latent_obs(g : graph)-> tuple[list, list]:
    X = []
    Y = []
    for v in g.Graph["V"]: #iterate over the vertices in the graph
        if v in g.Graph["Y"].keys(): #if its observed
            Y.append(v)
        else: #else its latent
            X.append(v)

    return X, Y

def sample_from_joint(g : graph) -> dict:

    ordered_vars = g.topological()
    env = standard_env()
    X, Y = split_latent_obs(g)

    for v in ordered_vars:
        if v in X: # if the vertex is latent, sample from the prior
            expr = g.Graph["P"][v]
            env[v], _ = Eval(expr, {"logW":tc.tensor(0.0)}, env)
        elif v in Y: #if the vertex is observed, get the observed value
            env[v] = g.Graph["Y"][v]

    sample = {v : env[v] for v in X+Y}
    return sample

def U(X : dict, Y : dict, g : graph)-> tc.tensor:
    env = standard_env()
    env.update(X)
    env.update(Y)
    sigma = {"logW": tc.tensor(0.0)}

    for y in Y.keys():
        d, _ = Eval(g.Graph["P"][y][1], {"logW":tc.tensor(0.0)}, env)
        sigma["logW"] = sigma["logW"] + d.log_prob(tc.tensor(Y[y]).float())
    
    for x in X.keys():
        d, _ = Eval(g.Graph["P"][x][1], {"logW":tc.tensor(0.0)}, env)
        sigma["logW"] = sigma["logW"] + d.log_prob(X[x]).float()
    
    return -sigma["logW"]

def K(R : tc.tensor, M_inv : tc.tensor) -> tc.tensor:
    if R.ndim<=1:
        return 0.5 * tc.matmul(R, tc.matmul(M_inv, R))
    else:
        return 0.5 * tc.matmul(R.T, tc.matmul(M_inv, R))

def H(X: dict ,Y: dict, R : tc.tensor, M_inv : tc.tensor, g : graph):
    
    return U(X, Y, g) + K(R, M_inv)

def grad(X : dict, Y : dict, g : graph):
    u = U(X, Y, g)
    gradients = tc.zeros(len(X))

    if u.requires_grad!=True: #hack : this happens when the function is Id; i.e. no operations in the graph
        return gradients

    u.backward()

    for i, key in enumerate(list(X.keys())):
        if X[key].grad!=None: 
            gradients[i] = X[key].grad
        else:
            pass
    
    return gradients

def leapfrog(X : dict, Y : dict, R : tc.tensor, T : int, eps : float, g : graph)-> tuple[dict, tc.tensor]:
    
    def inc_dict(d : dict, delta_vec : tc.tensor)-> dict: #utility function to increment a dictionary of values by delta_vec
        v = {}
        for i, k in enumerate(list(d.keys())):
            v[k] = d[k].detach() + delta_vec[i]
            v[k].requires_grad = True
        return v
        
    R = R - 0.5 * eps * grad(X, Y, g)

    for t in range(1, T):
        X = inc_dict(X, eps * R)
        R = R - eps * grad(X, Y, g)
    
    X = inc_dict(X, eps*R)
    R = R - 0.5 * eps * grad(X, Y, g)

    return X, R

def HMC(X : dict, Y : dict, num_samples : int, T : int, M : tc.tensor, eps : float, g : graph) -> list:
    samples = []

    M_inv = M.inverse()
    R_dist = tc.distributions.MultivariateNormal(tc.zeros(len(M)), M)

    for s in tqdm(range(num_samples)):
        R = R_dist.sample() #get a random momentum vector
        X_new, R_new = leapfrog(deepcopy(X), Y, R, T, eps, g) #integrate the Hamiltonian to get a new sample

        u = tc.distributions.Uniform(0,1).sample()
        ratio = tc.exp(- H(X_new, Y, R_new, M_inv, g) + H(X, Y, R, M_inv, g))
        
        if u < ratio: #MH step
            X = X_new
        
        samples.append(X)

    return samples

def HMC_sampling(program, prog_set="HW3", num_samples=int(1e3), T=10, eps=0.1, M_scale=1, verbose=False, wandb_name=None):
    
    #get the program
    json_prog = './programs/' + prog_set + '/%d_graph.json'%(program)

    with open(json_prog) as f:
        graph_json = json.load(f)

    g = graph(graph_json)

    X, Y = split_latent_obs(g)

    initial_sample = sample_from_joint(g) #get a sample from the joint

    Y = {k : initial_sample[k] for k in initial_sample.keys() if k in Y}
    X = {k : initial_sample[k] for k in initial_sample.keys() if k in X}

    for x in X.keys(): #make the latent floats if needed, and tell torch to require the gradient for the latent vars
        X[x] = (X[x] if tc.is_tensor(X[x]) else tc.tensor(X[x]).type(tc.float))
        X[x].requires_grad = True

    M = tc.eye(len(X)) * M_scale

    chain = HMC(X, Y, num_samples, T, M, eps, g)

    samples = []

    for i, sample in enumerate(chain):
        #initialize an environment
        env = standard_env()
        env.update(sample) #append the sample values

        #evaluate the program based on the environment containing the sample
        ret, _ = Eval(g.Program, {"logW":tc.tensor(0.0)}, env)
        samples.append(ret) #add the evaluation of the sample to the samples list
        if wandb_name is not None : log_sample(ret, i, wandb_name=wandb_name)

    return samples

    





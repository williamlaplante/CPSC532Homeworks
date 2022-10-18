import json
import numpy as np
import torch as tc
from copy import deepcopy
from tqdm import tqdm

from graph_based_sampling import standard_env, graph, evaluate_graph
from evaluation_based_sampling import Eval, Env
from utils import log_sample



def MH_gibbs(program, prog_set="HW3", num_samples=int(1e3), verbose=False, wandb_name = None):
    
    #get the program
    json_prog = './programs/' + prog_set + '/%d_graph.json'%(program)

    with open(json_prog) as f:
        graph_json = json.load(f)

    g = graph(graph_json)

    #get the topologically ordered latent variables in the graph
    ordered_vars = g.topological()
    ordered_latent_vars = [var for var in ordered_vars if var not in g.Graph["Y"].keys()]

    #Get the map of proposal expressions (this is a dict of the form {latent_var : non-evaluated distributions}, i.e. {"sample1": [normal 0 10]})
    Q = {var : g.Graph["P"][var][1] for var in ordered_latent_vars}


    #define the accept function
    def accept(x : str, X : Env, X_p :Env) -> tc.tensor:
        d, _ = Eval(Q[x], {"logW":tc.tensor(0.0)}, X) #Evaluate expression of x (sample ...) given the values in the Env X
        d_p, _ = Eval(Q[x], {"logW":tc.tensor(0.0)}, X_p) #Evaluate expression of x given Env X_p

        log_alpha = d_p.log_prob(X[x]) - d.log_prob(X_p[x])

        Vx = [var for var in ordered_vars if var in g.Graph["A"][x] + [x]] #the children of x, in topological order
        for v in Vx:
            d1, _ = Eval(g.Graph["P"][v][1], {"logW":tc.tensor(0.0)}, X_p)
            d2, _ = Eval(g.Graph["P"][v][1], {"logW":tc.tensor(0.0)}, X)

            if v in g.Graph["Y"].keys() :
                s1 = d1.log_prob(tc.tensor(g.Graph["Y"][v]).float())
                s2 = d2.log_prob(tc.tensor(g.Graph["Y"][v]).float())
            else :
                s1 = d1.log_prob(X_p[v])
                s2 = d2.log_prob(X[v])
            
            log_alpha += s1
            log_alpha -= s2

        return log_alpha.exp()

    #define the Gibb-step function
    def Gibbs_step(X : Env) -> Env:

        for x in ordered_latent_vars:
            d, _ = Eval(Q[x], {"logW":tc.tensor(0.0)}, X) #get the distribution for the latent variable x
            X_p = deepcopy(X) #make a copy of the past state and call it the new state

            X_p[x] = d.sample() #evaluate the latent variable's expression, which is a sample

            alpha = accept(x, X, X_p) #acceptance step

            u = tc.distributions.Uniform(0,1).sample()

            if u < alpha:
                X = deepcopy(X_p) #if X_p is better (alpha large), accept it and replace X

        return X

    #get initial sample X_0:
    _, _, X0 = evaluate_graph(g, verbose=verbose)
    S = range(1, num_samples)

    X = {0 : X0} #Dictionary of environments that need to be evaluated


    for s in tqdm(S): # We get S environments from gibbs steps
        X[s] = Gibbs_step(X[s-1])

    samples = []
    for i, env in enumerate(X.values()):
        sample = Eval(g.Program, {"logW":tc.tensor(0.0)}, env)[0]
        samples.append(sample)
        if wandb_name is not None: log_sample(sample, i, wandb_name=wandb_name)

    #samples = [Eval(g.Program, {"logW":tc.tensor(0.0)}, env)[0] for env in X.values()] #for each environment, we evaluate the program (get the return value)

    return samples

import json
from tkinter import E
import torch as tc
from tqdm import tqdm
import itertools

from graph_based_sampling import standard_env, graph, evaluate_graph
from evaluation_based_sampling import Eval
from utils import log_sample
from distributions import Normal, Dirichlet, Gamma, Categorical, Uniform, HalfUniform

def log_joint(X : dict, Y : dict, g : graph)->tc.tensor:
    
    log_likelihood = tc.tensor(0.0)
    log_prior = tc.tensor(0.0)

    env = standard_env()
    env.update(X)
    env.update(Y)

    ordered_vars = g.topological()
    latent = [var for var in ordered_vars if var not in g.Graph["Y"].keys()]
    observed = [var for var in ordered_vars if var in g.Graph["Y"].keys()]

    for y in observed: #compute the log likelihood
        d, _ = Eval(g.Graph["P"][y][1], {"logW" : tc.tensor(0.0)}, env) #evaluate the dist of y
        log_lik = d.log_prob(tc.tensor(g.Graph["Y"][y]).float()) #compute the likelihood log prob given the above sample
        log_likelihood += log_lik


    for x in latent: #compute the log prior
        d, _ = Eval(g.Graph["P"][x][1], {"logW":tc.tensor(0.0)}, env) #evaluate the dist of x
        log_p = d.log_prob(X[x])
        log_prior += log_p #compute the prior log prob of the above sample 


    log_P = log_likelihood+log_prior #compute the unormalized joint
    
    return log_P


def BBVI(program, prog_set="HW4", num_samples_per_step=100, num_steps=300, learning_rate=2e-1, verbose=False, wandb_name = None):
    
    #get the program
    json_prog = './programs/' + prog_set + '/%d_graph.json'%(program)

    with open(json_prog) as f:
        graph_json = json.load(f)

    g = graph(graph_json)

    ordered_vars = g.topological()
    ordered_latent_vars = [var for var in ordered_vars if var not in g.Graph["Y"].keys()]

    # Q distribution : mean field of N(0,1)
    Q = {}
    for x in ordered_latent_vars:
        dist = g.Graph["P"][x][1][0]
        if dist == "normal": Q[x] = Normal(loc=tc.tensor(0.0), scale=tc.tensor(1.0))
        elif dist == "dirichlet" : Q[x] = Dirichlet(tc.tensor([1., 1., 1.]))
        elif dist == "gamma" : Q[x] = Gamma(tc.tensor(1.0), tc.tensor(1.0))
        elif dist == "discrete" : Q[x] = Categorical(tc.tensor([0.3, 0.3, 0.4]))
        #elif dist == "uniform-continuous" : Q[x] = Normal(loc=tc.tensor(0.0), scale=tc.tensor(1.0))
        elif dist == "uniform-continuous" : Q[x] = HalfUniform(fixed_low=tc.tensor(0.01), high=tc.tensor(10.0))
        else : raise Exception("{} not in the list of distributions.".format(dist))

    ### Variational inference ###

    # Get the parameters -> [mu_0, sigma_0, ..., mu_n, sigma_n] with requires_grad=True on them
    params = list(itertools.chain.from_iterable([Q[x].optim_params() for x in ordered_latent_vars]))
    optimizer = tc.optim.Adam(params, lr=learning_rate)
    
    results = {}
    loss = []

    for i in tqdm(range(num_steps)):

        num_samples = 0
        log_Q = []
        log_P = []

        while num_samples != num_samples_per_step: #pseudo-rejection sampling; if can't evaluate joint because of domain problems, re-sample
            Q_sample = {x : Q[x].sample() for x in Q.keys()}
            
            try:
                log_P_sample = log_joint(Q_sample, g.Graph["Y"], g)
            except ValueError:
                #Occurs when the sequence of samples (latent) {"sample_1" : 0.4, ..., "sample_n":0.2} isn't in the support of the program 
                log_P_sample = -tc.tensor(1e10).float()

            log_Q_sample = tc.stack([Q[x].log_prob(Q_sample[x]) for x in Q_sample.keys()]).sum()
            log_Q.append(log_Q_sample)
            log_P.append(log_P_sample)
            num_samples+=1

        log_P = tc.stack(log_P)
        log_Q = tc.stack(log_Q)

        '''
        # Draw samples from the Q distribution
        Q_samples = [{x : Q[x].sample() for x in Q.keys()} for _ in range(num_samples_per_step)] #of the form [{"x1":0.3, "x2":0.2}, ..., {"x1":0.1, "x2":0.5}]

        # Calculate probability of the Q samples under the Q(X) distribution
        log_Q = tc.stack([tc.stack([Q[x].log_prob(sample[x]) for x in sample.keys()]).sum() for sample in Q_samples])

        assert log_Q.size().numel() == num_samples_per_step, "logQ has the wrong size."#make sure we sum properly

        # Calculate the probability of the Q_samples and observations under the joint P(X, Y) distribution
        log_P = tc.stack([log_joint(Q_sample, g.Graph["Y"], g) for Q_sample in Q_samples])
        '''   
        assert log_Q.shape == log_P.shape, "log Q(x) and log P(X,Y) have a mismatch in size."

        #We compute E(logP(x,y) - logQ(x;params)) over q(x;params), and negate it to get the loss function we want to minimize
        ELBO_loss = -(log_Q*((log_P-log_Q).detach())).mean()
        ELBO_loss.backward()

        # Step with the optimizer
        optimizer.step()
        optimizer.zero_grad() # NOTE: Must zero the gradient after each step!

        # Append results to a list

        results[i] = {x : [p.clone().detach() for p in Q[x].params()] for x in Q.keys()}
        loss.append(ELBO_loss.clone().detach())
    

    loss = tc.stack(loss).float()

    return Q, results, loss

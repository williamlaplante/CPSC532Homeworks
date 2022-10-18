# Standard imports
import torch as tc
from time import time

# Project imports
from evaluation_based_sampling import evaluate_program
from graph_based_sampling import evaluate_graph
from utils import log_sample

def flatten_sample(sample):
    if type(sample) is list: # NOTE: Nasty hack for the output from program 4 of homework 2
        flat_sample = tc.concat([element.flatten() for element in sample])
    else:
        flat_sample = sample
    return flat_sample


def get_sample(ast_or_graph, mode, verbose=False):
    if mode == 'desugar':
        ret, sig, _ = evaluate_program(ast_or_graph, verbose=verbose)
    elif mode == 'graph':
        ret, sig, _ = evaluate_graph(ast_or_graph, verbose=verbose)
    else:
        raise ValueError('Mode not recognised')
    ret = flatten_sample(ret)
    return ret, sig


def prior_samples(ast_or_graph, mode, num_samples, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of samples from the prior of a FOPPL program
    '''
    samples = []
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        sample, _ = get_sample(ast_or_graph, mode, verbose)
        if wandb_name is not None: log_sample(sample, i, wandb_name=wandb_name)
        samples.append(sample)
        if (tmax is not None) and time() > max_time: break
    return samples

def posterior_samples(ast_or_graph, mode, num_samples, tmax=None, wandb_name=None, verbose=False):
    
    samples = []
    weights = []

    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        sample, weight = get_sample(ast_or_graph, mode, verbose)
        if wandb_name is not None: log_sample(sample, i, wandb_name=wandb_name) #MIGHT HAVE TO CHANGE THIS
        samples.append(sample)
        weights.append(weight)

        if (tmax is not None) and time() > max_time: break
    return samples, weights
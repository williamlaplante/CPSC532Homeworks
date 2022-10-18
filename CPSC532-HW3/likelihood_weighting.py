import json
import torch as tc
from tqdm import tqdm

from evaluation_based_sampling import abstract_syntax_tree
from graph_based_sampling import graph
from general_sampling import get_sample
from utils import log_sample


def create_class(ast_or_graph, mode):
    if mode == 'desugar':
        return abstract_syntax_tree(ast_or_graph)
    elif mode == 'graph':
        return graph(ast_or_graph)
    else:
        raise ValueError('Input type not recognised')

def likelihood_weighting(program, mode, prog_set="HW3", num_samples=int(1e3), wandb_name=None, verbose=False):
        '''
        params
        -------
        program : int from 1,...,5 (5 programs in HW3)
        mode : graph or desugar
        '''
        json_prog = './programs/' + prog_set + '/%d_%s.json'%(program, mode)

        with open(json_prog) as f:
            ast_or_graph = json.load(f)
        
        ast_or_graph = create_class(ast_or_graph, mode)

        samples = []
        log_weights = []
        for i in tqdm(range(num_samples)):
            ret, sig = get_sample(ast_or_graph, mode, verbose=verbose)
            samples.append(ret)
            log_weights.append(sig["logW"])

        log_weights = tc.stack(log_weights).type(tc.float)
        samples = tc.stack(samples).type(tc.float)

        weights = tc.exp(log_weights)
        normalized_weights = tc.div(weights, weights.sum())

        if tc.isclose(weights.sum(), tc.tensor([0.0])):
            raise Exception("Likelihood weights are all 0's.")

        idx = tc.distributions.Categorical(normalized_weights).sample(tc.Size([num_samples]))

        resample = []
        for i in idx:
            s = samples[i.item()]
            resample.append(s)
            if wandb_name is not None: log_sample(s, i, wandb_name=wandb_name)

        return resample
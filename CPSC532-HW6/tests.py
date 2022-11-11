# Standard imports
import numpy as np
import torch as tc
from pyrsistent._pmap import PMap
from scipy.stats import kstest, norm, beta, expon
import json

class normalmix():
    def __init__(self, *args):
        self.normals = []
        self.wts = []
        for i in range(len(args)//3):
            self.normals.append(norm(args[3*i+1], args[3*i+2]))
            self.wts.append(args[3*i])
    def cdf(self, arg):
        cdf_vals = []
        for wt, normal in zip(self.wts, self.normals):
            cdf_vals.append(wt*normal.cdf(arg))
        return sum(cdf_vals)


def is_tol(a, b, tol=1e-5):
    if type(a) in [dict, PMap]:
        keys_match = (set(a) == set(b))
        if keys_match:
            for k,v in a.items(): # Check all items
                if not is_tol(v, b[k]): # Recursively check if they match
                    return False 
            return True # Return True if all items match
        else:
            return False # Otherwise False if keys do not match
    else:
        return not tc.any(tc.logical_not(tc.abs((a-b)) < tol))


def run_probabilistic_test(samples, truth):
    samples = tc.stack(samples) # NOTE: Necessary for test 6.
    dists = { # These come from scipy; to compare against
        'normal': norm,
        'beta': beta,
        'exponential': expon,
        'normalmix': normalmix,
        }
    print('Truth:', truth)
    truth_dist = dists[truth[0]](*truth[1:])
    _, p_val = kstest(np.array(samples), truth_dist.cdf) # NOTE: kstest is from scipy.stats
    return p_val


def load_truth(path): 
    # TODO: This is very hacky and will break for anything complicated
    with open(path) as f:
        truth = json.load(f)
    if type(truth) is list:
        if type(truth[0]) is str:
            truth = tuple(truth)
        else:
            truth = tc.tensor(truth)
    elif type(truth) is dict:
        truth = {float(k):v for k,v in truth.items()} # TODO: This will NOT work for nested dictionaries
    return truth



# Standard imports
import torch as tc
from pyrsistent import pmap
from time import time
import sys
# Project imports
from primitives import primitives

# Parameters
run_name = 'start'

class Env(dict):
    'An environment: a dict of {var: val} pairs, with an outer environment'
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    
    def find(self, var):
        'Find the innermost Env where var appears'
        if var in self:
            result = self
        else:
            if self.outer is None:
                # print('Not found in any environment:', var)
                raise ValueError('Outer limit of environment reached')
            else:
                result = self.outer.find(var)
        return result


class Procedure(object):
    'A user-defined HOPPL procedure'
    def __init__(self, params:list, body:list, sig:dict, env:Env):
        self.params, self.body, self.sig, self.env = params, body, sig, env
    def __call__(self, *args):
        return eval(self.body, self.sig, Env(self.params, args, self.env))


def standard_env():
    'An environment with some standard procedures'
    env = Env()
    env.update(primitives)
    return env


def eval(e, sig:dict, env:Env, verbose=False):
    '''
    The eval routine
    @params
        e: expression
        sig: side-effects
        env: environment
    '''

    if isinstance(e, bool):
        return tc.tensor(int(e)).float()
    
    elif isinstance(e, float) or isinstance(e,int): #case : constant
        return tc.tensor(e).float()
    
    elif isinstance(e, str): #case : variable or procedure (anything in the env)
        if e[0] == "\"":  # strings have double, double quotes
            return e[1:-1]
        if e[0:4] == 'addr':
            return e[4:]

        return env.find(e)[e]

            
    op, *args = e
    
    if op == 'sample' or op=="sample*":
        addr = eval(args[0], sig, env)
        d = eval(e[2], sig, env)
        k = eval(args[-1], sig, env)

        new_sig = pmap({
            "type" : "sample",
            "address": addr,
            "logW": sig["logW"],
        })
        return k, [d.sample()], new_sig
    
    elif op == 'observe' or op=="observe*":
        addr = eval(args[0], sig, env)
        dist = eval(args[1], sig, env)
        obs = eval(args[2], sig, env)
        k = eval(args[-1], sig, env)
        
        new_sig = pmap({
            "type" : "observe",
            "address" : addr,
            "logW" : sig["logW"] + dist.log_prob(obs),
        })

        return k, [obs], new_sig

    elif op == 'if': # conditional
        (test, conseq, alt) = args
        exp = (conseq if eval(test, sig, env) else alt)
        return eval(exp, sig, env)
    
    elif op == 'fn':         # procedure
        (params, body) = args
        return Procedure(params=params, body=body, sig=sig, env=env)
    
    else:                        # procedure call
        proc = eval(op, sig, env)

        vals = []
        
        for arg in args:
            val = eval(arg, sig, env)
            vals.append(val)
        
        if callable(proc):
            return proc(*vals)
        else:
            raise Exception("{} is not callable (User Exception).".format(proc))


def evaluate(ast:dict, sig={}, verbose=False):
    '''
    Evaluate a HOPPL program as desugared by daphne
    Args:
        ast: abstract syntax tree
    Returns: The return value of the program
    '''
    env = standard_env()
    output = lambda x: x # Identity function, so that output value is identical to output
    sig = {"logW" : tc.tensor(0.0)}
    exp = eval(ast, sig, env, verbose)(run_name, output) # NOTE: Must run as function with a continuation
    while type(exp) is tuple: # If there are continuations the exp will be a tuple and a re-evaluation needs to occur
        func, args, sig = exp
        exp = func(*args)
    return exp, sig
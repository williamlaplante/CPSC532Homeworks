# Standard imports
import torch as tc

# Project imports
from primitives import primitives # NOTE: Import and use this!

class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, env, sigma):
        self.parms, self.body, self.env, self.sigma = parms, body, env, sigma
    def __call__(self, *args): 
        e, sigma = Eval(self.body, self.sigma, Env(self.parms, args, self.env))
        return e


def standard_env() -> Env:
    "An environment with some Scheme standard procedures."
    env = Env()
    env.update(primitives)
    return env

class abstract_syntax_tree:
    def __init__(self, ast_json):
        self.functions = ast_json[:-1]
        self.program = ast_json[-1]

def Eval(x, sigma, env):
    
    if isinstance(x, bool):
        return tc.tensor(int(x)).float(), sigma
    
    elif isinstance(x, float) or isinstance(x,int): #case : constant
        return tc.tensor(x).float(), sigma #return a constant tensor
    
    elif isinstance(x, str): #case : variable or procedure (anything in the env)
        return env.find(x)[x], sigma #return the variable/procedure in the environment
    
    op, *args = x
    
    if op == 'sample' or op=="sample*":
        d, sigma = Eval(x[1], sigma, env)
        return d.sample(), sigma
    
    elif op == 'observe' or op=="observe*":
        dist, sigma = Eval(args[0], sigma, env)
        val, sigma = Eval(args[1], sigma, env)
        sigma = sigma + dist.log_prob(val)
        return val, sigma

    elif op == 'if':             # conditional
        (test, conseq, alt) = args
        exp = (conseq if Eval(test, sigma, env)[0] else alt)
        return Eval(exp, sigma, env)
    
    elif op == 'let':         # definition
        (symbol, exp) = args[0]
        env[symbol], sigma = Eval(exp, sigma, env)
        return Eval(args[1], sigma, env)
    
    elif op == 'defn':         # procedure
        (fname, parms, body) = args
        env[fname] = Procedure(parms, body, env, sigma)
    
    else:                        # procedure call
        proc, sigma = Eval(op, sigma, env)

        vals = []
        for arg in args:
            val, sigma = Eval(arg, sigma, env)
            vals.append(val)

        return proc(*vals), sigma


def evaluate_program(ast, verbose=False):
    #initialize global environment and sigma (unnormalized weight)
    global_env = standard_env()
    sigma = tc.tensor(0).float()

    #appending the defined function from the ast program to the environment
    for function in ast.functions:
        Eval(function, sigma, global_env)

    #evaluating the ast program
    e, sigma = Eval(ast.program, sigma, global_env)

    return e, sigma, global_env
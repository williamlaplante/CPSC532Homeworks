# Standard imports
import torch as tc

# Project imports
from primitives import primitives


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
        return Eval(self.body, self.sigma, Env(self.parms, args, self.env))

class abstract_syntax_tree:
    def __init__(self, ast_json):
        self.functions = ast_json[:-1]
        self.program = ast_json[-1]


def standard_env() -> Env:
    "An environment with some Scheme standard procedures."
    env = Env()
    env.update(primitives)
    return env


def Eval(x, sigma, env):
    
    if isinstance(x, bool):
        return tc.tensor(int(x)).float(), sigma
    
    elif isinstance(x, float) or isinstance(x,int): #case : constant
        return tc.tensor(x).float(), sigma #return a constant tensor
    
    elif isinstance(x, str): #case : variable or procedure (anything in the env)
        try :
            e = env.find(x)[x]
        except:
            raise Exception("Couldn't find " + x + " in the environment.")
        
        return e, sigma #return the variable/procedure in the environment
    
    op, *args = x
    
    if op == 'sample' or op=="sample*":
        dist, sigma = Eval(x[1], sigma, env)
        sample = dist.sample()
        sigma["logP"] += dist.log_prob(sample)
        sigma["logJoint"] += dist.log_prob(sample)
        return sample, sigma
    
    elif op == 'observe' or op=="observe*":
        dist, sigma = Eval(args[0], sigma, env)
        obs, sigma = Eval(args[1], sigma, env)
        sigma["logW"] += dist.log_prob(obs)
        sigma["logJoint"] += dist.log_prob(obs)
        sample = dist.sample()
        lik_sample = dist.log_prob(sample)
        sigma["lik_samples"].append({"obs" : obs, "dist" : dist, "y" : sample, "y_lik" : lik_sample})
        return obs, sigma

    elif op == 'if': # conditional
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
        
        if isinstance(proc, Procedure):
            return proc(*vals) #case where proc is a Procedure function -> returns both value and sigma
        else:
            return proc(*vals), sigma #case where proc its a primitive function (tc.add) -> only return the value (no sigma)
        


def evaluate_program(ast, verbose=False):
    #initialize global environment and sigma (unnormalized weight)
    global_env = standard_env()
    sigma = {"logW" : tc.tensor(0.0), "logP" : tc.tensor(0.0), "logJoint":tc.tensor(0.0), "lik_samples":[]}

    #appending the defined function from the ast program to the environment
    for function in ast.functions:
        Eval(function, sigma, global_env)

    #evaluating the ast program
    e, sigma = Eval(ast.program, sigma, global_env)

    return e, sigma, global_env
import torch as tc

def _and(first, second):
    if first and second:
        return tc.tensor([True])
    else:
        return tc.tensor([False])

def _or(first, second):
    if first or second:
        return tc.tensor([True])
    else:
        return tc.tensor([False])


def equal(first, second):
    return tc.tensor(tc.equal(first, second))

def matrepmat(vecname, val , length):
    return tc.full((length.int(),), val)

def get(vec, pos):
    if tc.is_tensor(pos):
        return vec[int(pos.item())]
    else:
        return vec[int(pos)]

def put(vec, pos, val):
    if tc.is_tensor(int(pos)):
        vec[int(pos.item())] = val
    else:
        vec[int(pos)] = val
    return vec

def first(vec):
    return vec[0]

def second(vec):
    return vec[1]

def rest(vec):
    return vec[1:]

def last(vec):
    return vec[-1]

def append(vec, val):
    return tc.cat((vec, tc.tensor([val])))

def vector(*x):
    # NOTE: This must support both lists and vectors
    try:
        result = tc.stack(x)
    except:
        result = list(x)
    return result

def hashmap(*x):
    _keys = [key for key in x[0::2]]
    keys = []
    for key in _keys: # Torch tensors cannot be dictionary keys, so convert here
        if type(key) is tc.Tensor: key = float(key)
        keys.append(key)
    values = [value for value in x[1::2]]
    return dict(zip(keys, values))

class Dirac():
    def __init__(self, x0, atol=10e-1):
        self.x0 = x0
        self.atol = atol
        self.inf = 1e10

    def log_prob(self, x):
        if tc.isclose(x, self.x0, rtol=10e-5, atol=self.atol):
            return tc.tensor(0.0)
        else : 
            return tc.tensor(-self.inf).float()

    def sample(self):
        return self.x0


# Primative function dictionary
    

primitives = {

    # Comparisons
    '<': tc.lt,
    '<=': tc.le,
    'and': _and,
    'or': _or,

    # Math
    '+': tc.add,
    '-': tc.sub,
    '*': tc.mul,
    '/': tc.div,
    '=': equal,
    'sqrt': tc.sqrt,


    # Containers
    'vector': vector,
    'hash-map': hashmap,
    'get' : get,
    'put' : put,
    'first' : first,
    'second': second,
    'last' : last,
    'append' : append,
    'rest' : rest,
    # ...

    # Matrices
    'mat-mul': tc.matmul,
    'mat-transpose': tc.t,
    'mat-tanh' : tc.tanh,
    'mat-add' : tc.add,
    'mat-repmat' : matrepmat,
    # ...

    # Distributions
    'normal': tc.distributions.Normal,
    'beta': tc.distributions.beta.Beta,
    'exponential': tc.distributions.Exponential,
    'uniform-continuous': tc.distributions.Uniform,
    'discrete' : tc.distributions.Categorical,
    'dirichlet' : tc.distributions.Dirichlet,
    'gamma' : tc.distributions.Gamma,
    'flip' : tc.distributions.Bernoulli,
    'dirac': Dirac
    
}
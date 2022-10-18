import torch as tc

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

# Primative function dictionary
# NOTE: You should complete this
primitives = {

    # Comparisons
    '<': tc.lt,
    '<=': tc.le,
    # ...

    # Math
    '+': tc.add,
    '-': tc.sub,
    '*': tc.mul,
    '/': tc.div,
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
    'discrete' : tc.distributions.Categorical
    # ...

}
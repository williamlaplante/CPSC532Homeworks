# Standard imports
import torch as tc
from pyrsistent import pmap, pvector

def isempty(vec):
    if tc.is_tensor(vec): #check length of tensor
        return (vec.numel()==0)
    else: #check length of pvector or hashmap
        return (len(vec)==0)

def get(vec, pos):
    if tc.is_tensor(pos): #if position is tensor (numeric)
        return vec[int(pos.item())] #
    elif isinstance(pos, float) or isinstance(pos, int): #if position is numeric
        return vec[int(pos)]
    elif isinstance(pos, str): #if position is string, then we have a dict, or hashmap
        return vec[pos]
    else: #any other cases should raise an error
        raise Exception("The position {} has invalid data type.".format(pos))

def put(vec, pos, val):

    #dealing with the position first
    if tc.is_tensor(pos):
        pos = int(pos.item())
    elif isinstance(pos, float) or isinstance(pos, int):
        pos = int(pos)
    elif isinstance(pos, str):
        pass
    else:
        raise Exception("The position {} has invalid data type.".format(pos))

    #dealing with the vector
    if tc.is_tensor(vec):
        newvec = vec.clone() #creating a copy of the input vector
        newvec[pos] = val
        return newvec #returning a copy of the old vector, with the value val at position pos

    else:
        return vec.set(pos, val) #for both pmap and pvector, .set() returns a new vector/map


def peek(vec):
    return vec[0]

def first(vec):
    return vec[0]

def second(vec):
    return vec[1]

def rest(vec):
    return vec[1:]

def last(vec):
    return vec[-1]

def append(vec, val):
    if not tc.is_tensor(vec):
        vec = tc.tensor(vec)

    return tc.cat([vec, tc.tensor([val])]) #tc.cat returns a new tensor -> persistent

def conj(vec, val):
    if not tc.is_tensor(vec):
        vec = tc.tensor(vec)

    return tc.cat([tc.tensor([val]), vec]) #tc.cat returns a new tensor -> safe


def vector(*x):
    # This needs to support both lists and vectors
    try:
        result = tc.stack(x) # NOTE: Important to use stack rather than tc.tensor
    except:
        result = pvector(x)
    return result


def hashmap(*x):
    # This is a dictionary
    keys, values = x[0::2], x[1::2]
    checked_keys = []
    for key in keys: # Torch tensors cannot be dictionary keys, so convert here
        if type(key) is tc.Tensor: key = float(key)
        checked_keys.append(key)
    dictionary = dict(zip(checked_keys, values))
    hashmap = pmap(dictionary)
    return hashmap


def push_address(*x):
    # Concatenate two addresses to produce a new, unique address
    previous_address, current_addreess = x[0], x[1]
    new_address = previous_address+'-'+current_addreess
    return new_address


# Primative function dictionary
# NOTE: Fill this in
primitives = {

    # HOPPL
    'push-address' : push_address,

    # Comparisons
    '<': lambda *x: tc.lt(*x[1:]),
    '>': lambda *x: tc.gt(*x[1:]),

    # Maths
    '+': lambda *x: tc.add(*x[1:]),
    '-': lambda *x: tc.sub(*x[1:]),
    'sqrt': lambda *x: tc.sqrt(*x[1:]),
    '*': lambda *x: tc.multiply(*x[1:]),
    '/': lambda *x: tc.div(*x[1:]),
    'log': lambda *x: tc.log(*x[1:]),

    # Containers
    'vector': lambda *x: vector(*x[1:]),
    'hash-map': lambda *x: hashmap(*x[1:]),
    'get': lambda *x: get(*x[1:]),
    'put': lambda *x: put(*x[1:]),
    'first': lambda *x: first(*x[1:]),
    'second': lambda *x: second(*x[1:]),
    'rest': lambda *x: rest(*x[1:]),
    'last': lambda *x: last(*x[1:]),
    'append': lambda *x: append(*x[1:]),
    'empty?': lambda *x: isempty(*x[1:]),
    'conj': lambda *x: conj(*x[1:]),
    'peek': lambda *x: peek(*x[1:]),


    # Matrices
    'mat-transpose': lambda *x: tc.transpose(*x[1:], 0, 1),

    # Distributions
    'normal': lambda *x: tc.distributions.Normal(*x[1:]),
    'beta': lambda *x: tc.distributions.Beta(*x[1:]),
    'exponential': lambda *x: tc.distributions.Exponential(*x[1:]),
    'uniform-continuous': lambda *x: tc.distributions.Uniform(*x[1:]),
    'flip': lambda *x: tc.distributions.Bernoulli(*x[1:]),
    'discrete': lambda *x: tc.distributions.Categorical(*x[1:]),


}
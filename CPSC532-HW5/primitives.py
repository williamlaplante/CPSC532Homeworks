# Standard imports
import torch as tc
from pyrsistent import pmap, plist

def vector(*x):
    # This needs to support both lists and vectors
    try:
        result = tc.stack(x) # NOTE: Important to use stack rather than tc.tensor
    except:
        result = plist(x)
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

    # Maths
    '+': lambda *x: tc.add(*x[1:]),

    # Containers
    'vector': lambda *x: vector(*x[1:]),
    'hash-map': lambda *x: hashmap(*x[1:]),

    # Matrices
    'mat-transpose': lambda *x: tc.transpose(*x[1:], 0, 1),

    # Distributions
    'normal': lambda *x: tc.distributions.Normal(*x[1:]),

}
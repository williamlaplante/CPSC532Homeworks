# Standard imports
import torch as tc
from graphlib import TopologicalSorter # NOTE: This is useful

# Project imports
from evaluation_based_sampling import Eval, standard_env
#from primitives import primitives # NOTE: Otherwise you could import and use this again!

def flatten(mylist):
    '''
    Functions to flatten any nested list
    '''
    newlist = []
    
    for el in mylist:
        if type(el) in [int, float, str]:
            newlist.append(el)
        else:
            newlist+=flatten(el)
    
    return newlist


class graph:
    def __init__(self, graph_json):
        self.json = graph_json
        self.foo, self.Graph, self.program = graph_json
        # NOTE: You need to write this!

    def get_DAG(self):
        dag = {}
        for var in self.Graph["V"]:
            dag[var] = self.find_parents(var)
        return dag
    
    def find_parents(self, v):
        v_expr = flatten(self.Graph["P"][v])
        parents = set()
        for v in self.Graph["V"]:
            if v in v_expr:
                parents.add(v)
        return parents
    
    def topological(self):
        return list(TopologicalSorter(self.get_DAG()).static_order())
    
def evaluate_graph(graph, verbose=False):
    #initialize global environment and sigma (unnormalized weight)
    global_env = standard_env()
    sigma = 0

    #append functions to global environment
    for key in graph.topological():
        expr = graph.Graph["P"][key]
        global_env[key], _ = Eval(expr, sigma, global_env)

    #evalute the program
    e, sigma = Eval(graph.program, sigma, global_env)

    return e, sigma, global_env
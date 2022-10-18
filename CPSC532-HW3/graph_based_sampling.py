# Standard imports
import torch as tc
from graphlib import TopologicalSorter

# Project imports
from evaluation_based_sampling import Eval, standard_env


class graph:
    def __init__(self, graph_json):
        self.json = graph_json
        self.Functions, self.Graph, self.Program = graph_json

    def topological(self):
        order = list(TopologicalSorter(self.Graph["A"]).static_order())
        order.reverse()
        if order:
            return order 
        else:
            return self.Graph["V"] #case where there is only one vertex in the graph
    
    
def evaluate_graph(graph, verbose=False):
    #initialize global environment and sigma (unnormalized weight)
    global_env = standard_env()
    sigma = {"logW":tc.tensor(0.0)}

    #append functions to global environment
    for key in graph.topological():
        expr = graph.Graph["P"][key]
        global_env[key], _ = Eval(expr, sigma, global_env)

    #evalute the program
    e, sigma = Eval(graph.Program, sigma, global_env)

    return e, sigma, global_env
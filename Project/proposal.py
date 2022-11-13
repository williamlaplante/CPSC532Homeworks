import torch as tc
from torch import nn
from torch.distributions.constraint_registry import transform_to
from distributions import Normal
from itertools import chain


#Typical NN. Takes input dimension (conditional), output dimension (# conditional dist parameters)
class NeuralNetwork(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NeuralNetwork, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, dim_out),
        )

    def forward(self, x):
        return self.net(x)

#The proposal, P(X | Y) = P(x_1 | Y, x_2,...x_n)P(x_2 | Y, x_3, ..., x_n)...P(x_n | Y)
#having the inverse dependencies allows us the remove some of the conditioning (and thus less parameter estimations). But it's not mandatory.
class Proposal():
    def __init__(self, g):
        super().__init__()
        self.graph = g
        self.ordered_vars = g.reverse_topological()
        self.ordered_latent_vars = [var for var in self.ordered_vars if var not in g.Graph["Y"].keys()]
        self.n = len(g.Graph["Y"])

        self.distributions = {var : Normal for var in self.ordered_latent_vars}
        self.constraints = {var : self.distributions[var].arg_constraints for var in self.ordered_latent_vars}
        self.links = {var : NeuralNetwork(dim_in = self.n + i, dim_out = self.distributions[var].NUM_PARAMS) for i, var in enumerate(self.ordered_latent_vars)}
        return
    
    def sample(self, y : tc.Tensor) -> dict:
        _sample = {}
        s = y
        with tc.no_grad():
            for latent in self.ordered_latent_vars:
                #get output of NN. Should be of dimension == # dist params
                nn_out = self.links[latent].forward(s)
                #iterate over nn output and the associated support for each element (ex : mu, sigma for a Normal, hence sigma has (0,inf) support) and apply appropriate transform
                dist_params = [transform_to(constr)(dist_param) for constr, dist_param in zip(self.distributions[latent].arg_constraints.values(), nn_out)] #apply constraint transform
                #sample from the conditional distribution
                s = self.distributions[latent](*dist_params).sample()
                _sample.update({latent : s})

        return _sample
    
    def log_prob(self, sample : tc.Tensor, y : tc.Tensor) -> tc.Tensor:
        lik = tc.tensor(0.0)
        val = y
        #we need to flip sample, since it comes in topological order and we iterate over the reverse order.
        for latent, x in zip(self.ordered_latent_vars, tc.flip(sample, [0])):
            nn_out = self.links[latent].forward(val)
            dist_params = [transform_to(constr)(dist_param) for constr, dist_param in zip(self.distributions[latent].arg_constraints.values(), nn_out)]
            lik += self.distributions[latent](*dist_params).log_prob(x)
            val = x
        return lik
    
    def get_params(self):
        params = []
        for latent, neuralnet in self.links.items():
            params += [*neuralnet.parameters()]
        return params


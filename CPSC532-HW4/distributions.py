import torch as tc
from utils import softplus, inverse_softplus

# Choose scaling function to map positive-definite quantities onto the real line and visa-versa
# NOTE: Seems not to matter too much
scaling_function = 'softplus'
#scaling_function = 'exponential'

# Scaling functions
if scaling_function == 'softplus':
    positive_function = softplus        # Scales the entire real line to a positive-definite number
    positive_inverse = inverse_softplus # Scales a positive-definite number to the entire real line
elif scaling_function == 'exponential':
    positive_function = tc.exp
    positive_inverse = tc.log
else:
    raise ValueError('Scaling function not recognised')

class Normal(tc.distributions.Normal):
    
    def __init__(self, loc, scale): # Define an optimizable scale that exists on the entire real line
        self.optim_scale = positive_inverse(scale)
        super().__init__(loc, scale)

    def params(self): # Return a list of standard parameters of the distribution
        return [self.loc, self.scale]

    def optim_params(self): # Return a list of (optimizeable) parameters of the distribution
        return [self.loc.requires_grad_(), self.optim_scale.requires_grad_()]

    def log_prob(self, x): # This overwrites the default log_prob function and updates scale
        self.scale = positive_function(self.optim_scale) # Needed to carry the gradient through
        return super().log_prob(x)

class HalfUniform(tc.distributions.Uniform):
    def __init__(self, fixed_low, high):
        self.fixed_low = fixed_low
        super().__init__(low = fixed_low, high = high)
    
    def params(self):
        return [self.fixed_low, self.high]

    def optim_params(self):
        return [self.fixed_low, self.high.requires_grad_()]
    
    def log_prob(self, x):
        return super().log_prob(x)

        
class Uniform(tc.distributions.Uniform):
    
    def __init__(self, low, high): # Define an optimizable scale that exists on the entire real line    
        super().__init__(low, high)

    def params(self): # Return a list of standard parameters of the distribution
        return [self.low, self.high]

    def optim_params(self): # Return a list of (optimizeable) parameters of the distribution
        return [self.low.requires_grad_(), self.high.requires_grad_()]

    def log_prob(self, x): # This overwrites the default log_prob function and updates scale
        return super().log_prob(x)


class Gamma(tc.distributions.Gamma):
    
    def __init__(self, concentration, rate):
        self.optim_concentration = positive_inverse(concentration)
        self.optim_rate = positive_inverse(rate)
        super().__init__(concentration, rate)

    def params(self):
        return [self.concentration, self.rate]

    def optim_params(self):
        return [self.optim_concentration.requires_grad_(), self.optim_rate.requires_grad_()]

    def log_prob(self, x):
        self.concentration, self.rate = positive_function(self.optim_concentration), positive_function(self.optim_rate)
        return super().log_prob(x)


class Exponential(tc.distributions.Exponential):
    
    def __init__(self, rate):
        self.optim_rate = positive_inverse(rate)
        super().__init__(rate)

    def params(self):
        return [self.rate]

    def optim_params(self):
        return [self.optim_rate.requires_grad_()]

    def log_prob(self, x):
        self.rate = positive_function(self.optim_rate)
        return super().log_prob(x)


class Beta(tc.distributions.Beta):
    
    def __init__(self, concentration1, concentration0):
        self.optim_concentration1 = positive_inverse(concentration1)
        self.optim_concentration0 = positive_inverse(concentration0)
        super().__init__(concentration1, concentration0)

    def params(self):
        return [self.concentration1, self.concentration0]

    def optim_params(self):
        return [self.optim_concentration1.requires_grad_(), self.optim_concentration0.requires_grad_()]

    def log_prob(self, x):
        self.concentration1 = positive_function(self.optim_concentration1)
        self.concentration0 = positive_function(self.optim_concentration0)
        return super().log_prob(x)


class Dirichlet(tc.distributions.Dirichlet):
    
    def __init__(self, concentration):
        self.optim_concentration = positive_inverse(concentration)
        super().__init__(concentration)

    def params(self):
        return [self.concentration]
    
    def optim_params(self):
        return [self.optim_concentration.requires_grad_()]

    def log_prob(self, x):
        self.concentration = positive_function(self.optim_concentration)
        return -1/(super().log_prob(x).exp())


class Bernoulli(tc.distributions.Bernoulli):
    
    def __init__(self, probs=None, logits=None):
        if logits is None and probs is None:
            raise ValueError('Set either probs or logits')
        elif logits is None:
            if type(probs) is float:
                probs = tc.tensor(probs)
            logits = tc.log(probs/(1.-probs)) # NOTE: This will fail if probs = 0
        super().__init__(logits=logits)

    def params(self):
        return [self.logits]

    def optim_params(self):
        return [self.logits.requires_grad_()]


class Categorical(tc.distributions.Categorical):

    def __init__(self, probs=None, logits=None):
        if (probs is None) and (logits is None):
            raise ValueError('Either `probs` or `logits` must be specified, but not both')
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError('`probs` parameter must be at least one-dimensional')
            probs = probs/probs.sum(-1, keepdim=True)
            logits = tc.distributions.utils.probs_to_logits(probs)
        else:
            if logits.dim() < 1:
                raise ValueError('`logits` parameter must be at least one-dimensional')
            logits = logits-logits.logsumexp(dim=-1, keepdim=True) # Normalize
        super().__init__(logits=logits)
        self.logits = logits
        self._param = self.logits

    def params(self):
        return [self.logits]

    def optim_params(self):
        return [self.logits.requires_grad_()]




if __name__ == '__main__':

    # Here is a stand-alone example program that uses BBVI to learn the parameters
    # of a Gaussian posterior distribution (Q1 of the homework)

    ### Parameters ###

    # Seed
    seed = 123

    # Variational inference
    num_samples_per_step = 100
    num_steps = 300
    learning_rate = 1e-1

    # Prior parameters
    prior_loc = tc.tensor(1.)
    prior_scale = tc.sqrt(tc.tensor(5.))

    # Likelihood parameters
    likelihood_scale = tc.sqrt(tc.tensor(2.))

    # Observations
    ys = tc.tensor([8., 9.])

    ### ###

    ### Calculations ###

    # Random seed
    tc.manual_seed(seed)

    # Stepping
    print('Number of steps in calculation:', num_steps)

    # Prior distribtuion
    prior = tc.distributions.Normal(prior_loc, prior_scale)
    print('Prior:', prior)

    # Q distribution
    Q_loc, Q_scale = tc.clone(prior_loc), tc.clone(prior_scale) # NOTE: Clone is necessary here
    Q = Normal(Q_loc, Q_scale)
    Q.loc, Q_optim_scale = Q.optim_params() # Get the optimisable parameters for the distribution
    print('Q distribution:', Q)
    print('Q location, scale, optim scale:', Q.loc, Q.scale, Q.optim_scale)
    print()

    ### ###

    ### Variational inference ###

    # Loop over steps
    locs = [Q_loc.clone().detach()]; scales = [Q_scale.clone().detach()]
    optimizer = tc.optim.Adam([Q_loc, Q_optim_scale], lr=learning_rate)
    print('Initial Q:', Q)
    for _ in range(num_steps):

        # Draw samples from the Q distribution
        Q_samples = Q.sample(sample_shape=(num_samples_per_step,))

        # Calculate probability of the Q samples under the Q(X) distribution
        log_Q = Q.log_prob(Q_samples)

        # Calculate the probability of the Q_samples and observations under the joint P(X, Y) distribution
        log_likelihood = 0.
        for y in ys:
            log_likelihood += tc.distributions.Normal(loc=Q_samples, scale=likelihood_scale).log_prob(y)
        log_prior = prior.log_prob(Q_samples)
        log_P = log_likelihood+log_prior

        # Calculate the log importance weights of the Q samples and obervations
        log_W = log_P-log_Q

        # Calculate the thing related to the ELBO
        # NOTE: Must detach part of the calculation here!
        ELBO_loss = -(log_Q*(log_W.detach())).mean()
        ELBO_loss.backward()
        print("ELBO loss" + str(ELBO_loss.clone().detach()))
        # Step with the optimizer
        optimizer.step()
        print('Q:', Q)
        optimizer.zero_grad() # NOTE: Must zero the gradient after each step!

        # Append results to a list
        locs.append(Q_loc.clone().detach()); scales.append(Q_scale.clone().detach())

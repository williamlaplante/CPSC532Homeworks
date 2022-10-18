import torch
import torch.distributions as dist
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import numpy as np
import random

##UBC CS532W H1 Q3


##first define the probability distributions as defined in the exercise:

#use 0 as false, 1 as true
#you should use type hinting throughout, and, you should familiarize
#yourself with PyTorch tenors and distributions
def p_C(c:torch.Tensor)->torch.Tensor:
    probs = torch.tensor([0.5,0.5])
    d = dist.Categorical(probs)
    return torch.exp(d.log_prob(c))


def p_S_given_C(s: torch.Tensor,c: torch.Tensor) -> torch.Tensor:
    probs = torch.tensor([[0.5, 0.9], [0.5, 0.1]])
    d = dist.Categorical(probs.t())
    lp = d.log_prob(s)[c.detach()] 
    return torch.exp(lp)


def p_R_given_C(r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    probs = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    d = dist.Categorical(probs.t())
    lp = d.log_prob(r)[c.detach()]  
    return torch.exp(lp)

def p_W_given_S_R(w: torch.Tensor, s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    probs = torch.tensor([
        [[1.0, 0.1], [0.1, 0.01]],  # w = False
        [[0.0, 0.9], [0.9, 0.99]],  # w = True
    ])
    return probs[w.detach(), s.detach(), r.detach()]

# we will be using hydra throughout the course to control configurations
# all arguments are command-line overrideable and you should definitely
# look at the contents of ./conf/config.yaml 
@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # in general you should use logging instead of simply printing
    # to the terminal
    log = logging.getLogger(__name__)

    # this is how you get configuration settings from the yaml via hydra
    wandb_entity = cfg['wandb_entity']
    wandb_project = cfg['wandb_project']
    wandb_logging = cfg['wandb_logging']
    seed = cfg['seed']

    # you should always control the seed in pseudo-random experiments
    # but be aware that GPU computation is not determinitistic 
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # you should let wandb log your code so that it is automatically turned-in
    # leaving the wandb_project set as it is in the yaml will let you "collaborate"
    # with others in the class working on the same project
    if wandb_logging:
        # set up wandb
        wandb.init(project=wandb_project, entity=wandb_entity)
        wandb.run.log_code(".")
        wandb.config.update(OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ))


    ##1. enumeration and conditioning:

    ## compute joint:
    p = torch.zeros((2,2,2,2)) #c,s,r,w
    for c in range(2):
        for s in range(2):
            for r in range(2):
                for w in range(2):
                    p[c, s, r, w] = p_C(torch.tensor(c))*p_S_given_C(torch.tensor(s), torch.tensor(c))*p_R_given_C(torch.tensor(r), torch.tensor(c))*p_W_given_S_R(torch.tensor(w), torch.tensor(s), torch.tensor(r))
    
    # NOTE: Fill this in...
    p_C_given_W = p[:,:,:,1].sum(axis=[1,2])/p[:,:,:,1].sum(axis=[0,1,2]) # p(C, W)/p(W)

    log.info('#1 (enum and cond) : There is a {:.2f}% chance it is cloudy given the grass is wet'.format(p_C_given_W[1]*100))

    if wandb_logging:
        wandb.log({'prob_cloudy_given_wet_grass': {
                  'enum and cond': p_C_given_W[1]*100}})

    # things to think about here: when can you enumerate?
    # we can enumerate for discrete and finite sample space
    # is the computational complexity of enumeration?
    #the complexity is proportional to the sample space size
    #obviously when things get complicated its a nightmare to deal with

    ##2. ancestral sampling and rejection:
    num_samples = 10000
    samples = torch.zeros(num_samples) #P(C, W)
    rejections = 0 #to compute P(W)
    i = 0

    # NOTE: Fill this in

    while i < num_samples:

        c = dist.Categorical(torch.tensor([0.5,0.5])).sample().item() #p(c)

        if c==0:
            r = dist.Categorical(torch.tensor([0.8, 0.2])).sample().item() #p(r|c=0)
            s = dist.Categorical(torch.tensor([0.5, 0.5])).sample().item() #p(s|c=0)

        elif c==1:
            r = dist.Categorical(torch.tensor([0.2, 0.8])).sample().item() #p(r|c=1)
            s = dist.Categorical(torch.tensor([0.9, 0.1])).sample().item() #p(s|c=1)


        if r+s==1:
            w = dist.Categorical(torch.tensor([0.1, 0.9])).sample().item() #p(w|r+s=1)

        elif r+s==2:
            w = dist.Categorical(torch.tensor([0.01, 0.99])).sample().item() #p(w|r+s=2)

        else:
            w=0

        if w==0:
            rejections+=1
            continue

        if w==1 & c==1:
            samples[i]=1

        i+=1

    log.info('#2 (ancestral + reject) : The chance of it being cloudy given the grass is wet is {:.2f}%'.format(samples.mean()*100))
    log.info('#2 (ancestral + reject) : {:.2f}% of the total samples were rejected'.format(
        100*rejections/float(num_samples+rejections)))

    # things to think about here: when will rejection sampling be efficient or 
    # even possible?  can you rejection sample if the conditioning event is
    # measure zero?  what if the conditioning event is extremely unlikely?

    if wandb_logging:
        wandb.log({'prob_cloudy_given_wet_grass': {
                  'ancestral + reject': samples.mean()*100}})

    #3: Gibbs
    # we can use the joint above to condition on the variables, to create the needed
    # conditional distributions:

    #we can calculate p(R|C,S,W) and p(S|C,R,W) from the joint, dividing by the right marginal distribution
    #indexing is [c,s,r,w]
    p_R_given_C_S_W = p/p.sum(axis=2, keepdims=True)
    p_S_given_C_R_W = p/p.sum(axis=1, keepdims=True)


    # but for C given R,S,W, there is a 0 in the joint (0/0), arising from p(W|S,R)
    # but since p(W|S,R) does not depend on C, we can factor it out:
    #p(C | R, S) = p(R,S,C)/(int_C (p(R,S,C)))

    #first create p(R,S,C):
    p_C_S_R = torch.zeros((2,2,2)) #c,s,r
    for c in range(2):
        for s in range(2):
            for r in range(2):
                p_C_S_R[c, s, r] = p_C(torch.tensor(c).int())*p_S_given_C(torch.tensor(s).int(), torch.tensor(c).int())*p_R_given_C(torch.tensor(r).int(), torch.tensor(c).int())

    #then create the conditional distribution:
    p_C_given_S_R = p_C_S_R[:,:,:]/p_C_S_R[:,:,:].sum(axis=(0),keepdims=True)

    ##Gibbs sampling
    
    num_samples = 11000
    samples = torch.zeros(num_samples)
    state = torch.zeros(4) #c,s,r,w
    state[3] = 1 #set w = True

    # NOTE: Fill this in
    i=0
    while i < num_samples:
        c,s,r,w = state.int().numpy()
        joint_prob = p[c,s,r,w].item()
        
        if wandb_logging:
            wandb.log({'gibbs':{'iteration':i, 'c':c, 's':s, 'r':r, 'w':w, 'p(c,s,r,w)':joint_prob}})
    
        r = dist.Categorical(p_R_given_C_S_W[c,s,:,w]).sample().item()
        s = dist.Categorical(p_S_given_C_R_W[c,:,r,w]).sample().item()
        c = dist.Categorical(p_C_given_S_R[:,s,r]).sample().item()

        if c==1:
            samples[i]=1

        state[0] = c
        state[1] = s
        state[2] = r
        i+=1

    # NOTE: Fill this in
    pcgwg = samples.mean() * 100
    if wandb_logging:
        wandb.log({'prob_cloudy_given_wet_grass':{'gibbs':pcgwg}})

    log.info('#3 (Gibbs) : The chance of it being cloudy given the grass is wet is {:.2f}%'.format(pcgwg))

    # things to think about here: can you always derive the exact conditionals required by
    # the Gibbs sampling algorithm?  what could you do if you can't? (HW 3)  what happens 
    # if a group of variables is very tightly coupled in the posterior?  will Gibbs sampling
    # be efficient in that case?  what would you do to solve such a problem?

if __name__ == "__main__":
    my_app()

# Standard imports
from time import time
import torch as tc

# Project imports
from evaluator import run_name, eval, evaluate, standard_env
from utils import log_sample_to_wandb, resample_using_importance_weights, check_addresses

def get_samples(ast:dict, num_samples:int, tmax=None, inference=None, wandb_name=None, verbose=False):
    '''
    Get some samples from a HOPPL program
    '''
    if inference is None:
        samples = get_prior_samples(ast, num_samples, tmax, wandb_name, verbose)
    elif inference == 'IS':
        samples = get_importance_samples(ast, num_samples, tmax, wandb_name, verbose)
    elif inference == 'SMC':
        samples = get_SMC_samples(ast, num_samples, wandb_name, verbose)
    else:
        print('Inference scheme:', inference, type(inference))
        raise ValueError('Inference scheme not recognised')
    return samples


def get_prior_samples(ast:dict, num_samples:int, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of samples from the prior of a HOPPL program
    '''
    samples = []
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        sample, _ = evaluate(ast, verbose=verbose)
        if wandb_name is not None: log_sample_to_wandb(sample, i, wandb_name=wandb_name)
        samples.append(sample)
        if (tmax is not None) and (time() > max_time): break
    return samples


def get_importance_samples(ast:dict, num_samples:int, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of importamnce samples from a HOPPL program
    '''
    samples = []
    sigmas = []
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        sample, sigma = evaluate(ast, verbose=verbose)
        samples.append(sample)
        sigmas.append(sigma["logW"])
        if (tmax is not None) and (time() > max_time): break
    
    log_weights = tc.stack(sigmas).type(tc.float)
    samples = tc.stack(samples).type(tc.float)
    _, resamples = resample_using_importance_weights(samples, log_weights)

    if wandb_name!=None:
        for i, sample in enumerate(resamples):
            log_sample_to_wandb(sample, i, wandb_name=wandb_name)

    return resamples


def get_SMC_samples(ast:dict, num_samples:int, wandb_name=None, verbose=False):
    '''
    Generate a set of samples via Sequential Monte Carlo from a HOPPL program
    '''

    def run_until_observe_or_end(res):
        cont, args, sigma = res
        res = cont(*args)
        while type(res) is tuple:
            if res[2]['type'] == 'observe':
                return res
            cont, args, sigma = res
            res = cont(*args)

        res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
        return res


    num_samples = num_samples
    particles = []
    weights = []
    logZs = []
    output = lambda x: x
    env = standard_env()
    sigma = {"logW": tc.tensor(0.0), 'type': None}
    output = lambda x: x # Identity function, so that output value is identical to output

    #Initial set of particles
    for i in range(num_samples):
        res = eval(ast, sigma, env, verbose)(run_name, output)
        logW = tc.tensor(0.0)
        particles.append(res)
        weights.append(logW)

    done = False
    smc_cnter = 0
    while not done:
        print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        for i in range(num_samples): #Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                #We hit an observe
                particles[i] = res
                cont, args, sigma = res
                weights[i] = sigma["logW"].clone()
                
                #checking address for the new particle (compare to the initial particle - and thus all past particles)
                if i == 0:
                    correct_address = sigma["address"]
                else:
                    if sigma["address"] != correct_address:
                        raise Exception("Error, address not matching, something went wrong. (User Exception)")
                    
                #check_addresses(particles) -> this would work if I reinitialize addresses - can't be fucked so I just reimplemented same code above

        if not done:
            #resample and keep track of logZs
            logZi, particles = resample_using_importance_weights(particles, weights)
            logZs.append(logZi)
            weights = [tc.tensor(0.0)]*num_samples
        smc_cnter += 1

    return particles
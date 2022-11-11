# Standard imports
import numpy as np
import torch as tc
import wandb

def calculate_effective_sample_size(weights:tc.Tensor, verbose=False):
    '''
    Calculate the effective sample size via the importance weights
    '''
    N = len(weights)
    weights /= weights.sum()
    ESS = 1./(weights**2).sum()
    ESS = ESS.type(tc.float)
    if verbose:
        print('Sample size:', N)
        print('Effective sample size:', ESS)
        print('Fractional sample size:', ESS/N)
        print('Sum of weights:', weights.sum())
    return ESS


def resample_using_importance_weights(samples:list, log_weights:list, wandb_name=None):
    '''
    Use the (log) importance weights to resample so as to generate posterior samples 
    '''
    nsamples = len(samples)
    weights = tc.exp(tc.tensor(log_weights)).type(tc.float64) 

    if weights.sum()==0:
        raise Exception("There is a problem with the weights. (User Exception)")

    norm_weights = weights/weights.sum() #normalize weights
    _ = calculate_effective_sample_size(weights, verbose=True)
    indices = np.random.choice(nsamples, size=nsamples, replace=True, p=norm_weights)
    new_samples = [samples[index] for index in indices]
    if wandb_name is not None:
        for i, sample in enumerate(new_samples):
            log_sample_to_wandb(sample, i, wandb_name, resample=True)
    
    log_Z = tc.log(weights.mean())

    return log_Z, new_samples


def check_addresses(samples:list):
    '''
    Perform a check that all samples (particles) are at the same address
    '''
    for i, sample in enumerate(samples):
        address = sample[2]['address']
        if i == 0:
            correct_address = address
        else:
            if address != correct_address:
                print('Correct address:', 0, correct_address)
                print('This address:', i, address)
                raise ValueError('Error, addresses do not match, something is terribly wrong')


def log_sample_to_wandb(sample, i:int, wandb_name:str, resample=False) -> None:
    '''
    Log a single W&B sample
    '''
    wandb_name_here = wandb_name+' samples' if not resample else wandb_name+' resamples'
    if sample.dim() == 0:
        samples_dict = {wandb_name_here+'; epoch': i, wandb_name_here: sample}
    else:
        samples_dict = {wandb_name_here+'; epoch': i, wandb_name_here: sample}
        for i, element in enumerate(sample):
            samples_dict[wandb_name_here+'; '+str(i)] = element
    wandb.log(samples_dict)


def log_samples_to_wandb(samples:list, wandb_name=None):
    '''
    Log a set of samples to W&B
    '''
    if wandb_name is not None:
        for i, sample in enumerate(samples):
            log_sample_to_wandb(sample, i, wandb_name=wandb_name)


def wandb_plots_homework5(samples:list, program:int):
    '''
    W&B logging of plots for homework 5
    '''
    wandb_log = {}
    if program == 1:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'n'])
        wandb_log['Program 1'] = wandb.plot.histogram(table, value='n', title='Program 1; n')
    elif program == 2:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'mu'])
        wandb_log['Program 2'] = wandb.plot.histogram(table, value='mu', title='Program 2; mu')
    elif program == 3:
        data = np.array(samples)
        xs = np.linspace(0, data.shape[1]-1, num=data.shape[1]) # [0, 1, ..., 16]
        x = []; y = []
        for i in range(data.shape[0]):     # 1000 values
            for j in range(data.shape[1]): # 16 values
                x.append(xs[j])
                y.append(data[i, j])
        xedges = np.linspace(-0.5, data.shape[1]-0.5, data.shape[1]+1) # -0.5, 0.5, ..., 16.5
        yedges = np.linspace(-0.5, data.max()+0.5, int(data.max())+2)  # -0.5, 0.5, 1.5, 2.5
        matrix, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
        xlabels = xedges[:-1]+0.5; ylabels = yedges[:-1]+0.5 # [0, 1, ..., 16]; [0, 1, 2]
        wandb_log['Program 3; heatmap'] = wandb.plots.HeatMap(xlabels, ylabels, matrix.T, show_text=True)
    else:
        raise ValueError('Program not recognised')
    wandb.log(wandb_log)


def wandb_plots_homework6(samples:list, program:int):
    '''
    W&B logging of plots for homework 5
    '''
    wandb_log = {}
    if program == 1:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'n'])
        wandb_log['Program 1'] = wandb.plot.histogram(table, value='n', title='Program 1; n')
    elif program == 2:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'mu'])
        wandb_log['Program 2'] = wandb.plot.histogram(table, value='mu', title='Program 2; mu')
    elif program == 3:
        data = np.array(samples)
        xs = np.linspace(0, data.shape[1]-1, num=data.shape[1]) # [0, 1, ..., 16]
        x = []; y = []
        for i in range(data.shape[0]):     # 1000 values
            for j in range(data.shape[1]): # 16 values
                x.append(xs[j])
                y.append(data[i, j])
        xedges = np.linspace(-0.5, data.shape[1]-0.5, data.shape[1]+1) # -0.5, 0.5, ..., 16.5
        yedges = np.linspace(-0.5, data.max()+0.5, int(data.max())+2)  # -0.5, 0.5, 1.5, 2.5
        matrix, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
        xlabels = xedges[:-1]+0.5; ylabels = yedges[:-1]+0.5 # [0, 1, ..., 16]; [0, 1, 2]
        wandb_log['Program 3; heatmap'] = wandb.plots.HeatMap(xlabels, ylabels, matrix.T, show_text=True)
    else:
        raise ValueError('Program not recognised')
    wandb.log(wandb_log)
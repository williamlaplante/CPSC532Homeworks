# Standard imports
import numpy as np
import wandb


def log_sample(sample, i: int, wandb_name: str, resample=False) -> None:
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


def wandb_plots_homework5(samples, program):
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
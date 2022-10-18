# Standard imports
from email.mime import base
from tabnanny import verbose
import numpy as np
import torch as tc
from time import time
import wandb
import hydra

# Project imports
from daphne import load_program
from tests import is_tol, run_probabilistic_test, load_truth
from general_sampling import get_sample, prior_samples
from evaluation_based_sampling import abstract_syntax_tree
from graph_based_sampling import graph
from utils import wandb_plots, wandb_plots_homework3
from likelihood_weighting import likelihood_weighting
from MH_gibbs import MH_gibbs
from HMC import HMC_sampling

def create_class(ast_or_graph, mode):
    if mode == 'desugar':
        return abstract_syntax_tree(ast_or_graph)
    elif mode == 'graph':
        return graph(ast_or_graph)
    else:
        raise ValueError('Input type not recognised')

def run_tests(tests, mode, test_type, base_dir, daphne_dir, num_samples=int(1e4), max_p_value=1e-4, compile=False, verbose=False,):

    # File paths
    test_dir = base_dir+'/programs/tests/'+test_type+'/'
    daphne_test = lambda i: test_dir+'test_%d.daphne'%(i)
    json_test = lambda i: test_dir+'test_%d_%s.json'%(i, mode)
    truth_file = lambda i: test_dir+'test_%d.truth'%(i)

    # Loop over tests
    print('Running '+test_type+' tests')
    for i in tests:
        print('Test %d starting'%i)
        print('Evaluation scheme:', mode)
        ast_or_graph = load_program(daphne_dir, daphne_test(i), json_test(i), mode=mode, compile=compile)
        ast_or_graph = create_class(ast_or_graph, mode)
        truth = load_truth(truth_file(i))
        if verbose: print('Test truth:', truth)
        if test_type == 'deterministic':
            ret, _ = get_sample(ast_or_graph, mode, verbose=verbose)
            if verbose: print('Test result:', ret)
            try:
                assert(is_tol(ret, truth))
            except AssertionError:
                raise AssertionError('Return value {} is not equal to truth {} for exp {}'.format(ret, truth, ast_or_graph))
        elif test_type == 'probabilistic':
            samples = []
            for _ in range(num_samples):
                sample, _ = get_sample(ast_or_graph, mode, verbose=verbose)
                samples.append(sample)
            p_val = run_probabilistic_test(samples, truth)
            print('p value:', p_val)
            assert(p_val > max_p_value)
        else:
            raise ValueError('Test type not recognised')
        print('Test %d passed'%i, '\n')
    print('All '+test_type+' tests passed\n')

def run_programs(programs, mode, prog_set, base_dir, daphne_dir, num_samples=int(1e3), tmax=None, compile=False, wandb_run=False, verbose=False):

    # File paths
    prog_dir = base_dir+'/programs/'+prog_set+'/'
    daphne_prog = lambda i: prog_dir+'%d.daphne'%(i)
    json_prog = lambda i: prog_dir+'%d_%s.json'%(i, mode)
    results_file_samples = lambda i: 'data/%s/%d_%s_samples.dat'%(prog_set, i, mode)
    results_file_weights = lambda i: 'data/%s/%d_%s_weights.dat'%(prog_set, i, mode)

    for i in programs:
        
        # Draw samples
        t_start = time()
        wandb_name = 'Program %s samples'%i if wandb_run else None
        print('Running: '+prog_set+':' ,i)
        print('Maximum samples [log10]:', np.log10(num_samples))
        print('Maximum time [s]:', tmax)
        print('Evaluation scheme:', mode)
        ast_or_graph = load_program(daphne_dir, daphne_prog(i), json_prog(i), mode=mode, compile=compile)
        ast_or_graph = create_class(ast_or_graph, mode)
        samples = prior_samples(ast_or_graph, mode, num_samples, tmax=tmax, wandb_name=wandb_name, verbose=verbose)
        samples = tc.stack(samples).type(tc.float)

        np.savetxt(results_file_samples(i), samples)

        # Calculate some properties of the data
        print('Samples shape:', samples.shape)
        print('First sample:', samples[0])
        print('Sample mean:', samples.mean(axis=0))
        print('Sample standard deviation:', samples.std(axis=0))

        # Weights & biases plots
        if wandb_run: wandb_plots(samples, i)

        # Finish
        t_finish = time()
        print('Time taken [s]:', t_finish-t_start)
        print('Number of samples:', len(samples))
        print('Finished program {}\n'.format(i))


@hydra.main(version_base=None, config_path='', config_name='config')
def run_all(cfg):

    # Configuration
    wandb_run = cfg['wandb_run']
    mode = cfg['mode']
    num_samples = int(cfg['num_samples'])
    compile = cfg['compile']
    base_dir = cfg['base_dir']
    daphne_dir = cfg['daphne_dir']
    prog_set = cfg["prog_set"]
    sampling_method = cfg["sampling_method"]
    entity = cfg["entity"]

    # Initialize W&B
    if wandb_run: wandb.init(project=prog_set +'-'+mode, entity=entity)
    
    # Deterministic tests
    tests = cfg['deterministic_tests']
    run_tests(tests, mode=mode, test_type='deterministic', base_dir=base_dir, daphne_dir=daphne_dir, compile=compile)

    # Proababilistic tests
    tests = cfg['probabilistic_tests']
    run_tests(tests, mode=mode, test_type='probabilistic', base_dir=base_dir, daphne_dir=daphne_dir, compile=compile)

    # Programs
    programs = cfg[prog_set + '_programs']

    if sampling_method=="likelihood_weighting":

        print("Likelihood weighting sampling : ")

        for program in programs:
            print("\nProgram {} currently running...".format(program))
            
            if wandb_run : 
                wandb_name = 'Program %s samples'%program
            else:
                wandb_name = None

            samples = likelihood_weighting(program, mode=mode, prog_set=prog_set, num_samples=num_samples, wandb_name=wandb_name)
            samples = tc.stack(samples).type(tc.float)

            print("Sample mean : {}".format(samples.mean(axis=0)))
            print("Sample standard deviation : {}".format(samples.std(axis=0)))

            if wandb_run : wandb_plots_homework3(samples, program)

            np.savetxt("./data/HW3/%s/samples_program_%d_%s.dat"%(sampling_method, program, mode), samples)

    

    elif sampling_method=="MH_gibbs":

        print("MC within Gibbs sampling : ")

        for program in programs:
            print("\nProgram {} currently running...".format(program))

            samples = MH_gibbs(program, prog_set=prog_set, num_samples=num_samples) #only runs mode=graph
            samples = tc.stack(samples).type(tc.float)

            print("Sample mean : {}".format(samples.mean(axis=0)))
            print("Sample standard deviation : {}".format(samples.std(axis=0)))

            if wandb_run : wandb_plots_homework3(samples, program)

            np.savetxt("./data/HW3/%s/samples_program_%d_%s.dat"%(sampling_method, program, mode), samples)


    elif sampling_method=="HMC":

        print("HMC sampling : ")

        for program in programs:
            if program not in [1,2,5]: #programs 3,4 are not differentiable, thus HMC is not applicable
                continue
            
            print("\nProgram {} currently running...".format(program))

            samples = HMC_sampling(program, prog_set=prog_set, num_samples=num_samples) #only runs mode=graph
            samples = tc.stack(samples).type(tc.float)

            print("Sample mean : {}".format(samples.mean(axis=0)))
            print("Sample standard deviation : {}".format(samples.std(axis=0)))

            if wandb_run : wandb_plots_homework3(samples, program)

            np.savetxt("./data/HW3/%s/samples_program_%d_%s.dat"%(sampling_method, program, mode), samples)

    

    else :
        run_programs(programs, mode=mode, prog_set=prog_set, base_dir=base_dir, daphne_dir=daphne_dir, num_samples=num_samples, compile=compile, wandb_run=wandb_run, verbose=verbose)

    # Finalize W&B
    if wandb_run: wandb.finish()

if __name__ == '__main__':
    run_all()
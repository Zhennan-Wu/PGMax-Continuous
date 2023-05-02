from pgmax import fgraph, vgroup, fgroup
from pgmax import factor as F
from pgmax import fgraph, fgroup, infer, vgroup
import numpy as np
import itertools


def extract_factors(particles, dist = None):
    '''
    Read from XADD file to get factor dependence
    return: list -- parent variables + child variable 
    '''
    pass


def extract_ft_dist():
    '''
    Using simulator to generate a lot of samples and calculate the experimental probability
    '''
    pass


def get_ft_var(factor, time):
    '''
    Generate variables of a factor at a certain time
    '''
    pass


def get_config(subs):
    '''
    Transform list to np array of value grounding
    '''
    pass


class MIX_MODEL:

    def __init__(self, num_mixture, total_num_vars, horizon):
        '''
        dimension of the mixture model: horizon x (state var + action var + reward) x number of mixture 
        '''
        self.num_mixture = num_mixture
        self.num_vars = total_num_vars
        self.horizon = horizon
        self.mixture_mean = np.ones((horizon, total_num_vars, num_mixture))
        self.mixture_std = np.ones((horizon, total_num_vars, num_mixture))
        self.mixture_weight = np.ones((horizon, total_num_vars, num_mixture))/num_mixture


def update_dbn(model, num_samples, random_seed, num_mixture, s_vars, a_vars, horizon):
    '''
    Generate dynamic bayesian network structure.
    s_vars: list -- a list of state variable
    a_vars: list -- a list of action variable
    horizon: int -- search depth
    
    '''

    # create variable for factored DBN 
    total_num_vars = len(s_vars) + len(a_vars) + 1 # reward variables


    f_actions = vgroup.NDVarArray(num_states=num_samples, shape=(horizon, len(a_vars)))
    f_states = vgroup.NDVarArray(num_states=num_samples, shape=(horizon, len(s_vars)))
    
    f_rwds = vgroup.NDVarArray(num_states=2, shape=(horizon,))
    f_cumus = vgroup.NDVarArray(num_states=2, shape=(horizon+1,))
    
    fg = fgraph.FactorGraph(variable_groups=[f_actions, f_states, f_rwds, f_cumus])

    # particle generation
    np.random.seed(random_seed)

    #? Could the for loop be removed?
    z = np.ones((horizon, total_num_vars, num_mixture))
    for i in range(horizon):
        for j in range(total_num_vars):
            z[i][j] = z[i][j] * np.random.multinomial(1, model.mixture_weight[i][j])
    
    mean = np.sum(model.mixture_mean * z, axis = -1)
    std = np.sum(model.mixture_std * z, axis = -1)
    particles = np.random.normal(mean, std, (num_samples, horizon, total_num_vars))

    #? Could it be rewritten into a class to avoid multiple parameter passing?
    factors = extract_factors()
    entry = {'subs': [], 'probs': []}

    completed_factors = []
    for t_step in range(0, horizon-1):
        p_table = dict.fromkeys(factors, entry)
        for s_id in range(0, num_samples):
            state = dict(zip(s_vars, particles[s_id][t_step][0: len(s_vars)]))
            action = dict(zip(a_vars, particles[s_id][t_step][len(s_vars)+1:len(s_vars)+len(a_vars)]))
            next_state =  dict(zip(s_vars, particles[s_id][t_step+1][0: len(s_vars)]))
            factors, subs, probs = extract_ft_dist(state, action, next_state)
            for ft, s, p in zip(factors, subs, probs):
                p_table[ft]['subs'].append(s)
                p_table[ft]['probs'].append(p)
        
        for ft in p_table.keys():
            completed_factors.append(F.enum.EnumFactor(
                variables=get_ft_var(ft, t_step),
                factor_configs=get_config(p_table[ft]['subs']),
                log_potentials=np.log(np.array(p_table[ft]['probs'])),))
    
    fg.add_factors(completed_factors)

    # connect distribution factor remains to be added
    return fg, particles


def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


def update_model(fg, model, num_samples):
    bp = infer.BP(fg.bp_state)
    bp.init()
    bp_arrays = bp.run_bp(bp_arrays, num_iters=10, damping=0.5)
    values, probs = bp.get_beliefs(bp_arrays)
    beliefs_match_mix = np.stack(probs, model.num_mixture)
    values_match_mix = np.stack(values, model.num_mixture)
    beliefs_match_mix = beliefs_match_mix.reshape((model.horizon, model.num_vars, model.num_mixture))
    values_match_mix = values_match_mix.reshape((model.horizon, model.num_vars, model.num_mixture))

    z_matrix = model.mixture_weight*beliefs_match_mix*normpdf(values_match_mix, model.mixture_mean, model.mixture_std)
    gamma_z = z_matrix/np.sum(z_matrix, axis=-1)

    N_matrix = np.sum(gamma_z, axis = 0)

    model.mixture_mean = gamma_z * beliefs_match_mix * values_match_mix/N_matrix
    model.mixture_std = gamma_z * (beliefs_match_mix * (values_match_mix - model.mixture_mean))^2/N_matrix
    model.mixture_weight = N_matrix/num_samples
    return model


def mepbp(domain_ins, num_mixture, horizon, num_samples, seed, iter = 100):
    total_num_vars = len(domain_ins.s_vars) + len(domain_ins.a_vars) + 1 
    model = MIX_MODEL(num_mixture, total_num_vars, horizon)
    for _ in range(iter):
        fg = update_dbn(model, num_samples, seed, num_mixture, domain_ins.s_vars, domain_ins.a_vars, horizon)
        model = update_model(fg, model, num_samples)
    
         

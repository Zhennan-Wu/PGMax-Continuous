from pgmax import factor as F
from pgmax import fgraph, fgroup, infer, vgroup
import numpy as np
import itertools

from typing import Optional, Tuple, Dict, Union, List, Set

from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder


class MIX_MODEL:

    def __init__(self, num_mixture, s_vars, a_vars, horizon, domain, instance):
        '''
        dimension of the mixture model: horizon x (state var + action var + reward) x number of mixture 
        '''
        self.num_mixture = num_mixture
        self.num_vars = len(s_vars) + len(a_vars) + 1 # reward variable
        self.s_vars = s_vars
        self.a_vars = a_vars
        self.horizon = horizon
        self.mixture_mean = np.ones((horizon, self.num_vars, num_mixture))
        self.mixture_std = np.ones((horizon, self.num_vars, num_mixture))
        self.mixture_weight = np.ones((horizon, self.num_vars, num_mixture))/num_mixture
        self.factors, self.model_xadd = self.extract_factors(domain, instance)


    def update_dbn(self, num_samples, random_seed):
        '''
        Generate dynamic bayesian network structure.    
        '''

        f_actions = vgroup.NDVarArray(num_states=num_samples, shape=(self.horizon, len(self.a_vars)))
        f_states = vgroup.NDVarArray(num_states=num_samples, shape=(self.horizon, len(self.s_vars)))
        
        f_rwds = vgroup.NDVarArray(num_states=2, shape=(self.horizon,))
        f_cumus = vgroup.NDVarArray(num_states=2, shape=(self.horizon+1,))
        
        fg = fgraph.FactorGraph(variable_groups=[f_actions, f_states, f_rwds, f_cumus])

        # particle generation
        np.random.seed(random_seed)

        #? Could the for loop be removed?
        # sample mixture latent variables
        z = np.ones((self.horizon, self.num_vars, self.num_mixture))
        for i in range(self.horizon):
            for j in range(self.num_vars):
                z[i][j] = z[i][j] * np.random.multinomial(1, self.mixture_weight[i][j])
        
        # sample the particles
        mean = np.sum(self.mixture_mean * z, axis = -1)
        std = np.sum(self.mixture_std * z, axis = -1)
        particles = np.random.normal(mean, std, (num_samples, self.horizon, self.num_vars))

        #? Could it be rewritten into a class to avoid multiple parameter passing?
        
        entry = {'subs': [], 'probs': []}

        completed_factors = []
        for t_step in range(0, self.horizon-1):
            p_table = dict.fromkeys(factors, entry)
            for s_id in range(0, num_samples):
                state = dict(zip(self.s_vars, particles[s_id][t_step][0: len(self.s_vars)]))
                action = dict(zip(self.a_vars, particles[s_id][t_step][len(self.s_vars)+1:len(self.s_vars)+len(self.a_vars)]))
                next_state = dict(zip(self.s_vars, particles[s_id][t_step+1][0: len(self.s_vars)]))
                factors, subs, probs = self.extract_ft_dist(state, action, next_state, self.factors, self.model_xadd)
                for ft, s, p in zip(factors, subs, probs):
                    p_table[ft]['subs'].append(s)
                    p_table[ft]['probs'].append(p)
            
            for ft in p_table.keys():
                completed_factors.append(F.enum.EnumFactor(
                    variables=self.get_ft_var(ft, t_step),
                    factor_configs=self.get_config(p_table[ft]['subs']),
                    log_potentials=np.log(np.array(p_table[ft]['probs'])),))
        
        fg.add_factors(completed_factors)

        # connect distribution factor remains to be added
        return fg, particles


    def normpdf(self, x, mean, sd):
        var = float(sd)**2
        denom = (2*np.pi*var)**.5
        num = np.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom


    def update_model(self, num_samples):
        bp = infer.BP(self.fg.bp_state)
        bp.init()
        bp_arrays = bp.run_bp(bp_arrays, num_iters=10, damping=0.5)
        values, probs = bp.get_beliefs(bp_arrays)
        beliefs_match_mix = np.stack(probs, self.num_mixture)
        values_match_mix = np.stack(values, self.num_mixture)
        beliefs_match_mix = beliefs_match_mix.reshape((self.horizon, self.num_vars, self.num_mixture))
        values_match_mix = values_match_mix.reshape((self.horizon, self.num_vars, self.num_mixture))

        z_matrix = self.mixture_weight*beliefs_match_mix*self.normpdf(values_match_mix, self.mixture_mean, self.mixture_std)
        gamma_z = z_matrix/np.sum(z_matrix, axis=-1)

        N_matrix = np.sum(gamma_z, axis = 0)

        self.mixture_mean = gamma_z * beliefs_match_mix * values_match_mix/N_matrix
        self.mixture_std = gamma_z * (beliefs_match_mix * (values_match_mix - self.mixture_mean))^2/N_matrix
        self.mixture_weight = N_matrix/num_samples
        
            
    def _hybrid_split(self):
        pass


    def _extract_factors(self, domain, instance):
        '''
        Read from XADD file to get factor dependence
        return: 
        - a dict with key to be child variables and the value to be the set of parent variables
        - xadd model
        '''
        # Read the domain and instance files
        env_info = ExampleManager.GetEnvInfo(domain)
        domain = env_info.get_domain()
        instance = env_info.get_instance(instance)

        # Read and parse domain and instance
        reader = RDDLReader(domain, instance)
        domain = reader.rddltxt
        parser = RDDLParser(None, False)
        parser.build()

        # Parse RDDL file
        rddl_ast = parser.parse(domain)

        # Ground domain
        grounder = RDDLGrounder(rddl_ast)
        model = grounder.Ground()

        # XADD compilation
        model_xadd = RDDLModelWXADD(model)
        model_xadd.compile()

        cpfs = model_xadd.cpfs
        cpfs.update({'reward': model_xadd.reward})
        cpfs.update({'terminals': model_xadd.terminals})

        # Extract factors
        factors = {}
        for cpf, node_id in model_xadd.cpfs.items():
            if not node_id or cpf == "terminals":
                continue
            parents = model_xadd.collect_vars(node_id)
            factors[cpf] = parents

        return factors, model_xadd


    def _extract_ft_dist(self, state, action, next_state, factors, model_xadd):
        '''
        Using simulator to generate a lot of samples and calculate the experimental probability
        return:
        - factors, the same as input
        - subs, the corresponding value assignment
        - probs, the corresponding probability
        '''
        subs = []
        probs = []

        for nd in factors.keys():
            bool_vars, cont_vars = self._hybrid_split(factors[nd])
            bool_assign = {}
            cont_assign = {}
            for bv in bool_vars:
                if bv in state.keys():
                    bool_assign[bv] = state[bv]
                elif bv in action.keys():
                    bool_assign[bv] = action[bv]
                else:
                    # reward factor distribution to be filled
                    continue
            
            for cv in cont_vars:
                if cv in state.keys():
                    cont_assign[cv] = state[cv]
                elif cv in action.keys():
                    cont_assign[cv] = action[cv]
                else:
                    # reward factor distribution to be filled
                    continue
            sub = []
            for v in factors[nd]:
                if (v in state.key()):
                    sub.append(state[v])
                else:
                    sub.append(action[v])
            subs.append(sub)
            probs.append(model_xadd.evaluate(nd, bool_assign = bool_assign, cont_assign = cont_assign))
        
        return factors, subs, probs


    def _get_ft_var(self, factor, time):
        '''
        Generate variables of a factor at a certain time
        '''
        pass


    def _get_config(self, subs):
        '''
        Transform list to np array of value grounding
        '''
        pass

def mepbp(domain_ins, num_mixture, horizon, num_samples, seed, iter = 100):
    model = MIX_MODEL(num_mixture, domain_ins.s_vars, domain_ins.a_vars, horizon)
    for _ in range(iter):
        model.update_dbn(num_samples, seed)
        model.update_model(model, num_samples)
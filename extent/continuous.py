from pgmax import fgraph, vgroup, fgroup
from pgmax import factor as F
import numpy as np
import itertools


def init_samples(num_of_samples, random_seed, num_mixture, reward_dist, connect_dist, trans_prob, policy, s_vars, a_vars, r_vars, horizon):
    '''
    Generate dynamic bayesian network structure.
    '''

    # create variable for factored DBN 
    total_num_vars = len(s_vars) + len(a_vars)

    f_actions = vgroup.NDVarArray(num_states=num_of_samples, shape=(horizon, len(a_vars)))
    f_states = vgroup.NDVarArray(num_states=num_of_samples, shape=(horizon, len(s_vars)))

    f_p_rwds = vgroup.NDVarArray(num_states=2, shape=(horizon, len(r_vars)+1))
    f_p_cumus = vgroup.NDVarArray(num_states=2, shape=(horizon, len(r_vars)+1))
    
    f_rwds = vgroup.NDVarArray(num_states=2, shape=(horizon,))
    f_cumus = vgroup.NDVarArray(num_states=2, shape=(horizon+1,))
    
    fg = fgraph.FactorGraph(variable_groups=[f_actions, f_states, f_p_rwds, f_p_cumus, f_rwds, f_cumus])

    # particle generation
    np.random.seed(random_seed)

    mixture_mean = np.ones((total_num_vars, num_mixture))
    mixture_std = np.ones((total_num_vars, num_mixture))
    mixture_weight = np.ones((total_num_vars, num_mixture))/num_mixture

    #? Could the for loop be removed?
    z = np.ones((total_num_vars, num_mixture))
    for i in range(len(z)):
        z[i] = z[i] * np.random.multinomial(1, mixture_weight[i])
    
    mean = np.sum(mixture_mean * z, axis = 1)
    std = np.sum(mixture_std * z, axis = 1)
    particles = np.random.normal(mean, std, (num_of_samples, total_num_vars, num_mixture))
    connecting_factor_vars = []
    policy_f_dists = []
    completed_factors = []
    for time_step in range(0, horizon):
        time_stamp = str(time_step)
        next_time = time_step + 1
        next_stamp = str(next_time)
        prev_time = time_step - 1
        prev_stamp = str(prev_time)

        policy_factor_vars = []
        factor_var = []
        factor_var.append("t" + time_stamp + "_atomic_action")
        policy_factor_vars.append(factor_var)
        for idx, factor_var in enumerate(policy_factor_vars):
            policy_f_dist = generate_factor_dist(time_step, s_vars, a_vars, valid_actions, atomic_action_lst, "policy", factor_var, policy, mes_type, a_mask)
            policy_f_dists.append(policy_f_dist)
        
        if (time_step != 0):
            trans_factor_vars = []
            f_var_names = []
            for cs_idx, s_var in enumerate(s_vars):
                child_var = s_var + "'"
                factor_var = []
                f_var_name = []

                for curr_s_var in state_dependency[child_var]:
                    ps_idx = s_vars.index(curr_s_var)
                    factor_var.append("t" + prev_stamp + "_" + curr_s_var)
                    f_var_name.append(f_states[prev_time, ps_idx])
                    if (time_step == 1):
                        if (not curr_s_var in init_candids):
                            init_candids.append(curr_s_var)

                factor_var.append("t" + prev_stamp + "_atomic_action")
                factor_var.append("t" + time_stamp + "_" + s_var)
                f_var_name.append(f_actions[prev_time])
                f_var_name.append(f_states[time_step, cs_idx])

                trans_factor_vars.append(factor_var)
                f_var_names.append(f_var_name)

            for landing_s, factor_var, fg_vars in zip(s_vars, trans_factor_vars, f_var_names):
                parent_s = state_dependency[landing_s + "'"]
                trans_f_dist = generate_factor_dist(time_step, parent_s, a_vars, valid_actions, atomic_action_lst, "trans", factor_var, trans_prob, mes_type, a_mask)

                s_configs = np.array(list(itertools.product(np.arange(2), repeat=len(fg_vars)-2)))
                s_dim = len(s_configs)
                s_configs = np.repeat(s_configs, len(atomic_action_lst), axis=0)
                a_configs = np.array(list(range(len(atomic_action_lst))))
                a_configs = np.tile(a_configs, (s_dim, 1))
                a_configs = a_configs.reshape((-1, 1))
                p_configs = np.append(s_configs, a_configs, axis=1)
                c_dim = len(p_configs)
                p_configs = np.repeat(p_configs, 2, axis=0)
                p_configs = p_configs.reshape((int(c_dim*2), -1))
                c_configs = np.array([0, 1])
                c_configs = np.tile(c_configs, (c_dim, 1))
                c_configs = c_configs.reshape((-1, 1))
                f_configs = np.append(p_configs, c_configs, axis=1)
                f_configs = f_configs.astype(int)

                completed_factors.append(F.enum.EnumFactor(
                    variables=fg_vars,
                    factor_configs=f_configs,
                    log_potentials=np.log(trans_f_dist.flatten()),))

        # Reformulation of rewards

        # because all action share the same reward table except for the intrinsic action cost. (we omit 2 domains that fail this assumption)
        # we only need to set one action to enumerate the state-dependent reward table
        default_act = 'noop'
        len_of_cases = len(reward_dist[default_act]['parents'])

        partial_rwd_s_factor_vars = []
        r_f_vars = []
        for idx in range(0, len_of_cases):
            factor_var = []
            r_f_var = []
            for svar in reward_dist[default_act]['parents'][idx]:
                ps_idx = s_vars.index(svar)
                factor_var.append("t" + time_stamp + "_" + svar)
                r_f_var.append(f_states[time_step, ps_idx])
                if (not svar in init_candids):
                    init_candids.append(svar)
            factor_var.append("t" + next_stamp + "_pr" + str(idx+1))
            r_f_var.append(f_p_rwds[ary_idx(next_time), idx])
            partial_rwd_s_factor_vars.append(factor_var)
            r_f_vars.append(r_f_var)

        # formalize distribution for factor nodes

        for case, vars, factor_var, r_factor in zip(reward_dist[default_act]['cases'], reward_dist[default_act]['parents'], partial_rwd_s_factor_vars, r_f_vars):
            r_f_configs = np.array(list(itertools.product(np.arange(2), repeat=len(r_factor))))
            pr_s_f_dist = generate_factor_dist(next_time, vars, None, None, None, "partial_s_rwd", normal_factor, case, mes_type, a_mask)

            completed_factors.append(F.enum.EnumFactor(
                variables=r_factor,
                factor_configs=r_f_configs,
                log_potentials=np.log(pr_s_f_dist.flatten()),))

        partial_rwd_a_factor_vars = []
        r_a_f_vars = []
        factor_var = []
        r_a_f_var = []
        factor_var.append("t" + time_stamp + "_atomic_action" )
        factor_var.append("t" + next_stamp + "_pr" + str(len_of_cases + 1))
        r_a_f_var.append(f_actions[time_step])
        r_a_f_var.append(f_p_rwds[ary_idx(next_time), len_of_cases])
        partial_rwd_a_factor_vars.append(factor_var)
        r_a_f_vars.append(r_a_f_var)
        # formalize distribution for factor nodes
        for factor_var, r_factor in zip(partial_rwd_a_factor_vars, r_a_f_vars):
            r_a_configs = np.array(list(range(len(atomic_action_lst))))
            r_a_configs = np.repeat(r_a_configs, 2, axis=0)
            r_configs = np.array([0, 1])
            r_configs = np.tile(r_configs, (len(atomic_action_lst), 1))
            r_a_configs = r_a_configs.reshape((-1, 1))
            r_configs = r_configs.reshape((-1, 1))
            f_configs = np.append(r_a_configs, r_configs, axis=1)

            pr_a_f_dist = generate_factor_dist(next_time, None, None, None, None, "partial_a_rwd", normal_factor, reward_dist, mes_type, a_mask)

            fg.add_factors(F.enum.EnumFactor(
                variables=r_factor,
                factor_configs=f_configs,
                log_potentials=np.log(pr_a_f_dist.flatten()),))

        completed_factors.append(F.enum.EnumFactor(
            variables=[(f_p_cumus[ary_idx(next_time), 0])],
            factor_configs=np.arange(2)[:, None],
            log_potentials=np.log(np.array([0.0001, 0.9999])),))
        
        cumu_rwd_factor_vars = []
        cumu_f_vars = []
        for idx in range(0, len_of_cases):
            factor_var = []
            cumu_f_var = []
            factor_var.append("t" + next_stamp + "_cr" + str(idx))
            factor_var.append("t" + next_stamp + "_pr" + str(idx+1))
            factor_var.append("t" + next_stamp + "_cr" + str(idx+1))

            cumu_f_var.append(f_p_cumus[ary_idx(next_time), idx])
            cumu_f_var.append(f_p_rwds[ary_idx(next_time), ary_idx(idx+1)])
            cumu_f_var.append(f_p_cumus[ary_idx(next_time), idx+1])
            cumu_rwd_factor_vars.append(factor_var)
            cumu_f_vars.append(cumu_f_var)
        # Final step reward
        factor_var = []
        cumu_f_var = []
        factor_var.append("t" + next_stamp + "_cr" + str(len_of_cases))
        factor_var.append("t" + next_stamp + "_pr" + str(len_of_cases+1))
        factor_var.append("r" + next_stamp)

        cumu_f_var.append(f_p_cumus[ary_idx(next_time), len_of_cases])
        cumu_f_var.append(f_p_rwds[ary_idx(next_time), ary_idx(len_of_cases+1)])
        cumu_f_var.append(f_rwds[ary_idx(next_time)])
        cumu_rwd_factor_vars.append(factor_var)
        cumu_f_vars.append(cumu_f_var)

        # formalize distribution for factor nodes
        auxiliary_dist = generate_connect_distribution(len_of_cases + 1)

        for idx, factor_var, cumu_f_var in zip(list(range(len(cumu_rwd_factor_vars))), cumu_rwd_factor_vars, cumu_f_vars):
            f_configs = np.array(list(itertools.product(np.arange(2), repeat=len(cumu_f_var))))
            aux_r_dist = np.array([[[1 - auxiliary_dist[idx+1]['00'], auxiliary_dist[idx+1]['00']], [1 - auxiliary_dist[idx+1]['01'], auxiliary_dist[idx+1]['01']]], [[1 - auxiliary_dist[idx+1]['10'], auxiliary_dist[idx+1]['10']], [1 - auxiliary_dist[idx+1]['11'], auxiliary_dist[idx+1]['11']]]])

            completed_factors.append(F.enum.EnumFactor(
                variables=cumu_f_var,
                factor_configs=f_configs,
                log_potentials=np.log(aux_r_dist.flatten()),))


        # trick to connect reward of different time slice
        connecting_factor_vars = []
        connecting_factor_vars.append("c" + time_stamp)
        connecting_factor_vars.append("r" + next_stamp)
        connecting_factor_vars.append("c" + next_stamp)

        connect_f_vars = []
        connect_f_vars.append(f_cumus[time_step])
        connect_f_vars.append(f_rwds[ary_idx(next_time)])
        connect_f_vars.append(f_cumus[next_time])

        con_f_dist_raw = np.array([[[1 - connect_dist[next_time]['00'], connect_dist[next_time]['00']], [1 - connect_dist[next_time]['01'], connect_dist[next_time]['01']]], [[1 - connect_dist[next_time]['10'], connect_dist[next_time]['10']], [1 - connect_dist[next_time]['11'], connect_dist[next_time]['11']]]])
        if (mes_type == 'bw' and next_time == horizon):
            # backward loopy bp formulation
            ob_mask = np.array([[[0.0001, 0.9999], [0.0001, 0.9999]],[[0.0001, 0.9999], [0.0001, 0.9999]]])
        else:
            ob_mask = np.array([[[1.0, 1.0], [1.0, 1.0]],[[1.0, 1.0], [1.0, 1.0]]])
        con_f_dist = np.multiply(con_f_dist_raw, ob_mask)

        f_configs = np.array(list(itertools.product(np.arange(2), repeat=len(connect_f_vars))))
        completed_factors.append(F.enum.EnumFactor(
            variables=connect_f_vars,
            factor_configs=f_configs,
            log_potentials=np.log(con_f_dist.flatten()),))

    policy_unaries = fgroup.EnumFactorGroup(
        variables_for_factors=[[f_actions[t]] for t in range(0, horizon)],
        factor_configs=np.arange(len(atomic_action_lst))[:, None],
        log_potentials=np.stack(policy_f_dists, axis=0),
    )

    completed_factors.insert(0, policy_unaries)

    completed_factors.insert(0, F.enum.EnumFactor(
        variables=[f_cumus[0]],
        factor_configs=np.arange(2)[:, None],
        log_potentials=np.log(np.array([0.0001, 0.9999])),))

    init_vs = []
    for s_var in init_candids:
        init_s_idx = s_vars.index(s_var)
        init_vs.append([f_states[0, init_s_idx]])
    init_unaries = fgroup.EnumFactorGroup(
        variables_for_factors=init_vs,
        factor_configs=np.arange(2)[:, None],
        log_potentials=np.stack([np.zeros(len(init_candids)), np.ones(len(init_candids))], axis=1),
    )

    completed_factors.insert(0, init_unaries)
    fg.add_factors(completed_factors)

    return [fg, init_candids, policy_unaries, init_unaries]
# import gymnasium as gym
# env = gym.make("CartPole-v1", render_mode="human")

# print(env.P)
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()

# import mujoco
# import mediapy as media

# xml = """
# <mujoco>
#   <worldbody>
#     <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
#     <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
#   </worldbody>
# </mujoco>
# """
# model = mujoco.MjModel.from_xml_string(xml)

# id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'green_sphere')
# model.geom_rgba[id, :]

# print('id of "green_sphere": ', model.geom('green_sphere').id)
# print('name of geom 1: ', model.geom(1).name)
# print('name of body 0: ', model.body(0).name)

# data = mujoco.MjData(model)
# print(data.geom_xpos)

# # Make renderer, render and show the pixels
# renderer = mujoco.Renderer(model)
# media.show_image(renderer.render())

import jax
import jax.numpy as jnp
import re

from pyRDDLGym.Core.Parser import parser as Rddlparser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Jax import JaxRDDLCompiler, JaxRDDLBackpropPlanner

def prepare_rddl_compilations(domain_path, instance_path): 
    # To read from pyRDDLGym
    # env_info = ExampleManager().GetEnvInfo(ENV)
    # domain = env_info.get_domain()
    # instance = env_info.get_instance(inst)    
    rddltxt = RDDLReader(domain_path, instance_path).rddltxt

    # For arg1, match everything except a comma. For arg2, match everything except ) followed by at max one ). For patterns like (?p)
    normal_pattern = re.compile('Normal\(([^,]+),\s*([^)]+[\)]?)\)')
    uniform_pattern = re.compile('Uniform\(([^,]+),\s*([^)]+[\)]?)\)')
    weibull_pattern = re.compile('Weibull\(([^,]+),\s*([^)]+[\)]?)\)')
    bernoulli_pattern = re.compile('Bernoulli\(([^)]+[\)]?)\)')
    
    for (dist, pattern, reparam_fn) in [("normal",normal_pattern, reparam_normal), 
                                      ("uniform",uniform_pattern, reparam_uniform), 
                                      ("weibull",weibull_pattern, reparam_weibull),
                                      ("bernoulli", bernoulli_pattern, reparam_bernoulli)]:
        if pattern.search(rddltxt) is not None:
            rddltxt = re.sub(pattern, reparam_fn, rddltxt)

            eps_str = EPS_STR[dist]


            # Insert state-fluent definition and CPF just once.
            eps_default = f"{eps_str} : {{ state-fluent, real, default = 0.0 }};"
            pvar_pattern = re.compile('pvariables[\s]*{')
            eps_match = pvar_pattern.search(rddltxt)
            rddltxt = f"{rddltxt[:eps_match.end()]} \n {eps_default} \n {rddltxt[eps_match.end():]}"
            
            eps_cpf_str = f"{eps_str}' = {eps_str};"
            cpf_pattern = re.compile('cpfs[\s]*{')
            eps_cpf_match = cpf_pattern.search(rddltxt)
            rddltxt = f"{rddltxt[:eps_cpf_match.end()]} \n {eps_cpf_str} \n {rddltxt[eps_cpf_match.end():]}"

    print(f"RDDL after noise injection: {rddltxt}")
                
    rddlparser = Rddlparser.RDDLParser()
    rddlparser.build()
    ast = rddlparser.parse(rddltxt)
    
    model = RDDLLiftedModel(ast)

    a_keys = model.actions.keys()
    s_keys = model.states.keys()
    
    ground_a_keys = model.groundactions().keys()

    # ns_mapping = model.next_state
    # ns_keys = [ns_mapping[k] for k in s_keys if k not in ["disprod_epsilon"]]
    ns_keys = [f"{k}'" for k in s_keys if k not in DISPROD_NOISE_VARS]

    compiled = JaxRDDLBackpropPlanner.JaxRDDLCompilerWithGrad(rddl=model)
        
    # compiled_expr_tree = compiled._compile_cpfs_into_exp_tree()
    # for k,v in compiled_expr_tree.items():
    #     print(f"Key: {k}, Expression: {v}")
    
    compiled.compile()
        
    # JaxRDDLCompiler turns this on causing a lot of logs on screen. 
    jax.config.update('jax_log_compiles', False)


    reward, cpfs = compiled.reward, compiled.cpfs
    model_params = compiled.model_params
    
    # This decides the order of processing
    levels = [_v for v in compiled.levels.values() for _v in v]  
    
    # compiled.init_values is a dict of values of constants and variables. Split it into two dicts
    const_dict = {k:compiled.init_values[k] for k in compiled.rddl.nonfluents.keys()}

    return reward, cpfs, const_dict, s_keys, list(a_keys), ground_a_keys, ns_keys, levels, model.grounded_names, model_params


def ns_and_reward(cpfs, s_keys, a_keys, ns_keys, const_dict, levels, extra_params, reward_fn, s_gs_idx, a_ga_idx, state, action, rng_key):
    """
    s_keys, a_keys: not grounded 
    gs_keys, ga_keys: grounded
    grounded_names: map s_keys -> gs_keys, a_keys -> ga_keys
    state, action: grounded
    """

    state_dict = {k: s_gs_idx[k][2](state[s_gs_idx[k][0] : s_gs_idx[k][1]]) for k in s_keys}
    action_dict = {k: a_ga_idx[k][2](action[a_ga_idx[k][0] : a_ga_idx[k][1]]) for k in a_keys}
    
    subs = {**state_dict, **action_dict, **const_dict}

    for key in levels:
        expr = cpfs[key]
        subs[key], rng_key, _ = expr(subs, extra_params, rng_key)
        
    reward, _, _ = reward_fn(subs, extra_params, rng_key)

    return jnp.hstack([subs[k] for k in ns_keys]), reward
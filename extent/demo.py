from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import RandomAgent

ENV = 'Wildfire'

# get the environment infos
EnvInfo = ExampleManager.GetEnvInfo(ENV)

# set up the environment class, choose instance 0 because every example has at least one example instance
myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))
# set up the environment visualizer
myEnv.set_visualizer(EnvInfo.get_visualizer())
# set up an example aget
agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.numConcurrentActions)

total_reward = 0
state = myEnv.reset()

for step in range(myEnv.horizon):
    myEnv.render()
    action = agent.sample_action()
    next_state, reward, done, info = myEnv.step(action)
    total_reward += reward
    state = next_state
    if done:
        break

print("episode ended with reward {}".format(total_reward))
myEnv.close()

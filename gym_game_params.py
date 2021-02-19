import gym
from gym import envs
env_specs = envs.registry.all()
envs_ids = [env_spec.id for env_spec in env_specs]
print(envs_ids)
disable_list=[]
env = gym.make('CarRacing-v0')
print('observation_space:',env.observation_space,'\taction_space:',env.action_space)
env.close()

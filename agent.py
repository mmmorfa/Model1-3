import gymnasium as gym

from stable_baselines3 import DQN

from gym_examples.envs.slice_creation_env3 import SliceCreationEnv3

env = SliceCreationEnv3()


#model = DQN.load("dqn_slices1.zip", env)
model = DQN.load("/home/mario/Documents/DQN_Models/Model 1/gym-examples2/dqn_slices2(Arch:16; learn:1e-3; starts:250k; fraction:0_5; train: 1.5M)", env)

obs, info = env.reset()

cont = 0
while cont<99:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print('Action: ', action,'Observation: ', obs, ' | Reward: ', reward, ' | Terminated: ', terminated)
    cont += 1
    if terminated or truncated:
        obs, info = env.reset()
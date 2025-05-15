from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
# from gym_unity.envs import UnityToGymWrapper
# from mlagents_envs.environment import UnityEnvironment

class UnityEnvironmentWrapper:
    def __init__(self, env_path):
        self.unity_env = UnityEnvironment(env_path, no_graphics_monitor=False, no_graphics=False)
        self.env = UnityToGymWrapper(self.unity_env, uint8_visual=True, allow_multiple_obs=True)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()

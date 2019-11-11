import gym, requests
import numpy as np

from gym import spaces
from time import sleep

class AimSensors(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, env_ip):
        super(AimSensors, self).__init__()

        # connect to the backend to retrieve state and action information

        ready = False
        while not ready:
            try:
                flows, f_state, p_state = self.get_state(env_ip)
                actions, action_categories = self.get_actions(env_ip)
                stack_size = len(f_state)
                frame_size = len(f_state[0][0])
                action_size = len(actions)
                ready = True
            except Exception as e:
                print(e)
                print('Connecting to {0}...'.format(env_ip))
                sleep(1)

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1, frame_size), dtype=np.float32)
        self.action_space = spaces.Discrete(action_size)
        self.

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass

    def get_state(self, env_ip):
        flows, f_state, p_state = requests.get('http://{0}/state'.format(env_ip)).json()
        return flows, f_state, p_state

    def get_actions(self, env_ip):
        return requests.get('http://{0}/actions'.format(env_ip)).json()
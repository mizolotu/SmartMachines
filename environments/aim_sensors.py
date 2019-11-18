import gym, requests
import numpy as np

from gym import spaces
from time import sleep, time

class AimSensors(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, env_ip, attack_vectors, delay, cfg):
        super(AimSensors, self).__init__()
        self.env_ip = env_ip
        self.attack_vectors = attack_vectors
        self.delay = delay
        self.flows = []

        # connect to the backend to retrieve state and action information

        ready = False
        while not ready:
            try:
                flows, f_state, p_state = self._get_state()
                actions, action_categories = self._get_actions()
                stack_size = len(f_state[0])
                frame_size = len(f_state[0][0])
                action_size = len(actions)
                self._set_gamma(cfg['gamma'])
                for key in cfg['coeff'].keys():
                    self._set_coeff(key, cfg['coeff'][key]['a'], cfg['coeff'][key]['b'])
                ready = True
                print('Environment {0} has been initialized.'.format(env_ip))
            except Exception as e:
                print(e)
                print('Trying to connect to {0}...'.format(env_ip))
                sleep(1)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(stack_size, frame_size), dtype=np.float32)
        self.action_space = spaces.Discrete(action_size)

    def step(self, action_list):
        t_start = time()
        self._take_action(action_list)
        t_action = time()
        if t_action < t_start + self.delay:
            sleep(t_start + self.delay - t_action)
        f_scores, f_counts = self._get_score()
        self.flows, f_state, p_state = self._get_state()
        t_state = time()
        # print('Step in {0} took {1} seconds'.format(self.env_ip, t_state - t_start))
        return f_state, f_scores, False, self.flows

    def reset(self):
        self._reset_env()
        self._start_episode()
        self.flows, f_state, p_state = self._get_state()
        return f_state, self.flows

    def render(self, mode='human', close=False):
        pass

    def _get_state(self):
        flows, f_state_framed, p_state_framed = requests.get('http://{0}/state'.format(self.env_ip)).json()
        f_state = []
        n = len(f_state_framed[0])
        for i in range(n):
            series = [frame[i] for frame in f_state_framed]
            f_state.append(series)
        return flows, f_state, p_state_framed

    def _get_actions(self):
        return requests.get('http://{0}/actions'.format(self.env_ip)).json()

    def _set_gamma(self, g):
        return requests.post('http://{0}/dns_gamma'.format(self.env_ip), json={'gamma': g}).json()

    def _set_coeff(self, attack, a, b):
        return requests.post('http://{0}/score_coeff/{1}'.format(self.env_ip, attack), json={'a': a, 'b': b}).json()

    def _reset_env(self):
        return requests.get('http://{0}/reset'.format(self.env_ip)).json()

    def _start_episode(self):
        attack = self.attack_vectors[np.random.randint(0, len(self.attack_vectors))]
        r = requests.post('http://{0}/start_episode'.format(self.env_ip), json={'attack': attack, 'start': self.delay})
        return r.json()

    def _take_action(self, ai):
        if type(ai).__name__ != 'list':
            ai = ai.tolist()
        requests.post('http://{0}/action'.format(self.env_ip), json={'patterns': self.flows, 'action_inds': ai})

    def _get_log(self):
        requests.get('http://{0}/log'.format(self.env_ip))

    def _get_score(self):
        ready = False
        while not ready:
            try:
                data = requests.get('http://{0}/score'.format(self.env_ip), json={'flows': self.flows}).json()
                ready = True
            except Exception as e:
                print(e)
        return data
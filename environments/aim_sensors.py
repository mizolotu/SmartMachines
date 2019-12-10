import gym, requests
import numpy as np

from gym import spaces
from time import sleep, time

class AimSensors(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, env_ip, attack_vectors, cfg, delay=0.0):
        super(AimSensors, self).__init__()
        self.env_ip = env_ip
        self.attack_vectors = attack_vectors
        self.delay = delay
        self.flows = []
        self.attack_flows = []

        # connect to the backend to retrieve state and action information

        ready = False
        while not ready:
            try:
                flows, f_state, p_state, infected = self._get_state()
                actions, action_categories = self._get_actions()
                frame_size = len(f_state[0])
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
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1, frame_size), dtype=np.float32)
        self.action_space = spaces.Discrete(action_size)

    def step(self, action_list):
        action_list = np.asarray(action_list, dtype=int)

        t_start = time()
        self._take_action(action_list)
        t_action = time()
        if t_action < t_start + self.delay:
            sleep(t_start + self.delay - t_action)
        f_scores, f_counts = self._get_score()
        self.flows, f_state, p_state, infected_devices = self._get_state()
        n_normal = len([x for x in f_scores if x > 0])
        n_attack = len([x for x in f_scores if x < 0])
        n_infected = len(infected_devices)
        return f_state, f_scores, False, {'flows': self.flows, 'n_normal_flows': n_normal, 'n_attack_flows': n_attack, 'n_infected': n_infected}

    def reset(self):
        self._reset_env()
        self._start_episode()
        self.flows, f_state, p_state, infected = self._get_state()
        return f_state, self.flows

    def render(self, mode='human', close=False):
        pass

    def _get_state(self):
        flows, f_state, p_state, infected_devices = requests.get('http://{0}/state'.format(self.env_ip)).json()

        # get attack flows

        if self.attack_flows == []:
            log = self._get_log()
            af = log['debug']['attack_flows']
            for item in af:
                self.attack_flows.append('.'.join(item))
                self.attack_flows.append('.'.join(item[::-1]))

        return flows, f_state, p_state, infected_devices

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
        return requests.get('http://{0}/log'.format(self.env_ip)).json()

    def _get_score(self):
        ready = False
        while not ready:
            try:
                data = requests.get('http://{0}/score'.format(self.env_ip), json={'flows': self.flows}).json()
                ready = True
            except Exception as e:
                print(e)
        return data
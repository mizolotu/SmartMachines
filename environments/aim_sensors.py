import gym, requests, json
import numpy as np
import os.path as osp

from gym import spaces
from time import sleep, time

class AimSensors(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, env_ip, attack_vectors, cfg_dir, delay=0.25, cfg_episodes=10, cfg_steps=100):
        super(AimSensors, self).__init__()
        self.env_ip = env_ip
        self.attack_vectors = attack_vectors
        self.delay = delay
        self.flows = []
        self.attack_flows = []

        # reconfigure backend if needed

        env_cfg_file = osp.join(cfg_dir, '{0}.json'.format(env_ip.split(':')[0]))
        try:
            with open(env_cfg_file, 'r') as f:
                cfg = json.load(f)
        except Exception as e:
            print(e)
            cfg = {}
        if cfg == {}:
            cfg_episodes = np.maximum(cfg_episodes, 2)
        if cfg_episodes > 0:
            cfg = {}
            coeff = {}
            gamma = 0
            for attack in attack_vectors:
                coeff_attack = - np.ones(2)
                while np.any(coeff_attack < 0):
                    print('Configuring backend {0} for {1}'.format(self.env_ip, attack))
                    n_arr = None
                    for e in range(cfg_episodes):
                        if n_arr is None:
                            n_arr = self._calculate_coefficients(attack, cfg_steps)
                        else:
                            n_arr = np.vstack([n_arr, self._calculate_coefficients(attack, cfg_steps)])
                    g = n_arr[:, 3] / n_arr[:, 2]  # number of resolved packets / number of dns replies
                    a_normal = n_arr[:, 3] + n_arr[:, 4]
                    b_normal = n_arr[:, 2] * g
                    a_attack = n_arr[:, 0]
                    b_attack = n_arr[:, 1]
                    coeff_attack = np.array([np.mean(a_normal / a_attack), np.mean(b_normal / b_attack)])

                print('Coefficients for {0}: alpha = {1}, beta = {2}'.format(attack, coeff_attack[0], coeff_attack[1]))
                coeff[attack] = {'a': coeff_attack[0], 'b': coeff_attack[1]}
                gamma = np.mean(g)
            cfg['coeff'] = coeff
            cfg['gamma'] = gamma
            with open(env_cfg_file, 'w') as f:
                json.dump(cfg, f)

        # retrieve state and action information

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
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(frame_size,), dtype=np.float32)
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

    def _calculate_coefficients(self, attack, cfg_steps):
        self._reset_env()
        self._start_episode(attack)
        sum_deltas = np.zeros(5)
        count_deltas = 0
        self.flows, f_state, p_state, infected_devices = self._get_state()
        t_state = time()
        for step in range(cfg_steps):
            action_list = [0 for _ in self.flows]
            self._take_action(action_list)
            t_action = time()
            if t_action < t_state + self.delay:
                sleep(t_state + self.delay - t_action)
            _, counts = self._get_score()
            self.flows, f_state, p_state, infected_devices = self._get_state()
            count_deltas += 1
            sum_deltas += np.array(counts)
        n_cf = sum_deltas / count_deltas
        return n_cf

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

    def _start_episode(self, attack=None):
        if attack is None:
            attack = self.attack_vectors[np.random.randint(0, len(self.attack_vectors))]
        else:
            assert attack in self.attack_vectors
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
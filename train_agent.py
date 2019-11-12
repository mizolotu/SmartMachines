import json

from environments.aim_sensors import AimSensors
from algorithms.bs_common.vec_env import DummyVecEnv
from algorithms.flow_ppo.ppo import learn

def create_env(env, attack_vectors, delay, cfg):
    return lambda : AimSensors(env, attack_vectors, delay, cfg)

if __name__ == '__main__':

    # environment backend ips

    env_urls = [
        # '192.168.176.10:5000',
        '192.168.176.11:5000',
        '192.168.176.12:5000',
        # '192.168.176.13:5000',
        # '192.168.176.14:5000',
        # '192.168.176.15:5000'
    ]

    # attack vectors

    attack_vectors = [
        'botnet_attack',
        # 'exfiltration_attack',
        # 'scan_attack',
        # 'exploit_attack',
        # 'slowloris_attack'
    ]

    # experiment parameters

    episode_duration = 20
    t_start = 5
    delay = 0.0
    n_episodes = 1000
    save_interval = 10
    cfg_file = 'config.json'
    with open(cfg_file, 'r') as f:
        cfg = json.load(f)

    n_steps = int(episode_duration / delay)
    n_total_steps = int(n_episodes * episode_duration / delay)
    alg_kwargs = {
        'network': 'mlp',
        'nsteps': n_steps,
        'total_timesteps': n_total_steps,
        'save_interval': save_interval
    }

    env_fns = [create_env(env_url, attack_vectors, delay, cfg) for env_url in env_urls]
    env = DummyVecEnv(env_fns)
    learn(env=env, **alg_kwargs)


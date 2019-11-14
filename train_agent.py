import sys, json

from environments.aim_sensors import AimSensors
from algorithms.bs_common.vec_env import DummyVecEnv
from algorithms.bs_common.vec_env import SubprocVecEnv
from algorithms.flow_ppo.ppo import learn as learn_ppo
from algorithms.flow_dqn.deepq import learn as learn_dqn

def create_env(env, attack_vectors, delay, cfg):
    return lambda : AimSensors(env, attack_vectors, delay, cfg)

if __name__ == '__main__':

    # parse arguments, substitute to parsearg later

    algorithm = sys.argv[1]
    env_inds = [int(idx) for idx in sys.argv[2].split(',')]
    av_inds = [int(idx) for idx in sys.argv[3].split(',')]

    # environment backend ips

    env_urls = [
        '192.168.176.10:5000',
        '192.168.176.11:5000',
        '192.168.176.12:5000',
        '192.168.176.13:5000',
        '192.168.176.14:5000',
        '192.168.176.15:5000'
    ]

    # attack vectors

    attack_vectors = [
        'botnet_attack',
        'exfiltration_attack',
        'scan_attack',
        'exploit_attack',
        'slowloris_attack'
    ]

    # experiment parameters

    policy = 'mlp'
    delay = 0.0
    n_steps = 30
    n_episodes = 10000
    n_total_steps = n_episodes * n_steps
    save_interval = 10
    cfg_file = 'config.json'
    with open(cfg_file, 'r') as f:
        cfg = json.load(f)

    if algorithm == 'ppo':
        learn = learn_ppo
    elif algorithm == 'dqn':
        learn = learn_dqn

    alg_kwargs = {
        'network': policy,
        'nsteps': n_steps,
        'total_timesteps': n_total_steps,
        'save_interval': save_interval,
        'load_path': 'logs/{0}/{1}/checkpoints/last'.format(algorithm, policy)
    }

    envs = [env_urls[idx] for idx in env_inds]
    avs = [attack_vectors[idx] for idx in av_inds]
    env_fns = [create_env(env, avs, delay, cfg) for env in envs]
    env = SubprocVecEnv(env_fns)
    learn(env=env, **alg_kwargs)


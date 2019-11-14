import sys, json

from environments.aim_sensors import AimSensors
from algorithms.bs_common.vec_env import DummyVecEnv
from algorithms.flow_ppo.ppo import learn

def create_env(env, attack_vectors, delay, cfg):
    return lambda : AimSensors(env, attack_vectors, delay, cfg)

if __name__ == '__main__':

    # parse arguments

    if len(sys.argv) == 1:
        env_inds = [1, 2]
        av_inds = [1]
    elif len(sys.argv) == 2:
        env_inds = [int(idx) for idx in sys.argv[1].split(',')]
        av_inds = [1]
    elif len(sys.argv) == 3:
        env_inds = [int(idx) for idx in sys.argv[1].split(',')]
        av_inds = [int(idx) for idx in sys.argv[2].split(',')]
    else:
        print('What?')
        sys.exit(1)

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
    episode_duration = 20
    t_start = 5
    delay = 0.0
    n_steps = 100
    n_episodes = 1000
    n_total_steps = n_episodes * n_steps
    save_interval = 10
    cfg_file = 'config.json'
    with open(cfg_file, 'r') as f:
        cfg = json.load(f)

    alg_kwargs = {
        'network': policy,
        'nsteps': n_steps,
        'total_timesteps': n_total_steps,
        'save_interval': save_interval,
        'value_network': 'shared',
        'load_path': 'logs/{0}/checkpoints/last'.format(policy)
    }

    envs = [env_urls[idx] for idx in env_inds]
    avs = [attack_vectors[idx] for idx in av_inds]
    env_fns = [create_env(env, avs, delay, cfg) for env in envs]
    env = DummyVecEnv(env_fns)
    learn(env=env, **alg_kwargs)


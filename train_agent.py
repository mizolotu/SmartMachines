import gym
from environments.aim_sensors import AimSensors
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

if __name__ == '__main__':

    # environment backend ips

    envs = [
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

    n_cycles = len(attack_vectors)
    episode_duration = 30
    t_start = 5
    delay = 0.25

    #env = DummyVecEnv([lambda: AimSensors(envs[0])])
    env = SubprocVecEnv([lambda: AimSensors(env) for env in envs])

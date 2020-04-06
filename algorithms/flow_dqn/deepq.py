import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import algorithms.bs_common.tf_util as U
from algorithms.bs_common.tf_util import load_variables, save_variables
from algorithms import logger
from algorithms.bs_common.schedules import LinearSchedule
from algorithms.bs_common import set_global_seeds

from algorithms.flow_dqn import build_train, build_act
from algorithms.flow_dqn.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from algorithms.flow_dqn.utils import ObservationInput

from algorithms.bs_common.tf_util import get_session
from algorithms.flow_dqn.models import build_q_func


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


def learn(env,
          network,
          seed=None,
          lr=1e-4,
          nupdates=10000,
          nsteps=100,
          buffer_size=50000,
          exploration_fraction=0.0,
          exploration_final_eps=0.0,
          train_freq=1,
          batch_size=512,
          print_freq=1,
          save_interval=10,
          log_prefix='',
          checkpoint_path=None,
          learning_starts=10,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          **network_kwargs
            ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    batch_size: int
        size of a batch sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug = build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer

    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = nsteps*nupdates
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters, initial_p=prioritized_replay_beta0, final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None

    # Create the schedule for exploration starting from 1.

    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * nupdates * nsteps // env.nremotes), initial_p=1.0, final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.

    U.initialize()
    update_target()
    episode_rewards = [0.0 for _ in range(env.nremotes)]
    normal_flows = [0.0]
    attack_flows = [0.0]
    infected_devices = [0.0]
    saved_mean_reward = None
    obs, flows = env.reset()
    reset = True

    log_path = os.path.abspath('logs/{0}/dqn/{1}'.format(log_prefix, network))
    tb_path = os.path.join(log_path, 'tb')
    checkpoint_path = os.path.join(log_path, 'checkpoints')
    format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
    logger.configure(os.path.abspath(log_path), format_strs)
    for the_file in os.listdir(tb_path):
        file_path = os.path.join(tb_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "last")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None and os.path.isfile(load_path):
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        for t in range(nupdates * nsteps // env.nremotes):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            env_actions = []
            for e in range(len(obs)):
                env_actions.append([])
                for i in range(len(obs[e])):
                    action = act(np.array(obs[e][i])[None], update_eps=update_eps, **kwargs)[0]
                    env_actions[-1].append(action)
            reset = False
            new_obs, rew, _, infos = env.step(env_actions)
            new_flows = [info['flows'] for info in infos]
            n_normal = [np.sum([v for v in info['stats']['n_normal'].values()]) for info in infos]
            n_attack = [np.sum([v for v in info['stats']['n_attack'].values()]) for info in infos]
            n_infected = [info['stats']['n_infected'] for info in infos]

            # Store transition in the replay buffer

            for e in range(len(obs)):
                for i in range(len(flows[e])):
                    if flows[e][i] in new_flows[e]:
                        new_i = new_flows[e].index(flows[e][i])
                        done_i = 0
                        obs_next = new_obs[e][new_i]
                    else:
                        done_i = 1
                        obs_next = obs[e][i]
                    replay_buffer.add(obs[e][i], env_actions[e][i], rew[e][i], obs_next, done_i)

            obs = new_obs
            flows = new_flows

            for e,r in enumerate(rew):
                if len(rew[e]) > 0:
                    episode_rewards[-1 - e] += np.mean(rew[e])

            normal_flows[-1] += np.mean(n_normal)
            attack_flows[-1] += np.mean(n_attack)
            infected_devices[-1] = np.mean(n_infected)
            if t > 0 and t % nsteps == 0:
                done = True
                obs, flows = env.reset()
                for e in range(env.nremotes):
                    episode_rewards[-1 - e] /= nsteps
                print(episode_rewards)
                normal_flows[-1] /= nsteps
                attack_flows[-1] /= nsteps
                episode_rewards.extend([0.0 for e in range(env.nremotes)])
                normal_flows.append(0.0)
                attack_flows.append(0.0)
                infected_devices.append(0.0)
                reset = True
            else:
                done = False

            if t > learning_starts * nsteps and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts * nsteps and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-11:-1]), 2)
            min_100ep_reward = round(np.min(episode_rewards[-11:-1]), 2) if len(episode_rewards) > 1 else np.nan
            max_100ep_reward = round(np.max(episode_rewards[-11:-1]), 2) if len(episode_rewards) > 1 else np.nan
            mean_100ep_normal_flows = round(np.mean(normal_flows[-6:-1]), 2)
            mean_100ep_attack_flows = round(np.mean(attack_flows[-6:-1]), 2)
            mean_100ep_infected_devices = round(np.mean(infected_devices[-6:-1]), 2)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t * env.nremotes)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("normal flows", mean_100ep_normal_flows)
                logger.record_tabular("attack flows", mean_100ep_attack_flows)
                logger.record_tabular("reward", mean_100ep_reward)
                logger.record_tabular("reward_min", min_100ep_reward)
                logger.record_tabular("reward_max", max_100ep_reward)
                logger.record_tabular("infected devices", mean_100ep_infected_devices)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (save_interval is not None and t > (learning_starts * nsteps) and t % (save_interval * nsteps) == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(saved_mean_reward, mean_100ep_reward))
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    return act

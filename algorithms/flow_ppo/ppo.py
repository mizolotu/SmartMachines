import os
import time
import numpy as np
import os.path as osp
from algorithms import logger
from collections import deque
from algorithms.bs_common import explained_variance, set_global_seeds
from algorithms.bs_common.policies import build_policy
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from algorithms.flow_ppo.runner import Runner


def constfn(val):
    def f(_):
        return val
    return f

def learn(network, env,
    nsteps=100,
    total_timesteps=1e6,
    eval_env=None,
    seed=None,
    ent_coef=0.0,
    lr=lambda f: 1e-3 * f,
    vf_coef=0.5,
    max_grad_norm=0.5,
    gamma=0.99,
    lam=0.95,
    log_interval=1,
    nminibatches=4,
    nbatch_train=100,
    noptepochs=4,
    cliprange=0.2,
    save_interval=0,
    load_path=None,
    model_fn=None,
    update_fn=None,
    init_fn=None,
    mpi_rank_weight=1,
    comm=None,
    log_prefix='',
    **network_kwargs
):

    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from algorithms.flow_ppo.model import Model
        model_fn = Model

    model = model_fn(
        policy=policy,
        ob_space=ob_space,
        ac_space=ac_space,
        nbatch_act=nenvs,
        nbatch_train=nbatch_train,
        nsteps=nbatch_train,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        comm=comm,
        mpi_rank_weight=mpi_rank_weight
    )

    if load_path is not None and osp.isfile(load_path):
        model.load(load_path)
        print('Loaded model from {0}'.format(load_path))
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    epinfobuf = deque(maxlen=log_interval*nenvs)

    if init_fn is not None:
        init_fn()

    # Start total timer
    tfirststart = time.perf_counter()

    format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
    logger.configure(os.path.abspath('logs/{0}/ppo/{1}'.format(log_prefix, network)), format_strs)

    nupdates = total_timesteps # //nbatch
    for update in range(1, nupdates+1):

        #assert nbatch % nminibatches == 0
        # Start timer

        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates

        # Calculate the learning rate

        lrnow = lr(frac)

        # Calculate the cliprange

        cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch

        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632

        nbatch = obs.shape[0]

        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')
        epinfobuf.extend(epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.

        mblossvals = []
        if states is None: # nonrecurrent version

            # Index of each element of batch_size
            # Create the indices array

            inds = np.arange(nbatch)
            for _ in range(noptepochs):

                # Randomize the indexes

                np.random.shuffle(inds)

                # 0 to batch_size with batch_train_size step

                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    if len(mbinds) < nbatch_train:
                        mbinds_ = np.random.choice(inds, nbatch_train - len(mbinds))
                        mbinds = np.hstack([mbinds, mbinds_])
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))

        else: # recurrent version

            # Index of each element of batch_size
            # Create the indices array

            envinds = np.arange(nenvs)
            env_batch_sizes = [epinfos[i]['batch_size'] for i in envinds]
            env_batch_starts = np.cumsum(env_batch_sizes) - env_batch_sizes

            for _ in range(noptepochs):

                np.random.shuffle(envinds)

                for env_idx in envinds:
                    inds = env_batch_starts[env_idx] + np.arange(epinfos[env_idx]['batch_size'])
                    for start in range(0, epinfos[env_idx]['batch_size'], nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        if len(mbinds) < nbatch_train:
                            mbinds_ = np.random.choice(inds, nbatch_train - len(mbinds))
                            mbinds = np.hstack([mbinds, mbinds_])
                            mask_length = len(mbinds_)
                        else:
                            mask_length = 0
                    mbobs = obs[mbinds]
                    slices_ = [arr[mbinds] for arr in (returns, masks, actions, values, neglogpacs)]
                    slices = slices_.copy()
                    if mask_length > 0:
                        slices[1][-mask_length] = True
                    mbstates = states[mbinds[0:1]]
                    mblossvals.append(model.train(lrnow, cliprangenow, mbobs, *slices, mbstates))

            #nminibatches = nenvs
            #assert nenvs % nminibatches == 0
            #envsperbatch = nenvs // nminibatches
            #envinds = np.arange(nenvs)
            #flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            #for _ in range(noptepochs):
            #    np.random.shuffle(envinds)
            #    for start in range(0, nenvs, envsperbatch):
            #        end = start + envsperbatch
            #        mbenvinds = envinds[start:end]
            #        mbflatinds = flatinds[mbenvinds].ravel()
            #        slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
            #        mbstates = states[mbenvinds]
            #        mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.perf_counter()
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)

        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("stats/updates", update)
            logger.logkv("stats/timestamps", update * nenvs * nsteps)
            logger.logkv("stats/reward", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv("stats/reward_min", np.min([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv("stats/reward_max", np.max([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv("stats/infected_devices", safemean([epinfo['n_infected'] for epinfo in epinfobuf]))
            logger.logkv("stats/fps", fps)
            logger.logkv("stats/explained_variance", float(ev))
            logger.logkv('stats/time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/' + lossname, lossval)
            nnormal = [epinfo['normal_vs_attack']['normal_dns'] + epinfo['normal_vs_attack']['normal_device'] + epinfo['normal_vs_attack']['normal_admin']for epinfo in epinfobuf]
            logger.logkv("normal/dns", safemean([epinfo['normal_vs_attack']['normal_dns'] for epinfo in epinfobuf]))
            logger.logkv("normal/device", safemean([epinfo['normal_vs_attack']['normal_device'] for epinfo in epinfobuf]))
            logger.logkv("normal/admin", safemean([epinfo['normal_vs_attack']['normal_admin'] for epinfo in epinfobuf]))
            nattack = [epinfo['normal_vs_attack']['attack_target'] + epinfo['normal_vs_attack']['attack_cc'] for epinfo in epinfobuf]
            logger.logkv("attack/target", safemean([epinfo['normal_vs_attack']['attack_target'] for epinfo in epinfobuf]))
            logger.logkv("attack/cc", safemean([epinfo['normal_vs_attack']['attack_cc'] for epinfo in epinfobuf]))
            logger.logkv("stats/normal", safemean(nnormal))
            logger.logkv("stats/normal_min", np.min(nnormal))
            logger.logkv("stats/normal_max", np.max(nnormal))
            logger.logkv("stats/attack", safemean(nattack))
            logger.logkv("stats/attack_min", np.min(nattack))
            logger.logkv("stats/attack_max", np.max(nattack))
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath_last = osp.join(checkdir, 'last')
            print('Saving to {0}'.format(savepath_last))
            model.save(savepath_last)
    return model

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
import numpy as np
from algorithms.bs_common.runners import AbstractEnvRunner
from time import time

class Runner(AbstractEnvRunner):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        self.obs, self.flows = self.env.reset()
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_flows, mb_states = [],[],[],[],[],[],[],[]
        epinfos = []
        normal_vs_attack = [{} for _ in range(self.nenv)]

        # For n in range number of steps

        for _ in range(self.nsteps):

            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init

            actions = [[] for _ in range(self.nenv)]
            values = [[] for _ in range(self.nenv)]
            states = [[] for _ in range(self.nenv)]
            neglogpacs = [[] for _ in range(self.nenv)]
            dones = [[] for _ in range(self.nenv)]

            obs_lens = [self.obs[i].shape[0] for i in range(self.nenv)]
            i_max = np.argmax(obs_lens)
            obs_padded = np.zeros((self.nenv, *self.obs[i_max].shape))
            for i in range(self.nenv):
                obs_padded[i, :self.obs[i].shape[0], :] = self.obs[i]
            for i in range(self.obs[i_max].shape[0]):
                actions_i, values_i, states_i, neglogpacs_i = self.model.step(obs_padded[:, i, :], S=self.states, M=self.dones)
                for e in range(self.nenv):
                    if i < self.obs[e].shape[0]:
                        actions[e].append(actions_i[e])
                        values[e].append(values_i[e])
                        if states_i is not None:
                            self.states[e] = states_i[e]
                            states[e].append(states_i[e])
                        neglogpacs[e].append(neglogpacs_i[e])
                #for j in range(len(self.obs[i])):
                #    masks = [False]
                #    actions_, values_, self.states[i], neglogpacs_ = self.model.step(self.obs[i][j, :], S=self.states[i], M=masks)
                #    actions[i].extend(actions_)
                #    values[i].extend(values_)
                #    states[i].append(self.states[i])
                #    dones[i].append(self.dones[i])
                #    neglogpacs[i].extend(neglogpacs_)
            mb_obs.append(self.obs.copy())
            mb_states.append(states.copy())
            mb_flows.append(self.flows.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(dones)

            # Take actions in env and look the results
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            for e in range(self.nenv):
                for key_1 in ['normal', 'attack']:
                    for key_2 in infos[e]['stats']['n_{0}'.format(key_1)].keys():
                        key = '{0}_{1}'.format(key_1, key_2)
                        if key not in normal_vs_attack[e].keys():
                            normal_vs_attack[e][key] = infos[e]['stats']['n_{0}'.format(key_1)][key_2]
                        else:
                            normal_vs_attack[e][key] += infos[e]['stats']['n_{0}'.format(key_1)][key_2]
            for e in range(self.nenv):
                for key in normal_vs_attack[e].keys():
                    normal_vs_attack[e][key] /= self.nsteps
            n_infected = [info['stats']['n_infected'] for info in infos]
            self.flows = [info['flows'] for info in infos]
            mb_rewards.append(rewards)

        # batch of steps to batch of rollouts

        obs_b = []
        states_b = []
        actions_b = []
        returns_b = []
        values_b = []
        neglopacs_b = []

        for e in range(self.nenv):
            flows = [each[e] for each in mb_flows]
            all_flows = []
            for state_flows in flows:
                all_flows.extend(state_flows)
            all_flows = list(set(all_flows))
            n_flows = len(all_flows)
            obs_per_flow = [[] for _ in range(n_flows)]
            states_per_flow = [[] for _ in range(n_flows)]
            actions_per_flow = [[] for _ in range(n_flows)]
            rewards_per_flow = [[] for _ in range(n_flows)]
            values_per_flow = [[] for _ in range(n_flows)]
            returns_per_flow = [[] for _ in range(n_flows)]
            neglopacs_per_flow = [[] for _ in range(n_flows)]
            for i, flow in enumerate(all_flows):
                for j in range(len(mb_obs)):
                    flows = mb_flows[j][e]
                    if type(flows).__name__ != 'list':
                        flows = flows.tolist()
                    if flow in flows:
                        idx = flows.index(flow)
                        obs_ = mb_obs[j][e]
                        obs_per_flow[i].append(obs_[idx])
                        state_ = mb_states[j][e]
                        if state_:
                            states_per_flow[i].append(state_[idx])
                        actions_per_flow[i].append(mb_actions[j][e][idx])
                        rewards_per_flow[i].append(mb_rewards[j][e][idx])
                        neglopacs_per_flow[i].append(mb_neglogpacs[j][e][idx])
                        values_per_flow[i].append(mb_values[j][e][idx])
            obs_per_flow = [np.array(x, ndmin=2) for x in obs_per_flow]
            states_per_flow = [np.array(x, ndmin=2) for x in states_per_flow]
            actions_per_flow = [np.array(x) for x in actions_per_flow]
            rewards_per_flow = [np.array(x, ndmin=1).reshape(len(x), 1) for x in rewards_per_flow]
            neglopacs_per_flow = [np.vstack(x) for x in neglopacs_per_flow]
            values_per_flow = [np.vstack(x) for x in values_per_flow]

            for i, flow in enumerate(all_flows):
                advantages = np.zeros_like(rewards_per_flow[i])
                nsteps = advantages.shape[0]
                for t in reversed(range(nsteps)):
                    if t == nsteps - 1:
                        lastgaelam = rewards_per_flow[i][t, 0] - values_per_flow[i][t, 0]
                        advantages[t, 0] = lastgaelam
                    else:
                        delta = rewards_per_flow[i][t, 0] + self.gamma * values_per_flow[i][t + 1, 0] - values_per_flow[i][t, 0]
                        lastgaelam = delta + self.gamma * self.lam * lastgaelam
                        advantages[t, 0] = lastgaelam
                returns_per_flow[i] = advantages + values_per_flow[i]

            scores = []
            for i in range(n_flows):
                obs_b.extend(obs_per_flow[i])
                states_b.extend(states_per_flow[i])
                actions_b.extend(actions_per_flow[i])
                returns_b.extend(returns_per_flow[i])
                values_b.extend(values_per_flow[i])
                neglopacs_b.extend(neglopacs_per_flow[i])
                scores.extend(rewards_per_flow[i])

            epinfos.append({
                'r': np.mean(scores),
                'normal_vs_attack': normal_vs_attack[e],
                'n_infected': n_infected[e]
            })

        mb_obs = np.array(obs_b, ndmin=2)
        mb_states = np.array(states_b, ndmin=2)
        if np.any(np.array(mb_states.shape) == 0):
            mb_states = None
        print(mb_obs.shape, mb_states)
        mb_actions = np.array(actions_b)
        mb_returns = np.vstack(returns_b)
        mb_masks = np.zeros_like(mb_returns)
        mb_values = np.vstack(values_b)
        mb_neglogpacs = np.vstack(neglopacs_b)

        return (mb_obs, *map(sf01, (mb_returns, mb_masks, mb_actions, mb_values, mb_neglogpacs)), mb_states, epinfos)

# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    if len(s) == 1:
        sw = arr
    elif len(s) > 1:
        sw = arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
    return sw



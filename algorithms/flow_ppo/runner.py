import numpy as np
from algorithms.bs_common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        self.obs, self.flows = self.env.reset()
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_flows = [],[],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions = [[] for _ in range(self.nenv)]
            values = [[] for _ in range(self.nenv)]
            self.states = [[] for _ in range(self.nenv)]
            neglogpacs = [[] for _ in range(self.nenv)]
            for i in range(self.nenv):
                actions[i], values[i], self.states[i], neglogpacs[i] = self.model.step(self.obs[i], S=self.states, M=self.dones)
                #print(len(actions[i]))
                #for f, a in zip(self.flows[i], actions[i]):
                #    print('{0}: {1}'.format(f, a))
            mb_obs.append(self.obs.copy())
            mb_flows.append(self.flows.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs, rewards, self.dones, self.flows = self.env.step(actions)
            mb_rewards.append(rewards)

        #batch of steps to batch of rollouts

        #mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        #mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        #mb_actions = np.asarray(mb_actions)
        #mb_values = np.asarray(mb_values, dtype=np.float32)
        #mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)

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
                        state_ = mb_obs[j][e]
                        states_per_flow[i].append(state_[idx])
                        actions_per_flow[i].append(mb_actions[j][e][idx])
                        rewards_per_flow[i].append(mb_rewards[j][e][idx])
                        neglopacs_per_flow[i].append(mb_neglogpacs[j][e][idx])
                        values_per_flow[i].append(mb_values[j][e][idx])
            states_per_flow = [np.array(x, ndmin=3) for x in states_per_flow]
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
            n_actions = 0
            for i in range(n_flows):
                states_b.extend(states_per_flow[i])
                actions_b.extend(actions_per_flow[i])
                returns_b.extend(returns_per_flow[i])
                values_b.extend(values_per_flow[i])
                neglopacs_b.extend(neglopacs_per_flow[i])
                scores.extend(rewards_per_flow[i])
                n_actions += len(actions_per_flow[i])
            epinfos.append({'r': np.mean(scores), 'l': n_actions})

        mb_obs = np.array(states_b, ndmin=3)
        mb_actions = np.array(actions_b)
        mb_returns = np.vstack(returns_b)
        mb_masks = np.zeros_like(mb_returns)
        mb_values = np.vstack(values_b)
        mb_neglogpacs = np.vstack(neglopacs_b)

        #mb_dones = np.asarray(mb_dones, dtype=np.bool)
        #last_values = [[] for _ in range(self.nenv)]
        #for i in range(self.nenv):
        #    last_values[i] = self.model.value(self.obs[i], S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        #mb_returns = np.zeros_like(mb_rewards)
        #mb_advs = np.zeros_like(mb_rewards)
        #lastgaelam = 0
        #for t in reversed(range(self.nsteps)):
        #    if t == self.nsteps - 1:
        #        nextnonterminal = 1.0 - self.dones
        #        nextvalues = last_values
        #    else:
        #        nextnonterminal = 1.0 - mb_dones[t+1]
        #        nextvalues = mb_values[t+1]
        #    delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
        #    mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        #mb_returns = mb_advs + mb_values

        return (mb_obs, *map(sf01, (mb_returns, mb_masks, mb_actions, mb_values, mb_neglogpacs)), None, epinfos)

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



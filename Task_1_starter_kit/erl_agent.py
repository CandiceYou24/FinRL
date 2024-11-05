import os
import torch as th
from typing import Tuple
from copy import deepcopy
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from erl_config import Config
from erl_replay_buffer import ReplayBuffer

#DQN
from erl_net import QNetTwin, QNetTwinDuel
#PPO
from erl_net import ActorPPO, ActorDiscretePPO, CriticPPO
#SAC
from erl_net import ActorSAC, ActorDiscreteSAC, CriticEnsemble

import numpy as np

from typing import Union, Tuple

TEN = th.Tensor


def get_optim_param(optimizer: th.optim) -> list:  # backup
    params_list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, th.Tensor)])
    return params_list


'''DQN'''

class AgentDoubleDQN:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        
        self.act_class = getattr(self, "act_class", QNetTwin)
        
        ## Critic is set to the same instance as actor
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        
        # self.act_class =  QNetTwin
        # self.cri_class = None
        
        self.gamma = args.gamma  # discount factor of future rewards
        self.num_envs = args.num_envs  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.batch_size = args.batch_size  # num of transitions sampled from replay buffer.
        self.repeat_times = args.repeat_times  # repeatedly update network using ReplayBuffer
        self.reward_scale = args.reward_scale  # an approximate target reward usually be closed to 256
        self.learning_rate = args.learning_rate  # the learning rate for network updating
        self.if_off_policy = args.if_off_policy  # whether off-policy or on-policy of DRL algorithm
        self.clip_grad_norm = args.clip_grad_norm  # clip the gradient after normalization
        self.soft_update_tau = args.soft_update_tau  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = args.state_value_tau  # the tau of normalize for value and state

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        
        print("  Base Agent: DoubleDQN")
        print("  device", self.device)
        print("  batch_size", self.batch_size)
        print("  num_envs", self.num_envs)
        print("  state_dim", state_dim)
        print("  net_dim", net_dims)

        '''network'''
        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        
#         self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
#         self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
#             if cri_class else self.act

        # Define directly
        self.act = self.act_target = QNetTwin(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = self.act
        
    
        print("network initialized")
        
        '''optimizer'''
        
        print(self.act.parameters())
        self.act_optimizer = th.optim.AdamW(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.AdamW(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer
        from types import MethodType  # built-in package of Python3
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)
        print(self.act_optimizer.parameters)

        self.criterion = th.nn.SmoothL1Loss(reduction="mean")
        
        print("optimizer initialized")

        """save and load"""
        self.save_attr_names = {'act', 'act_target', 'act_optimizer', 'cri', 'cri_target', 'cri_optimizer'}

        self.act_target = self.cri_target = deepcopy(self.act)
        self.act.explore_rate = getattr(args, "explore_rate", 1 / 32)

    def get_obj_critic(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with th.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)

            next_qs = th.min(*self.cri_target.get_q1_q2(next_ss)).max(dim=1, keepdim=True)[0].squeeze(1)
            q_labels = rewards + undones * self.gamma * next_qs

        q1, q2 = [qs.gather(1, actions.long()).squeeze(1) for qs in self.act.get_q1_q2(states)]
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        return obj_critic, q1

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        cwd: Current Working Directory. ElegantRL save training files in CWD.
        if_save: True: save files. False: load files.
        """
        assert self.save_attr_names.issuperset({'act', 'act_target', 'act_optimizer'})

        for attr_name in self.save_attr_names:
            file_path = f"{cwd}/{attr_name}.pth"
            print("Agent file location: ", file_path)
            if if_save:
                print(f"Saving agent network: {attr_name}")
                th.save(getattr(self, attr_name), file_path)
            elif os.path.isfile(file_path):
                print(f"Loading agent network: {attr_name}")
                setattr(self, attr_name, th.load(file_path, map_location=self.device))

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        # print("**initialize states**")
        # print("  states size:", horizon_len,"x", self.num_envs)

        states = th.zeros((horizon_len, self.num_envs, self.state_dim), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.num_envs, 1), dtype=th.int32).to(self.device)  # different
        rewards = th.zeros((horizon_len, self.num_envs), dtype=th.float32).to(self.device)
        dones = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)

        state = self.last_state  # last_state.shape = (num_envs, state_dim) for a vectorized env.

              
        # get_action = self.act.get_action # TODO check
        get_action = self.act_target.get_action
        
        print("**stepping: horizon_len = ", horizon_len)
        
        for t in range(horizon_len):
            # print("  time ", t)
            action = th.randint(self.action_dim, size=(self.num_envs, 1)) if if_random \
                else get_action(state).detach()  # different
            # print("  action ", action)
            states[t] = state

            state, reward, done, _ = env.step(action)  # next_state
            actions[t] = action
            rewards[t] = reward
            dones[t] = done
        
        print("**finish stepping")
        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(th.float32)
        return states, actions, rewards, undones

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        with th.no_grad():
            states, actions, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=rewards, undones=undones).reshape((-1,))
            )

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            
            # the online and target networks are the same object, soft update here is not effective
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with th.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            next_qs = self.cri_target(next_ss).max(dim=1, keepdim=True)[0].squeeze(1)  # next q_values
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states).gather(1, actions.long()).squeeze(1)
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, q_values

    @staticmethod
    def soft_update(target_net: th.nn.Module, current_net: th.nn.Module, tau: float):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def optimizer_update(self, optimizer: th.optim, objective: Tensor):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = th.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    def get_cumulative_rewards(self, rewards: Tensor, undones: Tensor) -> Tensor:
        returns = th.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        
        # tmp fix: should next value come from cri_target?
        next_value = self.act_target(last_state).argmax(dim=1).detach()  # actor is Q Network in DQN style
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns

    def update_avg_std_for_normalization(self, states: Tensor, returns: Tensor):
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = self.cri.state_std * (1 - tau) + state_std * tau + 1e-4
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.act.state_std

        returns_avg = returns.mean(dim=0)
        returns_std = returns.std(dim=0)
        self.cri.value_avg[:] = self.cri.value_avg * (1 - tau) + returns_avg * tau
        self.cri.value_std[:] = self.cri.value_std * (1 - tau) + returns_std * tau + 1e-4


class AgentD3QN(AgentDoubleDQN):  # Dueling Double Deep Q Network. (D3QN)
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNetTwinDuel)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        
        # Define directly
        self.act = self.act_target = QNetTwinDuel(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = self.act
        

## AgentTwinD3QN was defined exatly the same as DDQN, commenting out for now
# class AgentTwinD3QN(AgentDoubleDQN):
#     def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
#         self.act_class = getattr(self, "act_class", QNetTwin)
#         self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
#         super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)




'''PPO'''
        
class AgentPPO(AgentDoubleDQN):
    """PPO algorithm + GAE
    “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.
    “Generalized Advantage Estimation”. John Schulman. et al..
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        
        self.if_discrete: bool = args.if_discrete                        
        self.if_off_policy = False

            
        '''network'''
        self.act = ActorPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.cri = CriticPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        
        '''optimizer'''
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)
        
        # The target networks are assigned as references to the main networks 
        # It might be better to create independent network and soft update
        self.act_target = self.act
        self.cri_target = self.cri
        
        # self.criterion = th.nn.MSELoss(reduction="none")
        self.criterion = th.nn.SmoothL1Loss(reduction="mean")

        '''PPO clip and entropy'''
        # self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.ratio_clip = 0.1

        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = th.tensor(self.lambda_entropy, dtype=th.float32, device=self.device)
        self.if_use_v_trace = getattr(args, 'if_use_v_trace', True)

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[TEN, TEN, TEN, TEN, TEN, TEN]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        return: `(states, actions, logprobs, rewards, undones, unmasks)` for on-policy
            `states.shape == (horizon_len, num_envs, state_dim)`
            `actions.shape == (horizon_len, num_envs, action_dim)`
            `logprobs.shape == (horizon_len, num_envs, action_dim)`
            `rewards.shape == (horizon_len, num_envs)`
            `undones.shape == (horizon_len, num_envs)`
            `unmasks.shape == (horizon_len, num_envs)`
        """
        states = th.zeros((horizon_len, self.num_envs, self.state_dim), dtype=th.float32).to(self.device)
        
        actions = th.zeros((horizon_len, self.num_envs, self.action_dim), dtype=th.float32).to(self.device) \
            if not self.if_discrete else th.zeros((horizon_len, self.num_envs,), dtype=th.int32).to(self.device)
              
        logprobs = th.zeros((horizon_len, self.num_envs), dtype=th.float32).to(self.device)
        rewards = th.zeros((horizon_len, self.num_envs), dtype=th.float32).to(self.device)
        terminals = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)
        truncates = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)

        state = self.last_state  # shape == (num_envs, state_dim) for a vectorized env.

        convert = self.act.convert_action_for_env
        for t in range(horizon_len):
 
            
#             if if_random:
#                 action = th.randint(self.action_dim, size=(self.num_envs))
#                 logprob = 1.0986 # log(3): entropy for uniform distribution with dim 3 
#             else:
#                 action, logprob = self.explore_action(state)

            action, logprob = self.explore_action(state)

            states[t] = state
            actions[t] = action 
            logprobs[t] = logprob

            state, reward, terminal, _ = env.step(convert(action))  # next_state

            rewards[t] = reward
            terminals[t] = terminal
            truncates[t] = False #dummy value.

        self.last_state = state
        rewards *= self.reward_scale
        undones = th.logical_not(terminals)
        unmasks = th.logical_not(truncates)
        
        return states, actions, rewards, undones, unmasks, logprobs

    def explore_action(self, state: TEN) -> Tuple[TEN, TEN]:
        actions, logprobs = self.act.get_action(state)
        return actions, logprobs

    def update_net(self, buffer) -> Tuple[float, ...]:
        buffer_size = buffer[0].shape[0]

        '''get advantages reward_sums'''
        with th.no_grad():
            states, actions, rewards, undones, unmasks, logprobs = buffer     
            self.update_avg_std_for_normalization( states=states.reshape((-1, self.state_dim)))
            
            bs = 2 ** 10  # set a smaller 'batch_size' when out of GPU memory.
            values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)]
            values = th.cat(values, dim=0).squeeze(-1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(states, rewards, undones, unmasks, values)  # shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)
            assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, states.shape[1])
        
        buffer = states, actions, reward_sums, unmasks, logprobs, advantages

        '''update network'''
        obj_critics = []
        obj_actors = []

        th.set_grad_enabled(True)
        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        
        print("buffer_size: ", buffer_size)
        print("repeat_times: ", self.repeat_times)
        print("batch_size: ", self.batch_size)

        assert update_times >= 1
        for update_t in range(update_times):
            obj_critic, obj_actor = self.update_objectives(buffer, update_t)
            obj_critics.append(obj_critic)
            obj_actors.append(obj_actor)
        th.set_grad_enabled(False)

        obj_critic_avg = np.array(obj_critics).mean() if len(obj_critics) else 0.0
        obj_actor_avg = np.array(obj_actors).mean() if len(obj_actors) else 0.0
        a_std_log = getattr(self.act, 'a_std_log', th.zeros(1)).mean()
        # return obj_critic_avg, obj_actor_avg, a_std_log.item()
        return obj_critic_avg, obj_actor_avg

    def update_objectives(self, buffer, update_t: int) -> Tuple[float, float]:
        
        states, actions, reward_sums, unmasks, logprobs, advantages = buffer
        
        # print("states dimension in buffer: ", states.shape) # buffer size x env dim x state dim
        
        sample_len = states.shape[0]
        num_seqs = states.shape[1]
        
        #random sample from buffer, with sample size = batch_size
        ids = th.randint(sample_len * num_seqs, size=(self.batch_size,), requires_grad=False, device=self.device)
        # buffer item index, length = batch_size
        ids0 = th.fmod(ids, sample_len)  # ids % sample_len
        # environment index, length = batch_size
        ids1 = th.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        state = states[ids0, ids1]
        action = actions[ids0, ids1]
        unmask = unmasks[ids0, ids1]
        logprob = logprobs[ids0, ids1]
        advantage = advantages[ids0, ids1]
        reward_sum = reward_sums[ids0, ids1]

        # critic network predicts the reward_sum (Q value) of state
        value = self.cri(state).squeeze(1)  
        
        #compute the loss using expected reward and actual reward
        obj_critic = (self.criterion(value, reward_sum) * unmask).mean()
        
        # Update critic network: backpropogation using computed loss
        # Critic's objective: minimize the difference between the predicted value and the actual reward.
        self.optimizer_update(self.cri_optimizer, obj_critic)
        
        # soft update for critic target
        # self.soft_update(self.cri_target, self.cri, self.soft_update_tau)


        # policy gradient clip
        # new_logprob: log prob of same action under new policy (new actor from last interation)
        # obj_entropy: entropy of action distribution
        new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
        
        # action distribution change
        ratio = (new_logprob - logprob.detach()).exp()
        surrogate1 = advantage * ratio
        # clipping to avoid large update
        surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
        # select the most conservative surrogate
        obj_surrogate = th.min(surrogate1, surrogate2)
        # entropy regularization: add entropy term to encourage exploration
        obj_actor = ((obj_surrogate + obj_entropy) * unmask).mean() * self.lambda_entropy
        
        # Update actor network
        # Actor's objective: maximize expected rewards.
        # -obj_actor: minimize it means maximize obj_actor
        self.optimizer_update(self.act_optimizer, -obj_actor)
        
        # soft update for actor target 
        # self.soft_update(self.act_target, self.act, self.soft_update_tau)

        return obj_critic.item(), obj_actor.item()

    
    def get_advantages(self, states: TEN, rewards: TEN, undones: TEN, unmasks: TEN, values: TEN) -> TEN:
        advantages = th.empty_like(values)  # advantage value

        # update undones rewards when truncated
        truncated = th.logical_not(unmasks)
        if th.any(truncated):
            rewards[truncated] += self.cri(states[truncated]).squeeze(1).detach()
            undones[truncated] = False

        # apply discount factor    
        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_state = self.last_state.clone()
        next_value = self.cri(next_state).detach().squeeze(-1)

        advantage = th.zeros_like(next_value)  # last advantage value by GAE (Generalized Advantage Estimate)
        if self.if_use_v_trace:  # get advantage value in reverse time series (V-trace)
            for t in range(horizon_len - 1, -1, -1):
                next_value = rewards[t] + masks[t] * next_value
                advantages[t] = advantage = next_value - values[t] + masks[t] * self.lambda_gae_adv * advantage
                next_value = values[t]
        else:  # get advantage value using the estimated value of critic network
            for t in range(horizon_len - 1, -1, -1):
                # advantage measure how much more beneficial that action was compared to the average.
                advantages[t] = rewards[t] - values[t] + masks[t] * advantage
                advantage = values[t] + self.lambda_gae_adv * advantages[t]
        return advantages


    def get_cumulative_rewards(self, rewards: TEN, undones: TEN) -> TEN:
        cum_rewards = th.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_action = self.act_target(last_state)
        next_value = self.cri_target(last_state, next_action).detach()
        for t in range(horizon_len - 1, -1, -1):
            cum_rewards[t] = next_value = rewards[t] + masks[t] * next_value
        return cum_rewards

 
    def update_avg_std_for_normalization(self, states: TEN):

        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = (self.act.state_std * (1 - tau) + state_std * tau).clamp_min(1e-4)
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.act.state_std

        self.act_target.state_avg[:] = self.act.state_avg
        self.act_target.state_std[:] = self.act.state_std
        self.cri_target.state_avg[:] = self.cri.state_avg
        self.cri_target.state_std[:] = self.cri.state_std
        
        
class AgentDiscretePPO(AgentPPO):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        AgentPPO.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)
        self.if_off_policy = False

        self.act = ActorDiscretePPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.cri = CriticPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        # self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.ratio_clip = 0.1
        
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = th.tensor(self.lambda_entropy, dtype=th.float32, device=self.device)

        
'''A2C'''
        
class AgentA2C(AgentPPO):
    """A2C algorithm.
    “Asynchronous Methods for Deep Reinforcement Learning”. 2016.
    """
    
    # use A2C objectives
    def update_objectives(self, buffer: Tuple[TEN, ...], update_t: int) -> Tuple[float, float]:    
    
        states, actions, reward_sums, unmasks, logprobs, advantages = buffer
 
        # buffer size x env dim x state dim
        sample_len = states.shape[0]
        num_seqs = states.shape[1]
        
        #random sample from buffer, with sample size = batch_size
        ids = th.randint(sample_len * num_seqs, size=(self.batch_size,), requires_grad=False, device=self.device)
        # buffer item index, length = batch_size
        ids0 = th.fmod(ids, sample_len)  # ids % sample_len
        # environment index, length = batch_size
        ids1 = th.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        state = states[ids0, ids1]
        action = actions[ids0, ids1]
        unmask = unmasks[ids0, ids1]
        logprob = logprobs[ids0, ids1]
        advantage = advantages[ids0, ids1]
        reward_sum = reward_sums[ids0, ids1]

        # critic network predicts the reward_sum (Q value) of state
        value = self.cri(state).squeeze(1)  
        
        #compute the loss using expected reward and actual reward
        obj_critic = (self.criterion(value, reward_sum) * unmask).mean()
        
        # Update critic network: backpropogation using computed loss
        # Critic's objective: minimize the difference between the predicted value and the actual reward.
        self.optimizer_update(self.cri_optimizer, obj_critic)

        new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)        
        
        # Add entropy to encourage exploration, fine-tune entropy coefficient
        # obj_actor = (advantage * new_logprob).mean()  # obj_actor without policy gradient clip
        obj_actor = (advantage * new_logprob + obj_entropy * 0.1).mean()


        self.optimizer_update(self.act_optimizer, -obj_actor)
        return obj_critic.item(), obj_actor.item()



class AgentDiscreteA2C(AgentDiscretePPO):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        AgentDiscretePPO.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)
        self.if_off_policy = False

        self.act = ActorDiscretePPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.cri = CriticPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        # self.if_use_v_trace = getattr(args, 'if_use_v_trace', True)
        
    # use A2C objectives
    def update_objectives(self, buffer: Tuple[TEN, ...], update_t: int) -> Tuple[float, float]:    
    
        states, actions, reward_sums, unmasks, logprobs, advantages = buffer
 
        # buffer size x env dim x state dim
        sample_len = states.shape[0]
        num_seqs = states.shape[1]
        
        #random sample from buffer, with sample size = batch_size
        ids = th.randint(sample_len * num_seqs, size=(self.batch_size,), requires_grad=False, device=self.device)
        # buffer item index, length = batch_size
        ids0 = th.fmod(ids, sample_len)  # ids % sample_len
        # environment index, length = batch_size
        ids1 = th.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        state = states[ids0, ids1]
        action = actions[ids0, ids1]
        unmask = unmasks[ids0, ids1]
        logprob = logprobs[ids0, ids1]
        advantage = advantages[ids0, ids1]
        reward_sum = reward_sums[ids0, ids1]

        # critic network predicts the reward_sum (Q value) of state
        value = self.cri(state).squeeze(1)  
        
        #compute the loss using expected reward and actual reward
        obj_critic = (self.criterion(value, reward_sum) * unmask).mean()
        
        
        # Update critic network: backpropogation using computed loss
        # Critic's objective: minimize the difference between the predicted value and the actual reward.
        self.optimizer_update(self.cri_optimizer, obj_critic)

        new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)        
        
        # Add entropy to encourage exploration, fine-tune entropy coefficient
        # obj_actor = (advantage * new_logprob).mean()  # obj_actor without policy gradient clip
        obj_actor = (advantage * new_logprob + obj_entropy * 0.1).mean()


        self.optimizer_update(self.act_optimizer, -obj_actor)
        return obj_critic.item(), obj_actor.item()

    
'''SAC'''

class AgentDiscreteSAC(AgentDoubleDQN):
    
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        
        # self.if_discrete: bool = args.if_discrete
        self.if_discrete: bool = True

        # self.num_ensembles = getattr(args, 'num_ensembles', 4)  # the number of critic networks
        self.num_ensembles = 4

        # self.act = ActorSAC(net_dims, state_dim, action_dim).to(self.device)
        self.act = ActorDiscreteSAC(net_dims, state_dim, action_dim).to(self.device)
        self.cri = CriticEnsemble(net_dims, state_dim, action_dim, num_ensembles=self.num_ensembles).to(self.device)
        
        # soft update actor 
        # self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)
        
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.alpha_log = th.tensor((-1,), dtype=th.float32, requires_grad=True, device=self.device)  # trainable var
        self.alpha_optim = th.optim.Adam((self.alpha_log,), lr=args.learning_rate)
        self.target_entropy = np.log(action_dim)
        
    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[TEN, TEN, TEN, TEN, TEN]:
        states = th.zeros((horizon_len, self.num_envs, self.state_dim), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.num_envs, self.action_dim), dtype=th.float32).to(self.device) \
            if not self.if_discrete else th.zeros((horizon_len, self.num_envs), dtype=th.int32).to(self.device)
        rewards = th.zeros((horizon_len, self.num_envs), dtype=th.float32).to(self.device)
        terminals = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)
        
        # print(actions)

        state = self.last_state  
        
        for t in range(horizon_len):
            action = self.explore_action(state)
            # print(action)

            states[t] = state  
            actions[t] = action

            #convert action dimension to be compatible with envrionment
            state, reward, terminal, _ = env.step(action.unsqueeze(1))

            rewards[t] = reward
            terminals[t] = terminal

        self.last_state = state
        actions = actions.view((horizon_len, self.num_envs, 1)) if self.if_discrete else actions
        rewards *= self.reward_scale
        undones = th.logical_not(terminals)
        return states, actions, rewards, undones
    
    
    def explore_action(self, state: TEN) -> TEN:
        return self.act.get_action(state)

    
    def update_net(self, buffer: Union[ReplayBuffer, tuple]) -> Tuple[float, ...]:
        objs_critic = []
        objs_actor = []

        th.set_grad_enabled(True)
        
        # update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        update_times = int(buffer.add_size * self.repeat_times)        
                
        print("buffer_size: ", buffer.cur_size)
        print("repeat_times: ", self.repeat_times)
        print("batch_size: ", self.batch_size)
        

        for update_t in range(update_times):
            obj_critic, obj_actor = self.update_objectives(buffer=buffer, update_t=update_t)
            objs_critic.append(obj_critic)
            objs_actor.append(obj_actor) if isinstance(obj_actor, float) else None
        th.set_grad_enabled(False)

        obj_avg_critic = np.nanmean(objs_critic) if len(objs_critic) else 0.0
        obj_avg_actor = np.nanmean(objs_actor) if len(objs_actor) else 0.0
        return obj_avg_critic, obj_avg_actor
    

    def update_objectives(self, buffer: ReplayBuffer, update_t: int) -> Tuple[float, float]:
        assert isinstance(update_t, int)
        with th.no_grad():
            state, action, reward, undone, next_state = buffer.sample(self.batch_size)
            is_weight, is_index = None, None

            next_action, next_logprob = self.act.get_action_logprob(next_state)  # stochastic policy
            next_q = th.min(self.cri_target.get_q_values(next_state, next_action), dim=1)[0]
            alpha = self.alpha_log.exp()
            q_label = reward + undone * self.gamma * (next_q - next_logprob * alpha)

        '''objective of critic (loss function of critic)'''
        # Using 4 decoders - q_values has dimension of batch_size * 4
        q_values = self.cri.get_q_values(state, action)
        # print("q_values", q_values)
        q_labels = q_label.view((-1, 1)).repeat(1, q_values.shape[1])
        # print("q_labels", q_labels)
        
        #Fix dimension error
        #td_error = self.criterion(q_values, q_labels).mean(dim=1) 
        td_error = self.criterion(q_values, q_labels)

#         if self.if_use_per:
#             obj_critic = (td_error * is_weight).mean()
#             buffer.td_error_update_for_per(is_index.detach(), td_error.detach())
#         else:
#             obj_critic = td_error.mean()
        
        obj_critic = td_error.mean()
  
        #if self.lambda_fit_cum_r:
        #    cum_reward_mean = buffer.cum_rewards[buffer.ids0, buffer.ids1].detach_().mean().repeat(q_values.shape[1])
        #    obj_critic += self.criterion(cum_reward_mean, q_values.mean(dim=0)).mean() * self.lambda_fit_cum_r
        self.optimizer_update(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        '''objective of alpha (temperature parameter automatic adjustment)'''
        action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
        obj_alpha = (self.alpha_log * (self.target_entropy - logprob).detach()).mean()
        self.optimizer_update(self.alpha_optim, obj_alpha)

        '''objective of actor'''
        alpha = self.alpha_log.exp().detach()
        with th.no_grad():
            self.alpha_log[:] = self.alpha_log.clamp(-16, 2)

        q_value_pg = self.cri_target(state, action_pg).mean()
        obj_actor = (q_value_pg - logprob * alpha).mean()
        self.optimizer_update(self.act_optimizer, -obj_actor)
        
        # Soft update actor
        # self.soft_update(self.act_target, self.act, self.soft_update_tau)
        
        return obj_critic.item(), obj_actor.item()
    
   

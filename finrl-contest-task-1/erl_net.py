import torch as th
import torch.nn as nn
from typing import Tuple, List

TEN = th.Tensor

'''DQN'''

class QNetBase(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.explore_rate = 0.125
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(th.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(th.ones((1,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / self.state_std

    def value_re_norm(self, value: TEN) -> TEN:
        return value * self.value_std + self.value_avg


class QNetTwin(QNetBase):  # Double DQN
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2
        self.soft_max = nn.Softmax(dim=1)

        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        return q_val  # one group of Q values

    def get_q1_q2(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val1 = self.net_val1(s_enc)  # q value 1
        q_val1 = self.value_re_norm(q_val1)
        q_val2 = self.net_val2(s_enc)  # q value 2
        q_val2 = self.value_re_norm(q_val2)
        return q_val1, q_val2  # two groups of Q values

    def get_action(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        if self.explore_rate < th.rand(1):
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            # a_prob = self.soft_max(q_val)
            # action = th.multinomial(a_prob, num_samples=1)
            action = th.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class QNetTwinDuel(QNetBase):  # D3QN: Dueling Double DQN
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_adv1 = build_mlp(dims=[dims[-1], 1])  # advantage value 1
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
        self.net_adv2 = build_mlp(dims=[dims[-1], 1])  # advantage value 2
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2
        self.soft_max = nn.Softmax(dim=1)

        layer_init_with_orthogonal(self.net_adv1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_adv2[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        q_adv = self.net_adv1(s_enc)  # advantage value
        value = q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # one dueling Q value
        value = self.value_re_norm(value)
        return value

    def get_q1_q2(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state

        q_val1 = self.net_val1(s_enc)  # q value 1
        q_adv1 = self.net_adv1(s_enc)  # advantage value 1
        q_duel1 = q_val1 - q_val1.mean(dim=1, keepdim=True) + q_adv1
        q_duel1 = self.value_re_norm(q_duel1)

        q_val2 = self.net_val2(s_enc)  # q value 2
        q_adv2 = self.net_adv2(s_enc)  # advantage value 2
        q_duel2 = q_val2 - q_val2.mean(dim=1, keepdim=True) + q_adv2
        q_duel2 = self.value_re_norm(q_duel2)
        return q_duel1, q_duel2  # two dueling Q values

    def get_action(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        if self.explore_rate < th.rand(1):
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            # a_prob = self.soft_max(q_val)
            # action = th.multinomial(a_prob, num_samples=1)
            action = th.randint(self.action_dim, size=(state.shape[0], 1))
        return action

'''PPO'''

class ActorPPO(nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *net_dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.1)

        self.action_std_log = nn.Parameter(th.zeros((1, action_dim)), requires_grad=True)  # trainable parameter
        self.ActionDist = th.distributions.normal.Normal

        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / (self.state_std + 1e-4)

    def forward(self, state: TEN) -> TEN:
        state = self.state_norm(state)
        action = self.net(state)
        return self.convert_action_for_env(action)

    def get_action(self, state: TEN) -> Tuple[TEN, TEN]:  # for exploration
        state = self.state_norm(state)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: TEN, action: TEN) -> Tuple[TEN, TEN]:
        state = self.state_norm(state)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: TEN) -> TEN:
        # convert action to range between -1 and 1
        new_action = action.tanh()
        # add a singleton to make environment stepping method
        new_action = new_action.unsqueeze(1)
        return new_action


class ActorDiscretePPO(ActorPPO):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim)
        self.ActionDist = th.distributions.Categorical
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, state: TEN) -> TEN:
        state = self.state_norm(state)
        a_prob = self.net(state)  # action_prob without softmax
        # return a_prob.argmax(dim=1)  # get the indices of discrete action
        return a_prob  # return the full prob matrix to be consistent with evaluator.

    def get_action(self, state: TEN) -> (TEN, TEN):
        state = self.state_norm(state)
        a_prob = self.soft_max(self.net(state))
        a_dist = self.ActionDist(a_prob)
        action = a_dist.sample()
        logprob = a_dist.log_prob(action)
        return action, logprob

    def get_logprob_entropy(self, state: TEN, action: TEN) -> (TEN, TEN):
        state = self.state_norm(state)
        a_prob = self.soft_max(self.net(state))  # action.shape == (batch_size, 1), action.dtype = th.int
        dist = self.ActionDist(a_prob)        
        # print("actions: ", action)

        # fix dimension
        # logprob = dist.log_prob(action.squeeze(1))
        logprob = dist.log_prob(action)
        
        entropy = dist.entropy()
        # print("actions log prob: ", logprob)
        # print("action distribution entropy: ", entropy)
        return logprob, entropy

#     @staticmethod
#     def convert_action_for_env(action: TEN) -> TEN:
#         return action.long()
    
    @staticmethod
    def convert_action_for_env(action: TEN) -> TEN:
        # print("action", action)
        # convert action to range between -1 and 1
        new_action = action.long()
        # add a singleton to make environment stepping method
        new_action = new_action.unsqueeze(1)
        # print("new action", new_action)
        return new_action
    

class CriticPPO(th.nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__()
        assert isinstance(action_dim, int)
        self.net = build_mlp(dims=[state_dim, *net_dims, 1])
        layer_init_with_orthogonal(self.net[-1], std=0.5)

        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)

    def forward(self, state: TEN) -> TEN:
        state = self.state_norm(state)
        value = self.net(state)
        return value  # advantage value

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / (self.state_std + 1e-4)



'''SAC'''

class ActorBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = None  # build_mlp(net_dims=[state_dim, *net_dims, action_dim])

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.explore_noise_std = None  # standard deviation of exploration action noise
        self.ActionDist = th.distributions.normal.Normal

    def forward(self, state: TEN) -> TEN:
        action = self.net(state)
        return action.tanh()


class CriticBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(net_dims=[state_dim + action_dim, *net_dims, 1])

    def forward(self, state: TEN, action: TEN) -> TEN:
        values = self.get_q_values(state=state, action=action)
        value = values.mean(dim=-1, keepdim=True)
        return value  # Q value

    def get_q_values(self, state: TEN, action: TEN) -> TEN:
        values = self.net(th.cat((state, action), dim=1))
        return values  # Q values


class ActorSAC(ActorBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_s = build_mlp(dims=[state_dim, *net_dims], if_raw_out=False)  # network of encoded state
        self.net_a = build_mlp(dims=[net_dims[-1], action_dim * 2])  # the average and log_std of action
        layer_init_with_orthogonal(self.net_a[-1], std=0.1)
    
    def forward(self, state):
        s_enc = self.net_s(state)  # encoded state
        a_avg = self.net_a(s_enc)[:, :self.action_dim]
        return a_avg.tanh()  # action

    def get_action(self, state):
        s_enc = self.net_s(state)  # encoded state
        a_avg, a_std_log = self.net_a(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()
        dist = self.ActionDist(a_avg, a_std)
        return dist.rsample().tanh()  # action (re-parameterize)

    def get_action_logprob(self, state):
        s_enc = self.net_s(state)  # encoded state
        a_avg, a_std_log = self.net_a(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()

        dist = self.ActionDist(a_avg, a_std)
        action = dist.rsample()

        action_tanh = action.tanh()
        logprob = dist.log_prob(a_avg) #not action log prob?
        logprob -= (-action_tanh.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        return action_tanh, logprob.sum(1)

# Add discrete version fo SAC
class ActorDiscreteSAC(ActorSAC):
        
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim)
        self.ActionDist = th.distributions.Categorical
        self.soft_max = nn.Softmax(dim=-1)
        
    def get_action(self, state: TEN) -> (TEN, TEN):
        state = self.state_norm(state)
        a_prob = self.soft_max(self.net(state))
        a_dist = self.ActionDist(a_prob)
        action = a_dist.sample()
        logprob = a_dist.log_prob(action)
        return action, logprob


    def get_action(self, state):
        s_enc = self.net_s(state)  # encoded state
        a_avg, a_std_log = self.net_a(s_enc).chunk(2, dim=1)
        a_prob = self.soft_max(a_avg)
        dist = self.ActionDist(a_prob)
        return dist.sample().tanh()
    

    def get_action_logprob(self, state):
        s_enc = self.net_s(state)  # encoded state
        a_avg, a_std_log = self.net_a(s_enc).chunk(2, dim=1)
        a_prob = self.soft_max(a_avg)
        dist = self.ActionDist(a_prob)
        action = dist.sample()
        action_tanh = action.tanh()
        logprob = dist.log_prob(action)
        logprob -= (-action_tanh.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        return action_tanh, logprob


class CriticEnsemble(CriticBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, num_ensembles: int = 4):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        # self.encoder_sa = build_mlp(dims=[state_dim + action_dim, net_dims[0]])  # encoder of state and action
        #adjust for discrete action space
        self.encoder_sa = build_mlp(dims=[state_dim + 1, net_dims[0]])  # encoder of state and action

        self.decoder_qs = []
        for net_i in range(num_ensembles):
            decoder_q = build_mlp(dims=[*net_dims, 1])
            layer_init_with_orthogonal(decoder_q[-1], std=0.5)

            self.decoder_qs.append(decoder_q)
            setattr(self, f"decoder_q{net_i:02}", decoder_q)

    def get_q_values(self, state: TEN, action: TEN) -> TEN:
        # tensor_sa = self.encoder_sa(th.cat((state, action), dim=1))
        
        # adjust action dimension
        if action.dim() == 1:  # Check if action has a single dimension
            action = action.unsqueeze(-1)  # Make action [batch_size, 1] if discrete
        elif action.dim() == 3:  # Action may already have [batch_size, action_dim, 1]
            action = action.squeeze(-1)  # Ensure it is [batch_size, action_dim]
    
        tensor_sa = self.encoder_sa(th.cat((state, action), dim=1))
        values = th.concat([decoder_q(tensor_sa) for decoder_q in self.decoder_qs], dim=-1)    
        return values  # Q values
        
    
'''util'''
    
def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    
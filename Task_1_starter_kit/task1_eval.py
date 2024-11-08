import os
import torch
import numpy as np
from erl_config import Config, build_env
from trade_simulator import EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from collections import Counter
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown

from erl_agent import AgentPPO, AgentA2C, AgentDiscretePPO, AgentDiscreteA2C
from erl_agent import AgentDiscreteSAC

import pandas as pd

def to_python_number(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().item()
    else:
        return x


class EnsembleEvaluator:
    def __init__(self, save_path, agent_classes, args: Config):
        self.save_path = save_path
        self.agent_classes = agent_classes

        # args
        self.args = args
        self.agents = []
        self.thresh = 0.001
        self.num_envs = 1
        self.state_dim = 8 + 2
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        self.trade_env = build_env(args.env_class, args.env_args, gpu_id=args.gpu_id)

        self.current_btc = 0
        self.cash = [args.starting_cash]
        self.btc_assets = [0]
        # self.net_assets = [torch.tensor(args.starting_cash, device=self.device)]
        self.net_assets = [args.starting_cash]
        self.starting_cash = args.starting_cash
        
        self.random_seed = args.random_seed

    def load_agents(self):
        args = self.args
        for agent_class in self.agent_classes:
            agent = agent_class(
                args.net_dims,
                args.state_dim,
                args.action_dim,
                gpu_id=args.gpu_id,
                args=args,
            )
            agent_name = agent_class.__name__
            cwd = os.path.join(self.save_path, agent_name)
            agent.save_or_load_agent(cwd, if_save=False)  # Load agent
            self.agents.append(agent)

    def multi_trade(self, strategy = "majority_vote"):
        """Evaluation loop using ensemble of agents"""
        
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        agents = self.agents
        trade_env = self.trade_env
        
        state = trade_env.reset()

        last_state = state
        last_price = 0

        positions = []
        action_ints = []
        correct_pred = []
        current_btcs = [self.current_btc]

        for _ in range(trade_env.max_step):
            
            actions = []
            q_values_list = []
            intermediate_state = last_state
            

            # Collect actions from each agent
            for agent in agents:
                with torch.no_grad():
                    actor = agent.act

                tensor_state = torch.as_tensor(intermediate_state, dtype=torch.float32, device=agent.device)
                tensor_q_values = actor(tensor_state)
                tensor_action = tensor_q_values.argmax(dim=1)
                action = tensor_action.detach().cpu().unsqueeze(1)
                actions.append(action)
                q_values = tensor_q_values.cpu().detach()
                q_values_list.append(q_values)

            if strategy == "Majority_Voting":
                action = self._ensemble_action_majority_voting(actions=actions)   
            elif strategy == "Confidence_Based": # use Q value as confidence score
                action = self._ensemble_action_confidence_based(q_values_list=q_values_list) 
            elif strategy == "Boltzmann_Addition":
                action = self._ensemble_action_boltzmann_addition(q_values_list=q_values_list)
            elif strategy == "Boltzmann_Multiplication":
                action = self._ensemble_action_boltzmann_multiplication(q_values_list=q_values_list)
            else:
                raise NotImplementedError

            
            action_int = action.item() - 1

            state, reward, done, _ = trade_env.step(action=action)
            
            # Transform to the actual action used in stepping
            cur_pos = state[0][0].cpu().detach().numpy()
            pre_pos = last_state[0][0].cpu().detach().numpy()
            action_int = cur_pos - pre_pos            

            action_ints.append(action_int)
    
            # The positions below are from trade_env, which is a result of post-processed action after "stop loss"
            positions.append(trade_env.position)

            # Manually compute cumulative returns
            
            # Adjust index to be consistent with trading env
            # mid_price = trade_env.price_ary[trade_env.step_i, 2].to(self.device)
            mid_price = trade_env.price_ary[trade_env.step_i + trade_env.step_is, 2].to(self.device)
            new_cash = self.cash[-1]

            if action_int > 0 and self.cash[-1] > mid_price:  # Buy
                last_cash = self.cash[-1]
                new_cash = last_cash - mid_price
                self.current_btc += 1
            elif action_int < 0 and self.current_btc > 0:  # Sell
                last_cash = self.cash[-1]
                new_cash = last_cash + mid_price
                self.current_btc -= 1

            self.cash.append(new_cash)
            self.btc_assets.append((self.current_btc * mid_price).item())
            self.net_assets.append((to_python_number(self.btc_assets[-1]) + to_python_number(new_cash)))

            last_state = state

            # Log win rate
            if action_int == 1:
                correct_pred.append(1 if last_price < mid_price else -1 if last_price > mid_price else 0)
            elif action_int == -1:
                correct_pred.append(-1 if last_price < mid_price else 1 if last_price > mid_price else 0)
            else:
                correct_pred.append(0)
                
#             print("last price", last_price, "mid price", mid_price)
#             print("action", action_int)
#             print("correct_pred", correct_pred)

            last_price = mid_price
            current_btcs.append(self.current_btc)

        num_trades = correct_pred.count(1) + correct_pred.count(-1)
        num_wins = correct_pred.count(1)
        
        if num_trades > 0:
            win_rate = num_wins/num_trades
        else:
            win_rate = 0

        # Save results
        if len(agents)>1:
            positions = np.array([t.numpy() for t in positions]).flatten()
            np.save(f"{self.save_path}{strategy}_positions.npy", np.array(positions))
            np.save(f"{self.save_path}{strategy}_net_assets.npy", np.array(self.net_assets))
            np.save(f"{self.save_path}{strategy}_btc_positions.npy", np.array(self.btc_assets))
            np.save(f"{self.save_path}{strategy}_correct_predictions.npy", np.array(correct_pred))
        
        # Compute metrics
        pnl = sum(np.diff(self.net_assets))
        returns = np.diff(self.net_assets) / self.net_assets[:-1]
        final_sharpe_ratio = sharpe_ratio(returns)
        final_max_drawdown = max_drawdown(returns)
        final_roma = return_over_max_drawdown(returns)
        
        print(f"Initial Asset: {self.net_assets[0]}")
        print(f"Final Asset: {self.net_assets[-1]}")
        print(f"PnL: {pnl} \n")
           
        print(f"Mean Return {returns.mean()}")
        print(f"Volatility {returns.std()} \n")        
        
        print(f"Sharpe Ratio: {final_sharpe_ratio}")
        print(f"Max Drawdown: {final_max_drawdown}")
        print(f"Return over Max Drawdown: {final_roma}")
        print(f"Win Rate {win_rate}")

        
        res = {"Initial Asset": self.net_assets[0],
               "Final Asset": self.net_assets[-1],
               "PnL": pnl,
               "Mean Return": returns.mean(),
               "Volatility": returns.std(),
               "Sharpe Ratio": final_sharpe_ratio,
               "Max Drawdown": final_max_drawdown,
               "Return over Max Drawdown": final_roma,
               "Win Rate": win_rate,
              }
               
        return res

    def _ensemble_action_majority_voting(self, actions):
        """Returns the majority action among agents. Our code uses majority voting, you may change this to increase performance."""
        count = Counter([a.item() for a in actions])
        majority_action, _ = count.most_common(1)[0]
        return torch.tensor([[majority_action]], dtype=torch.int32)

    def _ensemble_action_confidence_based(self, q_values_list):
        """Returns the optimal action based on aggregated q values for all agents."""
        q_values_sum = sum(q_values_list)
        action = q_values_sum.argmax(dim=1).unsqueeze(1)
#         print("q values list", q_values_list) 
#         print("q values values sum",q_values_sum)
#         print("action", action)
        return action
    
    
    def _ensemble_action_boltzmann_addition(self, q_values_list):
        """Returns the optimal action based on the sum of boltzmann probabilities for all agents."""
        boltzmann_probabilities = [torch.softmax(q_values, dim=1) for q_values in q_values_list]
        boltzmann_addition = torch.stack(boltzmann_probabilities).sum(dim=0)
        action = boltzmann_addition.argmax(dim=1).unsqueeze(1)
#         print("q values list", q_values_list) 
#         print("boltzmann_probabilities", boltzmann_probabilities)
#         print("boltzmann_addition",boltzmann_addition)
#         print("action", action)
        return action

    def _ensemble_action_boltzmann_multiplication(self, q_values_list):
        """Returns the optimal action based on the product of boltzmann probabilities for all agents."""
        q_values_sum = torch.zeros((1, 3), dtype=torch.float32)
        boltzmann_probabilities = [torch.softmax(q_values, dim=1) for q_values in q_values_list]        
        boltzmann_multiplication = torch.stack(boltzmann_probabilities).prod(dim=0)  
        action = boltzmann_multiplication.argmax(dim=1).unsqueeze(1)
#         print("q values list", q_values_list) 
#         print("boltzmann_probabilities", boltzmann_probabilities)
#         print("boltzmann_multiplication",boltzmann_multiplication)
#         print("action", action)
        return action

def run_evaluation(save_path, agent_list, strategy):
    import sys

    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # Get GPU_ID from command line arguments
    
    num_sims = 1
    num_ignore_step = 60
    max_position = 1
    step_gap = 2
    slippage = 7e-7

    max_step = (4800 - num_ignore_step) // step_gap

    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
        
#         # use default dataset
#         "dataset_path": "path_to_evaluation_dataset",  # Replace with your evaluation dataset path
    }
    args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = np.abs(gpu_id) #random seed need to be non-negative
    args.net_dims = (128, 128, 128)
    args.starting_cash = 1e6

    ensemble_evaluator = EnsembleEvaluator(
        save_path,
        agent_list,
        args,
    )
    ensemble_evaluator.load_agents()
    res = ensemble_evaluator.multi_trade(strategy = strategy)
    return res

if __name__ == "__main__":
    save_path = "trained_agents/"
    
    agent_list = [AgentDoubleDQN, AgentD3QN, AgentDiscretePPO, AgentDiscreteA2C,AgentDiscreteSAC]
    #agent_list = [AgentDoubleDQN]

    metrics_data = []
    for agent in agent_list:
        print(f"\n------Individual Agent Performance: {agent.__name__}------\n")
        metrics = run_evaluation(f"{save_path}", [agent], "Majority_Voting") #single action
        
        metrics["Agent"] = agent.__name__
        metrics_data.append(metrics)
        
    for strategy in ["Majority_Voting", "Confidence_Based", "Boltzmann_Addition", "Boltzmann_Multiplication"]:
        print(f"\n------Ensemble RL with {strategy}------\n")
        metrics = run_evaluation(f"{save_path}", agent_list, strategy)
        
        metrics["Agent"] = "Ensemble: " + strategy
        metrics_data.append(metrics)
      
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.insert(0, 'Agent', metrics_df.pop('Agent'))
    metrics_df.to_excel("evaluation_results.xlsx")
    
    print(metrics_df)

# FinRL Task 1


## Results

Main results and scripts:

```
├── finrl-contest-task-1 
│ ├── trained_models # Trained component agent weights, and evaluation results.
│ ├── task1_ensemble.py # File for training the individual agents.
│ ├── task1_eval.py # File for implementing the ensemble method, and creating performance metrics.
│ ├── evaluation_results.xlsx # File to save evaluation results from task1_eval.py

```
Five agents are tested individually and jointly using ensemble learning methods.

(1) Agents:

The agents used are: DoubleDQN, D3QN, Discrete PPO, Discrete A2C, and Discrete SAC.
The agents are trained with 4096 (2^12) environments, and 1.28 million steps.
Trained agent weights and training plots are saved in 'trained_models'.

(2) Ensemble Methods:

Four ensemble methods are used:
        - majority voting
        - confidence based (q-value as confidence score)
        - Boltzmann Addition
        - Boltzmann Multiplication

Ensemble methods are saved in task1_eval.py.


## Script Updates


The code is based on starter kit, which includes updates as described below:

- `trade_simulator.py`: Contains a market replay simulator for single commodity.
  - Class `TradeSimulator`: A training-focused market replay simulator, complying with the older gym-style API requirements.
  - Class `EvalTradeSimulator`: A market replay simulator for evaluation.


- `erl_agent.py`: Contains the DQN class algorithm for reinforcement learning.
   Updates:
   (1) Add agents for PPO, Discrete PPO, A2C, Discrete A2C, and Discrete SAC are added.
   (2) Remove AgentTwinD3QN, as it is implemented the same way as DoubleDQN (should add new actor).


- `erl_net.py`: Neural network structures used in the reinforcement learning algorithm.
   Updates:
   Add actor and critic neural networks for PPO, Discrete PPO, SAC, and Discrete SAC are added.


- `erl_evaluator.py`: Evaluates the performance of the reinforcement learning agent.
   Updates:
   Minor fix for the data type of position_count and action_count to avoid CUDA segmentation error. 

- `task1_ensemble.py`: This file contains code that trains multiple models and then saves them to be tested during evaluation.
   Updates:
   (1) Minor fix for the data type of position_count and action_count to avoid CUDA segmentation error. 
   (2) Add training for all agents: DoubleDQN, D3QN, Discrete PPO, Discrete A2C, and Discrete SAC
   (3) Update break_step from 0.32 million to 1.28 million, to add training episodes.
   (4) For SAC, adjust number of environments from 2^12 to 2^10, due to large memory used by the agent. (avoid GPU out of memory)
   (5) Add a few fine-tuning params for PPO and A2C.


- `task1_eval.py`: This file contains code that loads your ensemble and simulates trading over a validation dataset. You may create this validation dataset by holding out a part of the training data.
    Updates:
    (1) Add individual evaluations for DoubleDQN, D3QN, Discrete PPO, Discrete A2C, and Discrete SAC.
    (2) Add ensemble learning methods based on: 
        - majority voting
        - confidence based (q-value as confidence score)
        - Boltzmann Addition
        - Boltzmann Multiplication
    (3) Add additional performance metrics and summary.
    (4) Minor fixes:
        - Use the index of mid_price to be consistent with trading environment and calcuate the return correctly.
        - Use the same action as the trading environment to be consistent with the positions.
        - Freeze the random seed using the random_seed set in args. (to allow reproducing the results and comparing across models)




# FinRL Task 1


## Results

### Main results and scripts:

```
├── finrl-contest-task-1 
│ ├── trained_models # Trained component agent weights, and evaluation results.
│ ├── task1_ensemble.py # File for training the individual agents.
│ ├── task1_eval.py # File for implementing the ensemble method, and creating performance metrics.
│ ├── evaluation_results.xlsx # File to save evaluation results from task1_eval.py

```
### The overall performance
The table below can be replicated using task1_eval.ipynb or task1_eval.py:

<img width="1040" alt="image" src="https://github.com/user-attachments/assets/12069a58-112f-4c43-9165-402288fb0022">

NOTE: The performance metric also depend on the market condition (price trend and volatility), so the test sample used can heavily impact the results. The same test set should be used to compare between different models.

This performance metrics used are based on the test sample from provided price and factor data, with the initial time stamp index 'randomly' sampled using random_seed = gpu_id (as initialized in the starter kit). If gpu_id is negative, it is coverted to absolute value.

The random seed is NOT frozen during training. It is only fixed in the evaluation to allow replication and comparison between agents and ensemble methods.

The results shown are run on cpu, with random_seed = 1. If running on gpu, the results will be slightly different (random_seed = 0), but generally inline.

### Brief description on the methods:

(1) Agents:

Five agents are tested individually and jointly using ensemble learning methods: DoubleDQN, D3QN, Discrete PPO, Discrete A2C, and Discrete SAC. 
The agents are trained with 4096 (2^12) environments, and 1.28 million steps.

Trained agent weights and training plots are saved in 'trained_models'.

(2) Ensemble Methods:

Four ensemble methods are used: Majority Voting, Confidence Based (q-value as confidence score), Boltzmann Addition, and Boltzmann Multiplication

Ensemble methods are saved in task1_eval.py.


## Script Updates


The code is based on starter kit, which includes updates as described below:

- `trade_simulator.py`: Contains a market replay simulator for single commodity.
  - Class `TradeSimulator`: A training-focused market replay simulator, complying with the older gym-style API requirements.
  - Class `EvalTradeSimulator`: A market replay simulator for evaluation.


- `erl_agent.py`: Contains the DQN class algorithm for reinforcement learning.

  Updates:

  (1) Add agents for PPO, Discrete PPO, A2C, Discrete A2C, and Discrete SAC.
  (2) Remove AgentTwinD3QN, as it is implemented the same way as DoubleDQN (should add new actor when time allows).


- `erl_net.py`: Neural network structures used in the reinforcement learning algorithm.

  Updates:

  Add actor and critic neural networks for PPO, Discrete PPO, SAC, and Discrete SAC.


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


- `task1_eval.py`: This file contains code that loads the ensemble and simulates trading over a validation dataset. 

  Updates:

  (1) Add individual evaluations for DoubleDQN, D3QN, Discrete PPO, Discrete A2C, and Discrete SAC.

  (2) Add ensemble learning methods based on: 

        - majority voting

        - confidence based (q-value as confidence score)

        - Boltzmann Addition

        - Boltzmann Multiplication

  (3) Add additional performance metrics and export summary.

  (4) Minor fixes:

        - Use the index of mid_price to be consistent with trading environment and calcuate the return correctly. If this is not fixed, the return will be random.

        - Use the same action as the trading environment to be consistent with the positions. If this is not fixed, the position and action will be out of sync.

        - Freeze the random seed using the random_seed set in args. (to allow reproducing the results and comparing across models)




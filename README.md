# Reinforcement Learning Value Function Approximation

## Getting Started

Install the prerequisite libraries: `pip install numpy scipy matplotlib`

You can simply clone/download the repository and run programs as-is. Below is a brief explanation of what is included:

### KEY FILES

VFA_Net is a sequential neural net written using primarily Numpy. This is used in place of PyTorch or Tensorflow due to the need for 
rapid flexibility in customization.

TD0_Template is a template for a generic implementation of VFA_Net with TD(0) learning. It contains functioning code as well as pseudocode. 

### Blackjack

Tabular_QL is equipped with typical methods used in decaying epsilon-greey Q-learning. Consistently converges to optimal.

Tabular_MC is similar, except it "learns" through the Monte Carlo algorithm.

Blackjack_Complete is a customizable blackjack environment for a learning agent to interact with.

Blackjack_VFA is functional; however, can be further optimized. Implements scaled TD0 learning.

### LQR

TODO WRITE HERE

## Test Results:

### Blackjack

Using **1-hot encoding**, the neural net converges (VFA_TD0). Also loosely converged when sampling instead of computing expected value for the TD update.

Architecture: Hidden Layer - 512 neurons, ReLU; Output Layer - 1 neuron, Linear 

Parameters: ALPHA = 1000, GAMMA = 1, NUM_TRIALS = 100000 and 500000, function initialized to zero



Using **normalized 3-vector**, the neural net converges (VFA_TDv6). Also converged when sampling instead of computing expected value (500000 trials)

Architecture: Hidden Layer - 512 neurons, ReLU; Output Layer - 1 neuron, Linear 

Parameters: ALPHA = 100, GAMMA = 1, NUM_TRIALS = 100000 function initialized to zero 

### LQR

TODO WRITE HERE

## Future Work:

1. Rewrite VFA_Net using PyTorch/Tensorflow. (and update all dependencies accordingly)
2. Change Tabular_MC policy from greedy to \epsilon-greedy

## Authors

* **Craig Chen** - *Initial work* - [craigxchen](https://github.com/craigxchen)
* **Andrea Agazzi** - *ADD HERE* - [agazzian](ADD HERE)



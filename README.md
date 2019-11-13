# BlackjackRL


## Getting Started

QL_Agent is equipped with typical methods used in decaying epsilon-greey Q-learning.

Blackjack_Complete is a customizable blackjack environment for a learning agent to interact with

VFA_Net is a sequential neural net written using primarily Numpy. This is used in place of PyTorch or Tensorflow due to the need for 
rapid flexibility in customization.

All Value Function Approximation (VFA) files are not complete

## Notes

Up-to-date versions:

VFA_TDv6 is functional, still does not converge to optimal. 

QL_Agent DOES converge to optimal, run QL_Agent.py to see what the optimal value function looks like.

## Test Results:

VFA_TDv6:

using 512 neurons in hidden layer, ALPHA = 1000, GAMMA = 1, NUM_TRIALS = 100000 and 500000, function initialized to zero
    
converges for:

1-hot encoding and relu, leakyRelu -- see results folder

normalized vector encoding still does not converge

## Authors

* **Craig Chen** - *Initial work* - [craigxchen](https://github.com/craigxchen)



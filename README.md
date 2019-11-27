# BlackjackRL


## Getting Started

QL_Agent is equipped with typical methods used in decaying epsilon-greey Q-learning. Consistently converges to optimal.

Blackjack_Complete is a customizable blackjack environment for a learning agent to interact with

VFA_Net is a sequential neural net written using primarily Numpy. This is used in place of PyTorch or Tensorflow due to the need for 
rapid flexibility in customization.

VFA_TD is a work in progress.

## Notes

Up-to-date versions:

VFA_TDv6 is functional

Blackjack_Complete_TEST is the same as the original version; however, it includes a few extra functions to simplify life in the TD learning model.

## Test Results:

Using **1-hot encoding**, the neural net converges (VFA_TDv6). Also loosely converged when sampling instead of computing expected value for the TD update.

Parameters: 512 neurons, ALPHA = 1000, GAMMA = 1, NUM_TRIALS = 100000 and 500000, function initialized to zero

Non-linearities: ReLU -- see results folder


Using **normalized 3-vector**, the neural net converges (VFA_TDv6). Also converged when sampling instead of computing expected value (500000 trials)

Parameters: 512neurons, ALPHA = 100, GAMMA = 1, NUM_TRIALS = 100000 function initialized to zero 

Non-linearities: ReLU -- see results folder

## Authors

* **Craig Chen** - *Initial work* - [craigxchen](https://github.com/craigxchen)



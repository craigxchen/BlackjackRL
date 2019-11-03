# BlackjackRL


## Getting Started

QL_Agent is equipped with typical methods used in decaying epsilon-greey Q-learning.

Blackjack_Complete is a customizable blackjack environment for a learning agent to interact with

VFA_Net is a sequential neural net written using primarily Numpy. This is used in place of PyTorch or Tensorflow due to the need for 
rapid flexibility in customization.

All Value Function Approximation (VFA) files are not complete

## Notes
Outdated versions:
VFA_TDv2 is meant to run on a CUDA-enabled GPU, although it is still somehow slower than training on the CPU
VFA_TDv2_CPU is the most updated version, it trains on CPU
VFA_TDv3 implements batch training on a CUDA-enabled GPU
VFA_TDv4 is an implementation of a custom-written sequential neural net. (VFA_Net.py)
* primary issue is that it converges to a constant function

Up-to-date versions:
VFA_TDv6 is functional, still does not converge to optimal
VFA_TDv7 is a copy of v6, but gives risk-free area to edit state-mappings/input formats

## Authors

* **Craig Chen** - *Initial work* - [craigxchen](https://github.com/craigxchen)



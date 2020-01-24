# Reinforcement Learning Value Function Approximation

## Immediate To-Do's

1. Fix PPO algorithm by separating the Actor and Critic
2. Implement PPO for simple LQR

## Getting Started

Install the prerequisite libraries: `pip install numpy scipy matplotlib pickle` and `pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`

You can simply clone/download the repository and run programs as-is. Below is a brief explanation of what is included:

### KEY FILES

VFA_Net is a sequential neural net written using primarily Numpy. This is used in place of PyTorch or Tensorflow due to the need for 
rapid flexibility in customization.

PG_Net backpropogates with respect to the objective function defined in the Policy Gradient Theorem.

TD0_Template is a template for a generic implementation of VFA_Net with TD(0) learning. It contains functioning code as well as pseudocode. 

### Blackjack

Tabular_QL is equipped with typical methods used in decaying epsilon-greey Q-learning. Consistently converges to optimal.

Tabular_MC is similar, except it "learns" through the Monte Carlo algorithm.

Blackjack_Complete is a customizable blackjack environment for a learning agent to interact with.

Blackjack_VFA is functional; however, can be further optimized. Implements scaled TD0 learning.

### LQR

lqr_control implements the dyanmics of an LQR controller and some standard procedures to find the optimal control for this problem (Riccati eq)

unstable_laplacian uses lqr_control to simulate an example found in a review paper by Ben Recht 

double_integrator is another example from the same paper

1dim_test uses lqr_control to solve a 1-dimensional example of LQR system

1dim_VFA also solves the LQR control problem simulated with "LQR_control" in 1 dimension but uses a neural network with quadratic activations to learn the cost function

### Test Results:

runtests performs backpropagation in a supervised learning setting to approximate a linear function in a certain interval with a wide, shallow single layer neural network. 

crude_anim_line plots the evolution of the approximating function and compares it with the original linear function.

### Blackjack

Using **1-hot encoding**, the neural net converges (VFA_TD0). Also loosely converged when sampling instead of computing expected value for the TD update.

Architecture: Hidden Layer - 512 neurons, ReLU; Output Layer - 1 neuron, Linear 

Parameters: ALPHA = 1000, GAMMA = 1, NUM_TRIALS = 100000 and 500000, function initialized to zero



Using **normalized 3-vector**, the neural net converges (VFA_TDv6). Also converged when sampling instead of computing expected value (500000 trials)

Architecture: Hidden Layer - 512 neurons, ReLU; Output Layer - 1 neuron, Linear


Parameters: ALPHA = 100, GAMMA = 1, NUM_TRIALS = 100000 function initialized to zero 

### LQR

The tests in the 1D and the cases of Ben Recht's paper converge immediately if one assumes linearity of the optimal policy. In the case of approximation of the value function with a shallow NN with quadratic nonlinearity we also observe convergence to the global optimum when the policy is assumed to be linear.

### Supervised

We observe convergence of a wide, shallow NN with relu nonlinearities to a linear function, although the convergence is not perfect in the interval of interest. 

## Future Work:

1. Combine VFA_Net and PG_Net into one file (and update all dependencies accordingly)
2. Combine the approximation NN of the value function for the LQR with the NN approximation of the policy from the supervised learning setting.
3. Play around to find regimes of convergence/divergence? Change nonlinearities to see if the result changes. 

## Authors

* **Craig Chen** - *Initial work* - [craigxchen](https://github.com/craigxchen)
* **Andrea Agazzi** - *Improvements* - [agazzian](https://github.com/agazzian)



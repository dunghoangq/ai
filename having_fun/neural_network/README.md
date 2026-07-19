# Neural Networks for Dummies

To understand how Neural Network works, the underlying mathematics, I start with Building from Scratch.

Each lesson introduces genuinely new ideas rather than revisiting the same implementation from a different angle.

## Roadmap

0. ✅ Linear Algebra
1. ✅ Neuron*
2. ✅ Single Layer Neural Network
3. ✅ Backpropagation
4. ✅ Gradient Descent
5. ✅ Multi-layer Network
6. ✅ Activation Functions
7. ✅ Loss Functions
8. Optimizers
9. Mini-batch Training
10. Regularization
11. Build a mini PyTorch


## Coding milestones

1. `Neuron`: Linear model, weights, bias
2. `LinearLayer`: Matrix multiplication and batches
3. `Activation`: Non-linearity (e.g. ReLU, Sigmoid)
4. `MSELoss`: Measuring prediction error
5. Manual gradients: Derivatives for a single neuron
6. Backpropagation: Chain rule through layers
7. `SGD` optimizer: Updating parameters
8. `Sequential`: Chaining layers into networks
9. Mini-batches: Efficient training
10. Training loop: End-to-end learning
11. Validation & metrics: Evaluating performance
12. Mini autograd engine: How PyTorch's `backward()` works

# Logs
You designed me a Neural Networks from scratch learning roadmap that you would teach me as teaching an AI researcher at OpenAI, Anthropic.

Part I — Why Optimisation is Hard

1. The Limits of Gradient Descent
Why isn't ordinary gradient descent enough?
New ideas:
- optimisation as trajectory
- conditioning
- curvature
- narrow valleys
- plateaus
- saddle points
- ravines
- exploding directions
- local approximation using Taylor expansion

Implementation:
- visualise optimisation paths
- implement vanilla GD
- compare on different surfaces

2. Loss Landscape Geometry

Topics:
- Hessian
- eigenvalues
- curvature
- condition number
- convex vs non-convex
- anisotropic curvature
- why deep learning landscapes are weird

Implementation:
- compute Hessian numerically
- eigen decomposition
- visualise curvature

3. Stochastic Optimisation
Topics:

- full batch
- mini-batch
- stochastic gradients
- gradient noise
- variance
- why noise sometimes helps

Implementation:
simulate noisy gradients.

Part II — Classical Optimisers

4. Momentum
Topics:

- physical intuition
- heavy-ball method
- exponential moving average
- frequency-domain intuition
- oscillation suppression

Implementation:
Momentum class.

5. Nesterov Acceleration
- derive mathematically
- estimate future position
- compare convergence proofs intuitively
- visual comparison

Part III — Adaptive Learning Rates

6. Why One Learning Rate Cannot Fit All Parameters
Topics:

- sparse gradients
- embeddings
- coordinate-wise optimisation
- diagonal preconditioning

Implementation:
different learning rates for each parameter.

7. AdaGrad
Topics:

- cumulative statistics
- online learning
- regret minimisation
- strengths
- fatal weakness

8. RMSProp
- exponential averages
- adaptive memory
- stationary vs changing distributions

Part IV — Adam Family

9. Adam from First Principles
Derive:

- Momentum
- RMSProp
- bias correction

10. Why Adam Needs Bias Correction
- EMA initialisation bias
- expectation
- derivation
- proof
- why correction matters early training

11. Adam's Failure Modes

Topics:

- non-convergence examples
- bad generalisation
- sharp minima
- flat minima
- learning rate warm-up
- instability

Implementation:
reproduce failure cases.

12. AdamW
- L2 regularisation
- weight decay
- why they are equivalent for SGD
- why Adam breaks equivalence
- decoupled optimisation

Part V — Beyond Adam

13. Learning Rate Scheduling
- step decay
- exponential decay
- cosine annealing
- warm restarts
- warm-up
- one-cycle policy

14. Modern Optimisers
Topics:

- Lion
- AdaFactor
- LAMB
- Shampoo
- Sophia
- Muon

When each is used.
Memory/computation trade-offs.
Implementation of Lion.

15. Second-Order Optimisation
Topics:

- Newton's method
- Hessian
- Hessian inverse
- Gauss-Newton
- quasi-Newton
- L-BFGS
- natural gradient

Implementation:
Newton optimiser.

Part VI — Research Perspective

16. Optimisation in Deep Learning Today
- scaling laws
- optimisation at trillion-parameter scale
- distributed optimisation
- gradient compression
- mixed precision
- optimizer state memory
- ZeRO
- fused optimizers
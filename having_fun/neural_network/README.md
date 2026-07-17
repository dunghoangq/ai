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
7. Loss Functions
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

1. Why We Need a Loss Function

Understand optimisation from first principles.
- Learning as optimisation
- Objective functions
- Why weights alone cannot tell us how to improve
- Loss as distance between prediction and reality
- Geometry of optimisation
- Continuous objective landscapes

2. Regression Losses

Understand losses for predicting numbers.

- Mean Squared Error
- Mean Absolute Error
- Huber Loss
- Log-Cosh Loss

Compare

- sensitivity to outliers
- smoothness
- derivatives
- optimisation behaviour

Implement each from scratch.

3. Probabilistic View of Regression Losses

This is where things become much deeper.

We'll derive

- Why MSE assumes Gaussian noise
- Why MAE  assumes Laplace noise
- Maximum likelihood estimation
- Negative log likelihood
- Connection between probability and optimisation

This lesson explains something most courses never mention.

4. Binary Classification Loss

- Why MSE performs poorly for classification
- Sigmoid output
- Binary Cross Entropy
- Log Loss
- Likelihood interpretation
- Decision boundaries
- Derive BCE mathematically.

Implement
binary_cross_entropy()

5. Multi-class Classification

- Softmax
- Categorical Cross Entropy
- One-hot labels
- Probability simplex
- Negative log likelihood
- Derive everything from scratch.

Implement
- softmax()
- cross_entropy()

6. Why Cross Entropy Works So Well

- Confidence
- Calibration
- Large gradients for confident mistakes
- Small gradients for correct predictions
- Information gain
- Entropy
- KL divergence intuition

You'll finally understand why Cross Entropy trains so much faster than MSE.

7. Information Theory Behind Deep Learning

This lesson connects Shannon's work to neural networks.
- Entropy
- Cross Entropy
- KL Divergence
- Mutual Information
- Surprisal
- Compression viewpoint
- Coding interpretation
- Why predicting probabilities equals data compression

8. Numerical Stability
Engineering lesson.
Implement stable versions of
- Softmax
- Cross Entropy
- LogSumExp

Avoid
- overflow
- underflow
- NaN
- Inf

Learn tricks used in PyTorch.

9. Automatic Differentiation Through Losses

We already know backprop.

Now we derive gradients specifically for
- MSE
- MAE
- Huber
- BCE
- Cross Entropy

Then compare gradient behaviour.

10. Loss Landscape

- Local minima
- Saddle points
- Flat minima
- Sharp minima
- High-dimensional geometry

11. Modern Loss Functions

Survey
- Focal Loss
- Label Smoothing
- Dice Loss
- IoU Loss
- Triplet Loss
- Contrastive Loss
- Cosine Loss
- ArcFace
- InfoNCE

When each is used.

12. Research Frontier

- Loss shaping
- Self-supervised losses
- Energy-based models
- Diffusion objectives
- Reinforcement learning objectives
- Preference optimisation (DPO)
- Reward modelling
- Learnable losses
- Meta-losses
- Neural loss search

This lesson connects directly to current research papers.

13. Final project

We'll build a loss library identical in spirit to PyTorch.

- class MSELoss:
- class BCELoss:
- class CrossEntropyLoss:
- class HuberLoss:
- class MAELoss:
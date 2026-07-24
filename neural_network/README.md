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
8. ✅ Optimizers
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

Part I — Why Mini-batches Exist

1. The Three Worlds: Batch, Stochastic and Mini-batch Gradient Descent

Why don't we simply compute the exact gradient?

New ideas:

- dataset as an expectation
- empirical risk
- computational cost
- exact gradient vs sampled gradient
- optimisation trajectory changes
- deterministic vs stochastic optimisation

Implementation:

- Implement all three methods
- Visualise optimisation trajectories
- Compare computation time

2. Gradient Noise: Why Imperfect Gradients Can Be Better

Why can a worse gradient produce a better model?

New ideas:

- unbiased gradient estimator
- variance
- gradient covariance
- optimisation noise
- escaping saddle points
- implicit regularisation
- diffusion intuition

Implementation:

- Inject controlled gradient noise
- Observe optimisation paths
- Compare convergence

3. Statistical View of Mini-batching

Why is averaging gradients statistically sensible?

New ideas:

- Monte Carlo estimation
- Law of Large Numbers
- Central Limit Theorem
- estimator variance
- confidence of gradient estimates

Implementation:

- Estimate gradients using different batch sizes
- Plot variance vs batch size

Part II — Batch Size Changes Optimisation

4. The Critical Batch Size

Why doesn't increasing batch size always help?

New ideas:

- diminishing returns
- gradient noise scale
- compute efficiency
- critical batch size
- scaling laws for batches

Implementation:

- Measure training speed
- Measure optimisation progress
- Find critical batch size experimentally

5. Small Batches vs Large Batches

Which produces better models?

New ideas:

- sharp minima
- flat minima
- Hessian spectrum
- generalisation gap
- optimisation vs generalisation

Implementation:

Train identical networks using

- batch = 16
- batch = 64
- batch = 512

Compare

- training loss
- validation loss
- Hessian eigenvalues (numerically)

6. Why Gradient Noise Regularises Learning

Why does that noise improve generalisation?

New ideas:

- stochastic differential equation view
- temperature analogy
- escaping sharp minima
- entropy
- implicit bias of SGD

Implementation:

- Visualise trajectories with different noise levels.

Part III — Practical Training

7. Shuffling the Dataset

Why shuffle every epoch?

New ideas:

- correlation
- IID assumptions
- sampling bias
- curriculum effects
- deterministic ordering

Implementation:

Train

- shuffled
- sorted
- clustered

Observe convergence.

8. Data Loaders and Input Pipelines

Why do GPUs spend so much time waiting?

New ideas:

- asynchronous loading
- workers
- prefetching
- pinned memory
- bottlenecks
- throughput

Implementation:

Build a mini DataLoader from scratch.

9. Gradient Accumulation

How can we simulate a batch of 4096 on a GPU that fits only 128?

New ideas:

- accumulated gradients
- delayed optimiser step
- effective batch size
- memory vs computation

Implementation:

Implement gradient accumulation manually.

Part IV — Modern Large-Scale Training

10. Large Batch Training

How did ImageNet training shrink from weeks to hours?

New ideas:

- linear learning-rate scaling
- square-root scaling
- warm-up necessity
- instability
- communication efficiency

Implementation:

Compare

- scaled LR
- unscaled LR

for increasing batches.

11. Batch Normalisation and Batch Size

Why does BatchNorm depend on batch size?

New ideas:

- batch statistics
- noisy estimates
- micro-batches
- Ghost BatchNorm
- alternatives (LayerNorm, GroupNorm)

Implementation:

Observe BatchNorm failure for tiny batches.

12. Distributed Mini-batching

How do 2048 GPUs create one giant mini-batch?

New ideas:

- local batch
- global batch
- gradient averaging
- AllReduce
- communication cost

Implementation:

Simulate multiple workers using NumPy.

Part V — Research Perspective

13. SGD as Bayesian Inference

Is SGD secretly performing approximate Bayesian inference?

New ideas:

- Langevin dynamics
- posterior sampling
- Bayesian interpretation
- optimisation vs inference
- uncertainty

No implementation.

Mostly conceptual.

14. The Double Descent of Batch Size

Why do some batch sizes unexpectedly perform worse?

New ideas:

interpolation regime
double descent
modern scaling behaviour
training dynamics

Implementation:

Reproduce generalisation curves.

15. Training Recipes Used by OpenAI, Anthropic and Google

How modern LLMs choose

- batch size
- sequence length
- gradient accumulation
- learning-rate schedule
- warm-up
- weight decay
- optimiser
- mixed precision

We'll study training configurations from influential papers such as GPT-3, PaLM, Chinchilla, Llama, and similar large-scale models to understand the engineering principles behind their choices rather than memorising hyperparameters.

# Prompt

You designed me a Neural Networks from scratch learning roadmap that you would teach me as teaching an AI researcher at OpenAI, Anthropic. Modules:
0. ✅ Linear Algebra
1. ✅ Neuron*
2. ✅ Single Layer Neural Network
3. ✅ Backpropagation
4. ✅ Gradient Descent
5. ✅ Multi-layer Network
6. ✅ Activation Functions
7. ✅ Loss Functions
8. ✅ Optimizers 
9. Mini-batch Training
10. Regularization
11. Build a mini PyTorch
We're at module 9 - Mini-batch Training, which have these lessons below. Each lesson introduces genuinely new ideas rather than revisiting the same implementation from a different angle. 
Part I — Why Mini-batches Exist

1. The Three Worlds: Batch, Stochastic and Mini-batch Gradient Descent

Why don't we simply compute the exact gradient?

New ideas:

- dataset as an expectation
- empirical risk
- computational cost
- exact gradient vs sampled gradient
- optimisation trajectory changes
- deterministic vs stochastic optimisation

Implementation:

- Implement all three methods
- Visualise optimisation trajectories
- Compare computation time

2. Gradient Noise: Why Imperfect Gradients Can Be Better

Why can a worse gradient produce a better model?

New ideas:

- unbiased gradient estimator
- variance
- gradient covariance
- optimisation noise
- escaping saddle points
- implicit regularisation
- diffusion intuition

Implementation:

- Inject controlled gradient noise
- Observe optimisation paths
- Compare convergence

3. Statistical View of Mini-batching

Why is averaging gradients statistically sensible?

New ideas:

- Monte Carlo estimation
- Law of Large Numbers
- Central Limit Theorem
- estimator variance
- confidence of gradient estimates

Implementation:

- Estimate gradients using different batch sizes
- Plot variance vs batch size

Part II — Batch Size Changes Optimisation

4. The Critical Batch Size

Why doesn't increasing batch size always help?

New ideas:

- diminishing returns
- gradient noise scale
- compute efficiency
- critical batch size
- scaling laws for batches

Implementation:

- Measure training speed
- Measure optimisation progress
- Find critical batch size experimentally

5. Small Batches vs Large Batches

Which produces better models?

New ideas:

- sharp minima
- flat minima
- Hessian spectrum
- generalisation gap
- optimisation vs generalisation

Implementation:

Train identical networks using

- batch = 16
- batch = 64
- batch = 512

Compare

- training loss
- validation loss
- Hessian eigenvalues (numerically)

6. Why Gradient Noise Regularises Learning

Why does that noise improve generalisation?

New ideas:

- stochastic differential equation view
- temperature analogy
- escaping sharp minima
- entropy
- implicit bias of SGD

Implementation:

- Visualise trajectories with different noise levels.

Part III — Practical Training

7. Shuffling the Dataset

Why shuffle every epoch?

New ideas:

- correlation
- IID assumptions
- sampling bias
- curriculum effects
- deterministic ordering

Implementation:

Train

- shuffled
- sorted
- clustered

Observe convergence.

8. Data Loaders and Input Pipelines

Why do GPUs spend so much time waiting?

New ideas:

- asynchronous loading
- workers
- prefetching
- pinned memory
- bottlenecks
- throughput

Implementation:

Build a mini DataLoader from scratch.

9. Gradient Accumulation

How can we simulate a batch of 4096 on a GPU that fits only 128?

New ideas:

- accumulated gradients
- delayed optimiser step
- effective batch size
- memory vs computation

Implementation:

Implement gradient accumulation manually.

Part IV — Modern Large-Scale Training

10. Large Batch Training

How did ImageNet training shrink from weeks to hours?

New ideas:

- linear learning-rate scaling
- square-root scaling
- warm-up necessity
- instability
- communication efficiency

Implementation:

Compare

- scaled LR
- unscaled LR

for increasing batches.

11. Batch Normalisation and Batch Size

Why does BatchNorm depend on batch size?

New ideas:

- batch statistics
- noisy estimates
- micro-batches
- Ghost BatchNorm
- alternatives (LayerNorm, GroupNorm)

Implementation:

Observe BatchNorm failure for tiny batches.

12. Distributed Mini-batching

How do 2048 GPUs create one giant mini-batch?

New ideas:

- local batch
- global batch
- gradient averaging
- AllReduce
- communication cost

Implementation:

Simulate multiple workers using NumPy.

Part V — Research Perspective

13. SGD as Bayesian Inference

Is SGD secretly performing approximate Bayesian inference?

New ideas:

- Langevin dynamics
- posterior sampling
- Bayesian interpretation
- optimisation vs inference
- uncertainty

No implementation.

Mostly conceptual.

14. The Double Descent of Batch Size

Why do some batch sizes unexpectedly perform worse?

New ideas:

interpolation regime
double descent
modern scaling behaviour
training dynamics

Implementation:

Reproduce generalisation curves.

15. Training Recipes Used by OpenAI, Anthropic and Google

How modern LLMs choose

- batch size
- sequence length
- gradient accumulation
- learning-rate schedule
- warm-up
- weight decay
- optimiser
- mixed precision

We'll study training configurations from influential papers such as GPT-3, PaLM, Chinchilla, Llama, and similar large-scale models to understand the engineering principles behind their choices rather than memorising hyperparameters. 
We finished lessons 1 to 9. Please move to lesson 10.
# XGBoost

## First-principles

Start with a weak model:

$$
F_0(x)
$$

Then add trees $f_1, f_2,..., f_T$:

$$
F_T(x) = F_0(x) + \sum_{1}{T}f_i(x)
$$

## XGBoost updates functions, not params

Recall Gradient descent:

$$
\theta \leftarrow \theta - \eta \nabla_{\theta}L
$$

XGBoost does the same, but:

$$
F \leftarrow F + \eta f
$$

Where $f$ (aka new added tree) plays the role of direction that reduces the loss.

## How XGBoost fits

Each tree fits on negative grad of loss from prev tree.

$$
L(y, F) \rightarrow \nabla_{F}L = \frac{\partial L}{\partial F} \rightarrow -\nabla_{F}L
$$

But XGBoost also use second-order derivative.

Taylor:

$$
f(x + \Delta x) \approx f(x) + \frac{\Delta x}{1!}f'(x) + \frac{\Delta x}{2!}f''(x) + ... \\
= f(x) + \nabla f^T \Delta x + \frac{1}{2}\Delta x^T \nabla^2 f \Delta x + ...
$$

with Hessian:

$$
H = \nabla^2 f
$$

So if we define:

$$
g_i = \frac{\partial L(y_i, \hat{y}_i)}{\partial \hat{y}_i} \\
h_i = \frac{\partial^2 L(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}
$$

then:

$$
\theta \leftarrow \theta - \eta g \\
\theta \leftarrow \theta - \frac{g}{h}
$$

## What XGBoost minimises

Suppose at iteration $t$.

$$
F_t = F_{t-1} + f_t
$$

XGBoost minimises:

$$
\mathcal{L}^{(t)} = \sum_{i=1}^n L(y_i, F_{t-1}(x_i) + f_{t}(x_i)) + \Omega(f)
$$

with regularisation (like in neural networks to get more generalisation):

$$
\Omega(f) = \gamma K + \frac{\lambda}{2}\sum_{j=1}^K w_j^2
$$

- $K$: # leaves
- $\gamma$: leaves' penalty
- $\lambda / 2$: L2 regularisation
- $w_j$: leaf score

Now apply Taylor Expansion to the loss:

$$
L(y_i, F_{t-1} + f_t) \approx L(y_i, F_{t-1}) + g_if_t(x_i) + \frac{1}{2}h_if_t^2(x_i)
$$

Ignore the first term because it doesn't have $f_t$ in it -> Optimise this:

$$
\boxed{
    \~{\mathcal{L}}^{(t)} = \sum_i\left[ g_if_t(x_i) + \frac{1}{2}h_if_t^2(x_i)  \right] + \Omega(f_t)
}
$$

But all samples in the same leaf have the same score:

$$
f_t(x_i) = w_{q(x_i)}
$$

And in leaf j-th, all samples in leaf is $I_j$ , total first-order derivative and second-order derivative are:

$$
G_j = \sum_{i \in I_j}g_i, \qquad H_j = \sum_{i \in I_j}h_i
$$

then the loss function (with L2 regularisation) at leaf j becomes:

$$
\mathcal{L}_j = G_jw_j + \frac{1}{2}(h_j + \lambda)w_j^2
$$

Differentiate wrt $w_j$ and set it to zero:

$$
\frac{\partial \mathcal{L}_j}{\partial w_j} = G_j + (H_j + \lambda)w_j = 0 \\
\Rightarrow \boxed{
    w_j^* = -\frac{G_j}{H_j + \lambda}
}
$$

This is called **leaf formula**.

## What does leaf formula mean?

- $G_j$: direction
- $H_j + \lambda$: curvature, magnitude, how aggressively model responds.
- High $w_j$ means increase predictions for observations in that leaf.

## How XGBoost splits

Imagine a split turns a node into left child and right child with:

$$
G = G_L + G_R \\
H = H_L + H_R
$$

then the gain (= quality after split - quality before split - complexity cost) tells us how much useful the split is:

$$
\text{Gain} = \frac{1}{2}\left[
    \frac{G_L^2}{H_L + \lambda}
    + \frac{G_R^2}{H_R + \lambda}
    - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}
\right] - \gamma
$$

> Gain > 0 means the split is **useful**.

Then compute leaf scores $w_L, w_R$, plug them into the update function.

## Fraud classification problem

All the loss functions mentioned above are for contiuous values.

But fraud is binary classification -> **binary cross entropy**.

$$
L(y, p) = -y\log(p) - (1 - y)\log(1 - p)
$$

with $p = \sigma(F)$ (sigmoid), $F$ is raw score.

Now differentiate the loss wrt $F$ with chain rule:

$$
\frac{\partial L}{\partial F} = \frac{\partial L}{\partial p} \frac{\partial p}{\partial F}
$$

where:

$$
\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p}
$$

and

$$
\frac{\partial p}{\partial F} = p(1-p)
$$

Hence:

$$
\boxed{
    g_i = p_i - y_i
}
$$

And Hessian:

$$
\boxed{
    h_i = p_i(1 - p_i)
}
$$

## Initialisation for Fraud

Init $F_0$ based on the empirical fraud rate. Say if $p_{\text{fraud}} = 0.001$, then:

$$
F_0 = \log \frac{p}{1 - p} \approx -6.907
$$

## XGBoost train loop

```text
F = F0

for tree t = 1...T
    p = sigmoid(F)
    g = p - y
    h = p(1 - p)

    CHOOSE-BEST-SPLIT(g, h, lambda, gamma) <- compute GAIN
    CALCULATE-LEAF-VALUE(g, h)
    UPDATE-PREDICTION(w, learning_rate)
```

## Some model params

- learning_rate
- max_depth
- leaf regularisation: $\lambda$
- split penalty: $\gamma$
- min child weight: prevents tiny leaf
- row subsampling: train each tree with a proportion of rows
- column subsampling: use some features in each tree

## The golden question

Why XGBoost works so well on tabular data???

## Other considerations

### Loss weight

Add weight to loss,

$$
\alpha_iL_i
$$

for $\alpha_i = 20$ with fraud, and $\alpha_i = 1$ with legitimate.

And the derivaties and hessians:

$$
g_i = \alpha_i(p_i - y_i) \\
h_i = \alpha_i p_i(1 - p_i)
$$

### Learned threshold

$$
\hat{y} = \begin{cases}
    1, & p >= \tau \\
    0, & \text{otherwise}
\end{cases}
$$

## The problem of Exact split

A feature has k distinct values, there are total $d$ features -> Total # candidate thresholds per node (worst scenario):

$$
O(kn)
$$

Imagine $k = 10^6, d = 100$ -> Total $10^8$ splits.

That's why we need **Quantisation**.

## Turn continuous values into Bins

Split data by range. Each bin = 1 range, instead of trying thresholds to split each value.

Use quantiles to choose bin ranges, ensure each bin has equal items.

## But quantile also has a problem

To have quantile, we need to sort data which is $O(n \log n)$.

That's why we have **Quantile sketch**, instead of storing every data point $O(n)$ and sort them $O(n \log n)$, it stores representative points and their approximate rank.

## Central data structure: Gradient Histogram

It's array of gradients, hessians:

- `grad_hist[i]` = sum gradients in bin i-th
- `hess_hist[i]` = sum hessians in bin i-th

It takes $O(n)$ to build histogram, but only $O(B)$ to evaluate every candiadate threshold.

e.g.,

| bin | G | H |
|-----|--:|--:|
| 0 | -3 | 2 |
| 1 | 2 | 1 |
| 2 | 7 | 2 |

split after bin 0,

$$
G_L = -3, \quad H_L = 2, \quad G_R = 9, \quad H_R = 3
$$

Then if $\lambda = 1, \gamma = 0$, the gain is:

$$
\text{Gain} = \frac{1}{2} \left[ 
    \frac{(-3)^2}{2 + 1} + \frac{9^2}{3 + 1} - \frac{6^2}{5 + 1}
\right] = 8.625
$$

Maintain **prefix sums** traverse bins left2right:

$$
G_L^{(k)} = \sum_{b=0}^k G_b, \qquad H_L^{(k)} = \sum_{b=0}^k H_b
$$

Then:

$$
G_R = G - G_L \\
H_R = H - H_L
$$

or element-wise version of code:

```python
hist_right = hist_parent - hist_left
```

## Full implementation

```text
INPUT
X, y
│
├──────────────────────────────────────┐
│                                      │
▼                                      │
Fit QuantileBinner                     │
│                                      │
▼                                      │
X_binned                               │
                                       │
Calculate fraud rate                   │
                                       ▼
                            base_score = log-odds
                                       │
                                       ▼
                             initialise F₀(x)
                                       │
                                       ▼
                          ┌── boosting loop ──┐
                          │                   │
                          │  p = sigmoid(F)   │
                          │                   │
                          │  g = p - y        │
                          │                   │
                          │  h = p(1-p)       │
                          │                   │
                          │  train tree       │
                          │                   │
                          │  f = tree(X)      │
                          │                   │
                          │  F ← F + ηf       │
                          │                   │
                          └───────────────────┘
                                       │
                                       ▼
                                   trees[]
```
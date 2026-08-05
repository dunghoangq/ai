# Autoencoders with fraud detection problem

## The problem

Input:

$$
x = \begin{bmatrix}
    \text{amount} \\
    \text{hour} \\
    \text{country} \\
    \text{merchant category} \\
    \text{time since previous transaction} \\
    \text{customer spending deviation} \\
    \vdots
\end{bmatrix} \in \mathbb{R}^d
$$

Output: risk score

$$
s(x) \in \mathbb{R}
$$

## Approaches

### Supervised approach

Learn directly from labelled examples:

$$
f_{\theta}(x) \approx P(y = 1|x)
$$

### Anomaly-detection approach

learn normal transactions:

$$
x \sim p_{\text{normal}}(x)
$$

See if a transaction is unlike a normal one.

## What Autoencoder is

- Try to reproduce inputs.
- Two functions:

$$
z = f_{\theta}(x) \\
\hat{x} = g_{\phi}(z)
$$

where:
- $f_{\theta}(x)$: encoder
- $z$: latent representation
- $g_{\phi}(z)$: decoder
- $\hat{x}$: reconstruction of $x$

The model is trained to minimise loss:

- individual loss

$$
\mathcal{L}(x, \hat{x}) = \frac{1}{d}\sum_{i=1}^d (x_i - \hat{x}_i)^2
$$

- Batch size $N$

$$
J(\theta, \phi) = \frac{1}{N}\sum_{i=1}^N\mathcal{L}(x^{(i)}, g_{\phi}(f_{\theta}(x^{(i)})))
$$

## How autoencoders reconstruct inputs

- Not simply copy, it reduces # features then increases it to learn. (e.g., $30 \rightarrow 16 \rightarrow 4 \rightarrow 16 \rightarrow 30$, the model **compresses 30 features into only 4 latent values**).

## Autoencoder geometry

Suppose a transaction have 30 features, it lives in $\mathbb{R}^30$ vector space.

An abnormal transaction may lie near the smaller structure called a **manifold**, aka lower dimensional surface inside a higher-dimensional space.

e.g., customer behaviour can be learned from 5 hidden factors:

- spending capacity
- lifestyle
- location
- purchasing habit
- time patterns

then **encoder** learns:

$$
f_{\theta} : \mathbb{R}^30 \rightarrow \mathbb{R}^5
$$

and **decoder** learns:

$$
g_{\phi} : \amthbb{R}^5 \rightarrow \mathbb{R}^30
$$

> Autoencoder learns to reconstruct normal pattern. Any data points violate that pattern is fraud.


## Why fraud detection is hard?

1. **Class imbalance**: 0.1% fraud
2. Incomplete fraud labels
3. Labels arrive late
4. **Concept drift**: Fraud behaviour changes - $P_t(x,y) \neq P_{t + \Delta t}(x, y)$.
5. Different fraud types:
   - stolen card
   - account takeover
   - merchant fraud
   - mule account
   - refund abuse
6. Even legit transactions are unusual (e.g., buy expensive laptop, travel abroad, midnight payment)

## How autoencoder solves fraud?

### Step 1: Construct transaction features

Vectors

- Raw transaction features
- Behavioural features

### Step 2: Train on normal transactions

Train on $D_{\text{normal}}$, minimising loss:

$$
J(\theta, \phi) = \frac{1}{N} \sum_{i=1}^N \|x^{(i)} - \hat{x}^{(i)} \|_2^2
$$

### Step 3: Compute reconstruction error

For a new transaction $x$:

$$
\hat{x} = g_{\phi}(f_{\theta}(x))
$$

then calculate:

$$
e(x) = \frac{1}{d}\sum_{i=1}^d(x_i - \hat{x}_i)^2
$$

Then risk score becomes abnormaly score:

$$
s(x) = e(x)
$$

### Step 4: Choose a threshold

$$
\hat{y} = \begin{cases}
    1, & e(x) > \tau \\
    0, & e(x) \le \tau
\end{cases}
$$

where:

- $\tau$: abnormaly threshold, can be chosen by percentile:

$$
\tau = Q_{0.99}(e(x_{\text{validation normal}}))
$$

In real system, the threshold can reflect business costs:

$$
\text{Expected cost } = C_{\text{Fn}}\dot FN + C_{\text{FP}}\dot FP
$$

where:

- $C_{\text{Fn}}$: cost of missing fraud;
- $C_{\text{FP}}$: cost of investigating or blocking legitimate activity

### Step 5: Evaluate against labelled data

Test whether on average:

$$
e(x_{\text{fraud}}) > e(x_{\text{normal}})
$$

Metrics:

- precision;
- recall;
- F1;
- precision–recall AUC;
- recall at a fixed alert rate;
- fraud value captured;
- false positives per thousand transactions.

#### Scaling features

For feature i-th:

$$
x_i' = \frac{x_i  - \mu_i}{\sigma_i}
$$

Or robust scaling:

$$
x_i' = \frac{x_i - \text{median}(x_i)}{\text{IQR}(x_i)}
$$

For txn amount, the transformation is

$$
a' = \log(1 + a)
$$

#### What about categorical features?

- **One-hot encoding**: $x_{\text{category}} \in \{ 0, 1 \}^K$
- **Embeddings**: high-cardinality vars like merchant ID: $\text{var}\rightarrow e_{\text{merchant}} \in \mathbb{R}^k$
- For device id, we don't embed. Instead we create `is_new_device` $\in \{0, 1\}$ and maintain `device_transaction_count`$_{30d}$

## How to prevent trivial memorisation

- Undercomplete autoencoder: $\dim(z) < \dim(x)$
- Regularisation (weight decay): $J = J_{\text{reconstruction}} + \lambda\sum_l\| W \|_2^2$
- Denoising autoencoder: reconstruct the original $\~{x} = x + \epsilon$
- Sparse autoencoder: L1 regularisation $J = J_{\text{reconstruction}} + \lambda\| z \|_1$
- Early stopping to avoid overfitting.

## Compare Autoencoder with other Techniques

- vs Isolation Forest
- vs One-class SVM
- vs PCA
- vs Clustering
- vs Fraud Rules


## Tips on loss functions

- Use weight for features

$$
e_w(x) = \sum_{i=1}^dw_i(x_i - \hat{x}_i)^2
$$

- Use composite loss

$$
\mathcal{L} = \lambda_c\mathcal{L}_{\text{continuous}} + \lambda_b\mathcal{L}_{\text{binary}} + \lambda_k\mathcal{L}_{\text{categorical}}
$$
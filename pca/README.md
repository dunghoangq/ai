# PCA with SVD

Build from scracth with Numpy only to understand it (deeply).

# Roadmap





# PCA via SVD from Scratch

## 1. The problem PCA was invented to solve

Suppose each transaction is represented by \(d\) features:

$$
x =
\begin{bmatrix}
\text{amount} \\
\text{transaction frequency} \\
\text{time since previous transaction} \\
\text{merchant risk score} \\
\vdots
\end{bmatrix}
\in \mathbb{R}^{d}
$$

A dataset of \(n\) transactions is therefore a matrix

$$
X \in \mathbb{R}^{n \times d}
$$

where rows are transactions and columns are features.

Real datasets often have three related problems:

1. **Redundancy**: several features contain nearly the same information.
2. **Noise**: some directions contain little meaningful structure.
3. **High dimensionality**: more features increase storage, computation and modelling difficulty.

For example:

- transaction amount in dollars;
- transaction amount in cents;
- transaction amount normalised by account balance.

These are not three independent pieces of information. They may largely describe one underlying direction: **transaction magnitude**.

The central PCA question is:

> Can we replace the original \(d\) features with \(k < d\) new features while preserving as much information as possible?

PCA answers yes, where “information” is measured by **variance**.

---

# 2. What PCA is

**Principal Component Analysis**, or PCA, constructs new orthogonal axes called **principal components**.

The first principal component is the direction along which the data varies most.

The second principal component is the direction of greatest remaining variation, subject to being perpendicular to the first.

The process continues until we have at most

$$
\min(n-1,d)
$$

non-zero principal components.

PCA changes the coordinate system.

It does not initially change the data itself.

Instead of describing a transaction using the original features,

$$
x =
\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_d
\end{bmatrix},
$$

PCA describes it using coordinates along new directions:

$$
z =
\begin{bmatrix}
z_1 \\ z_2 \\ \vdots \\ z_k
\end{bmatrix}.
$$

Each \(z_j\) is a linear combination of the original features:

$$
z_j = x^\top v_j,
$$

where \(v_j\) is the \(j\)-th principal direction.

---

# 3. Why variance?

Suppose all observations lie almost perfectly along a line:

$$
x_2 \approx 2x_1.
$$

Although the data has two features, it is effectively one-dimensional.

Along the line, observations differ substantially. Perpendicular to the line, there is almost no variation.

So the high-variance direction contains the meaningful structure, while the low-variance direction contains little more than noise.

PCA therefore searches for a unit vector \(v\) that maximises the variance of the projected data.

---

# 4. Centre the data first

Let the dataset be

$$
X =
\begin{bmatrix}
x_1^\top \\
x_2^\top \\
\vdots \\
x_n^\top
\end{bmatrix}.
$$

The feature-wise mean is

$$
\mu = \frac{1}{n}\sum_{i=1}^{n}x_i.
$$

We centre every observation:

$$
X_c = X - \mu.
$$

More precisely,

$$
(X_c)_{ij} = X_{ij} - \mu_j.
$$

After centring,

$$
\frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)=0.
$$

This matters because PCA should describe variation **around the centre of the data**, not variation caused by the dataset being far from the origin.

Without centring, PCA may choose a direction pointing towards the mean rather than a direction describing the shape of the data.

---

# 5. Projecting data onto one direction

Choose a unit direction

$$
v \in \mathbb{R}^{d},
\qquad
\|v\|_2=1.
$$

For one centred observation \(x_i\), its projection onto \(v\) is

$$
z_i = x_i^\top v.
$$

For the complete dataset:

$$
z = X_c v.
$$

Here,

$$
X_c \in \mathbb{R}^{n\times d},
\qquad
v \in \mathbb{R}^{d},
\qquad
z \in \mathbb{R}^{n}.
$$

Because \(X_c\) is centred, \(z\) is also centred. Its sample variance is

$$
\operatorname{Var}(z)
=
\frac{1}{n-1}z^\top z.
$$

Substitute \(z=X_cv\):

$$
\operatorname{Var}(z)
=
\frac{1}{n-1}(X_cv)^\top(X_cv).
$$

Therefore,

$$
\operatorname{Var}(z)
=
\frac{1}{n-1}v^\top X_c^\top X_c v.
$$

Define the sample covariance matrix:

$$
C = \frac{1}{n-1}X_c^\top X_c.
$$

Then

$$
\operatorname{Var}(z)=v^\top C v.
$$

PCA solves

$$
\max_{\|v\|_2=1} v^\top C v.
$$

The unit-length constraint is essential. Without it, we could make the variance arbitrarily large merely by multiplying \(v\) by a large number.

---

# 6. Why eigenvectors appear

We maximise

$$
v^\top C v
$$

subject to

$$
v^\top v=1.
$$

Introduce a Lagrange multiplier \(\lambda\):

$$
\mathcal{L}(v,\lambda)
=
v^\top C v-\lambda(v^\top v-1).
$$

Differentiate with respect to \(v\):

$$
\nabla_v \mathcal{L}
=
2Cv-2\lambda v.
$$

Set the gradient to zero:

$$
Cv=\lambda v.
$$

This is the eigenvalue equation.

Therefore:

- principal directions are eigenvectors of the covariance matrix;
- the variance along each direction is its corresponding eigenvalue.

The first principal component is the eigenvector associated with the largest eigenvalue.

The second is associated with the second-largest eigenvalue, and so on.

---

# 7. Why use SVD instead?

The covariance approach computes

$$
C = \frac{1}{n-1}X_c^\top X_c.
$$

Then it eigendecomposes \(C\).

But forming \(X_c^\top X_c\) has disadvantages:

- it may be computationally expensive;
- it can worsen numerical precision;
- it squares the condition number of \(X_c\);
- it creates a \(d\times d\) matrix even when the original data matrix has a more convenient shape.

SVD gives the same principal directions directly from the centred data matrix.

---

# 8. Singular Value Decomposition

Every real matrix \(X_c\in\mathbb{R}^{n\times d}\) can be decomposed as

$$
X_c = U\Sigma V^\top.
$$

For the reduced, or thin, SVD:

$$
U \in \mathbb{R}^{n\times r},
\qquad
\Sigma \in \mathbb{R}^{r\times r},
\qquad
V^\top \in \mathbb{R}^{r\times d},
$$

where

$$
r = \operatorname{rank}(X_c)
\leq \min(n,d).
$$

The matrices have distinct meanings:

- columns of \(U\): directions in observation space;
- diagonal values of \(\Sigma\): singular values;
- columns of \(V\): directions in feature space.

PCA needs directions in **feature space**, so its principal directions are the columns of \(V\).

NumPy returns

```python
U, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)
```

where the rows of `Vt` are the principal directions.

---

# 9. Connecting SVD to PCA

Start from

$$
X_c = U\Sigma V^\top.
$$

The covariance matrix is

$$
C = \frac{1}{n-1}X_c^\top X_c.
$$

Substitute the SVD:

$$
C
=
\frac{1}{n-1}
(U\Sigma V^\top)^\top
(U\Sigma V^\top).
$$

Transpose:

$$
C
=
\frac{1}{n-1}
V\Sigma^\top U^\top U\Sigma V^\top.
$$

Because the columns of \(U\) are orthonormal,

$$
U^\top U=I.
$$

Therefore,

$$
C
=
V\frac{\Sigma^2}{n-1}V^\top.
$$

This is precisely an eigendecomposition of \(C\).

Hence:

$$
\boxed{\text{principal directions} = \text{columns of }V}
$$

and

$$
\boxed{\lambda_j = \frac{\sigma_j^2}{n-1}}
$$

where:

- \(\sigma_j\) is the \(j\)-th singular value;
- \(\lambda_j\) is the variance explained by the \(j\)-th component.

This is the heart of PCA via SVD.

---

# 10. Transforming the data

Let the first \(k\) principal directions form

$$
V_k =
\begin{bmatrix}
v_1 & v_2 & \cdots & v_k
\end{bmatrix}
\in \mathbb{R}^{d\times k}.
$$

The low-dimensional representation is

$$
Z = X_cV_k.
$$

Shapes:

$$
X_c: n\times d,
\qquad
V_k: d\times k,
\qquad
Z: n\times k.
$$

Each row of \(Z\) is the original observation expressed in the new coordinate system.

Using SVD,

$$
X_cV
=
U\Sigma V^\top V
=
U\Sigma.
$$

Therefore,

$$
Z_k = U_k\Sigma_k.
$$

The following are mathematically equivalent:

$$
Z_k=X_cV_k
$$

and

$$
Z_k=U_k\Sigma_k.
$$

For implementation, using \(X_cV_k\) makes the transformation logic especially clear.

---

# 11. Explained variance

The variance explained by component \(j\) is

$$
\lambda_j
=
\frac{\sigma_j^2}{n-1}.
$$

The explained variance ratio is

$$
\rho_j
=
\frac{\lambda_j}{\sum_{\ell}\lambda_\ell}.
$$

Since the factor \(1/(n-1)\) cancels,

$$
\rho_j
=
\frac{\sigma_j^2}
{\sum_{\ell}\sigma_\ell^2}.
$$

The cumulative explained variance for \(k\) components is

$$
R_k=\sum_{j=1}^{k}\rho_j.
$$

A common rule is to choose the smallest \(k\) satisfying

$$
R_k \geq \tau,
$$

where \(\tau\) might be \(0.90\), \(0.95\), or \(0.99\).

This threshold is a design choice, not a mathematical law.

---

# 12. Reconstructing the original data

The compressed representation is

$$
Z=X_cV_k.
$$

To map it back into the original feature space:

$$
\hat{X}_c=ZV_k^\top.
$$

Then add the mean back:

$$
\hat{X}=ZV_k^\top+\mu.
$$

Combining the equations:

$$
\hat{X}
=
X_cV_kV_k^\top+\mu.
$$

The matrix

$$
V_kV_k^\top
$$

projects observations onto the \(k\)-dimensional principal subspace.

Unless \(k\) includes every non-zero component, reconstruction is imperfect.

The reconstruction error for observation \(i\) is often measured by

$$
e_i
=
\|x_i-\hat{x}_i\|_2^2.
$$

This will matter for fraud detection.

---

# 13. Why PCA gives the best linear compression

Among all rank-\(k\) linear approximations of \(X_c\), truncated SVD minimises reconstruction error:

$$
\min_{\operatorname{rank}(\hat{X}_c)\leq k}
\|X_c-\hat{X}_c\|_F^2.
$$

The solution is

$$
\hat{X}_c
=
U_k\Sigma_kV_k^\top.
$$

The Frobenius norm is

$$
\|A\|_F^2
=
\sum_{i,j}A_{ij}^2.
$$

So PCA finds the \(k\)-dimensional linear subspace that loses the least total squared information.

This is known as the **Eckart–Young theorem**.

The discarded reconstruction error is exactly

$$
\|X_c-\hat{X}_c\|_F^2
=
\sum_{j=k+1}^{r}\sigma_j^2.
$$

Every discarded singular value measures information lost in one omitted direction.

---

# 14. Numerical example

Consider four two-dimensional observations:

$$
X=
\begin{bmatrix}
2 & 1 \\
3 & 2 \\
4 & 3 \\
5 & 4
\end{bmatrix}.
$$

The two features are perfectly related:

$$
x_2=x_1-1.
$$

So although the data is stored in two dimensions, it contains only one independent direction.

## 14.1 Compute the mean

$$
\mu=
\begin{bmatrix}
3.5 & 2.5
\end{bmatrix}.
$$

## 14.2 Centre the data

$$
X_c=
\begin{bmatrix}
-1.5 & -1.5 \\
-0.5 & -0.5 \\
0.5 & 0.5 \\
1.5 & 1.5
\end{bmatrix}.
$$

All points lie along the direction

$$
\begin{bmatrix}
1\\1
\end{bmatrix}.
$$

The corresponding unit vector is

$$
v_1=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\1
\end{bmatrix}.
$$

A perpendicular direction is

$$
v_2=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\-1
\end{bmatrix}.
$$

There is variance along \(v_1\), but zero variance along \(v_2\).

## 14.3 Covariance matrix

$$
X_c^\top X_c
=
\begin{bmatrix}
5 & 5 \\
5 & 5
\end{bmatrix}.
$$

Since \(n=4\),

$$
C=
\frac{1}{3}
\begin{bmatrix}
5 & 5 \\
5 & 5
\end{bmatrix}
=
\begin{bmatrix}
5/3 & 5/3 \\
5/3 & 5/3
\end{bmatrix}.
$$

Its eigenvalues are

$$
\lambda_1=\frac{10}{3},
\qquad
\lambda_2=0.
$$

Therefore, the first component explains

$$
\frac{10/3}{10/3+0}=1
$$

or \(100\%\) of the variance.

## 14.4 Project onto the first component

$$
Z=X_cv_1.
$$

Thus,

$$
Z=
\begin{bmatrix}
-1.5 & -1.5 \\
-0.5 & -0.5 \\
0.5 & 0.5 \\
1.5 & 1.5
\end{bmatrix}
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\1
\end{bmatrix}.
$$

So

$$
Z=
\begin{bmatrix}
-3/\sqrt{2}\\
-1/\sqrt{2}\\
1/\sqrt{2}\\
3/\sqrt{2}
\end{bmatrix}
\approx
\begin{bmatrix}
-2.121\\
-0.707\\
0.707\\
2.121
\end{bmatrix}.
$$

We have compressed every observation from two numbers to one number without losing information.

## 14.5 Reconstruct

$$
\hat{X}_c=Zv_1^\top.
$$

For the first observation:

$$
\hat{x}_{c,1}
=
-\frac{3}{\sqrt{2}}
\cdot
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1
\end{bmatrix}
=
\begin{bmatrix}
-1.5 & -1.5
\end{bmatrix}.
$$

Adding the mean:

$$
\hat{x}_1
=
\begin{bmatrix}
-1.5 & -1.5
\end{bmatrix}
+
\begin{bmatrix}
3.5 & 2.5
\end{bmatrix}
=
\begin{bmatrix}
2 & 1
\end{bmatrix}.
$$

The reconstruction is exact because the data truly has rank one.

---

# 15. PCA algorithm via SVD

Given

$$
X\in\mathbb{R}^{n\times d}
$$

and a target number of components \(k\):

### Training

1. Compute the feature mean:

$$
\mu=\frac{1}{n}\sum_{i=1}^{n}x_i.
$$

2. Centre the data:

$$
X_c=X-\mu.
$$

3. Compute the reduced SVD:

$$
X_c=U\Sigma V^\top.
$$

4. Retain the first \(k\) right singular vectors:

$$
V_k=V[:,1:k].
$$

5. Compute explained variances:

$$
\lambda_j=\frac{\sigma_j^2}{n-1}.
$$

### Transformation

$$
Z=(X-\mu)V_k.
$$

### Reconstruction

$$
\hat{X}=ZV_k^\top+\mu.
$$

---

# 16. NumPy-only implementation

```python
import numpy as np


class PCA:
    """
    Principal Component Analysis implemented using NumPy SVD.

    Parameters
    ----------
    n_components : int, float, or None
        - int: retain exactly this many components.
        - float in (0, 1]: retain the smallest number of components
          whose cumulative explained variance reaches this threshold.
        - None: retain all available components.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

        self.mean_ = None
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None
        self.n_features_in_ = None

    def fit(self, X):
        X = self._validate_X(X)

        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError("PCA requires at least two observations.")

        self.n_features_in_ = n_features

        # 1. Compute the feature-wise mean.
        self.mean_ = np.mean(X, axis=0)

        # 2. Centre the data.
        X_centered = X - self.mean_

        # 3. Reduced singular value decomposition.
        #
        # X_centered = U @ diag(singular_values) @ Vt
        U, singular_values, Vt = np.linalg.svd(
            X_centered,
            full_matrices=False
        )

        # 4. Variance associated with each principal direction.
        explained_variance = singular_values**2 / (n_samples - 1)

        total_variance = np.sum(explained_variance)

        if total_variance > 0:
            explained_variance_ratio = (
                explained_variance / total_variance
            )
        else:
            # Every observation is identical.
            explained_variance_ratio = np.zeros_like(
                explained_variance
            )

        # 5. Determine how many components to retain.
        k = self._resolve_n_components(
            explained_variance_ratio,
            max_components=len(singular_values)
        )

        self.n_components_ = k

        # NumPy returns principal directions as rows of Vt.
        self.components_ = Vt[:k]
        self.singular_values_ = singular_values[:k]
        self.explained_variance_ = explained_variance[:k]
        self.explained_variance_ratio_ = (
            explained_variance_ratio[:k]
        )

        return self

    def transform(self, X):
        self._check_fitted()
        X = self._validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"received {X.shape[1]}."
            )

        X_centered = X - self.mean_

        # components_: (k, d)
        # components_.T: (d, k)
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        self._check_fitted()

        Z = np.asarray(Z, dtype=np.float64)

        if Z.ndim == 1:
            Z = Z.reshape(1, -1)

        if Z.ndim != 2:
            raise ValueError("Z must be a two-dimensional array.")

        if Z.shape[1] != self.n_components_:
            raise ValueError(
                f"Expected {self.n_components_} components, "
                f"received {Z.shape[1]}."
            )

        # Z: (n, k)
        # components_: (k, d)
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X):
        """
        Return squared reconstruction error for each observation.
        """
        X = self._validate_X(X)
        Z = self.transform(X)
        X_reconstructed = self.inverse_transform(Z)

        return np.sum((X - X_reconstructed) ** 2, axis=1)

    def _resolve_n_components(
        self,
        explained_variance_ratio,
        max_components
    ):
        if self.n_components is None:
            return max_components

        if isinstance(self.n_components, (int, np.integer)):
            if not 1 <= self.n_components <= max_components:
                raise ValueError(
                    "Integer n_components must satisfy "
                    f"1 <= n_components <= {max_components}."
                )

            return int(self.n_components)

        if isinstance(
            self.n_components,
            (float, np.floating)
        ):
            threshold = float(self.n_components)

            if not 0.0 < threshold <= 1.0:
                raise ValueError(
                    "Float n_components must lie in (0, 1]."
                )

            cumulative_variance = np.cumsum(
                explained_variance_ratio
            )

            # searchsorted returns the first index where
            # cumulative variance reaches the threshold.
            return int(
                np.searchsorted(
                    cumulative_variance,
                    threshold
                ) + 1
            )

        raise TypeError(
            "n_components must be an integer, a float, or None."
        )

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")

        return X

    def _check_fitted(self):
        if self.components_ is None:
            raise RuntimeError(
                "The PCA model must be fitted before use."
            )
```

---

# 17. Run the numerical example

```python
X = np.array([
    [2.0, 1.0],
    [3.0, 2.0],
    [4.0, 3.0],
    [5.0, 4.0],
])

pca = PCA(n_components=1)

Z = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(Z)
errors = pca.reconstruction_error(X)

print("Mean:")
print(pca.mean_)

print("\nPrincipal direction:")
print(pca.components_)

print("\nSingular value:")
print(pca.singular_values_)

print("\nExplained variance:")
print(pca.explained_variance_)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print("\nTransformed data:")
print(Z)

print("\nReconstructed data:")
print(X_reconstructed)

print("\nReconstruction errors:")
print(errors)
```

The output will be approximately:

```text
Mean:
[3.5 2.5]

Principal direction:
[[0.70710678 0.70710678]]

Singular value:
[3.16227766]

Explained variance:
[3.33333333]

Explained variance ratio:
[1.]

Transformed data:
[[-2.12132034]
 [-0.70710678]
 [ 0.70710678]
 [ 2.12132034]]

Reconstructed data:
[[2. 1.]
 [3. 2.]
 [4. 3.]
 [5. 4.]]

Reconstruction errors:
[0. 0. 0. 0.]
```

The principal direction might appear with negative signs:

```text
[[-0.70710678 -0.70710678]]
```

That is not an error.

A direction \(v\) and its negation \(-v\) describe the same axis. The projected values also change signs, but reconstruction remains identical:

$$
(X_c(-v))(-v)^\top
=
(X_cv)v^\top.
$$

---

# 18. Choosing components by variance threshold

The class also supports:

```python
pca = PCA(n_components=0.95)
Z = pca.fit_transform(X)
```

This retains the smallest number \(k\) such that

$$
\sum_{j=1}^{k}\rho_j \geq 0.95.
$$

For example, suppose the explained variance ratios are

$$
[0.60,\ 0.25,\ 0.10,\ 0.05].
$$

Their cumulative sums are

$$
[0.60,\ 0.85,\ 0.95,\ 1.00].
$$

Then a \(95\%\) threshold retains three components.

---

# 19. Feature scaling: an essential practical issue

PCA is sensitive to feature scale.

Suppose a fraud dataset contains:

- transaction amount: \(0\) to \(100{,}000\);
- merchant risk score: \(0\) to \(1\);
- transaction hour: \(0\) to \(23\).

The amount feature may dominate the variance simply because its numerical scale is much larger.

PCA does not know that dollars and risk scores use different units. It sees only numbers.

A common solution is standardisation:

$$
X'_{ij}
=
\frac{X_{ij}-\mu_j}{s_j},
$$

where \(s_j\) is the sample standard deviation of feature \(j\).

Then every feature has approximately:

$$
\text{mean}=0,
\qquad
\text{variance}=1.
$$

A NumPy-only standardiser is:

```python
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=1)

        # Avoid division by zero for constant features.
        self.scale_ = np.where(
            self.scale_ == 0.0,
            1.0,
            self.scale_
        )

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        X_scaled = np.asarray(X_scaled, dtype=np.float64)
        return X_scaled * self.scale_ + self.mean_
```

Usage:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=0.95)
Z_train = pca.fit_transform(X_scaled)
```

For new observations:

```python
X_new_scaled = scaler.transform(X_new)
Z_new = pca.transform(X_new_scaled)
```

Never fit the scaler or PCA separately on validation or test data. That would leak information from those datasets into training.

---

# 20. PCA for fraud detection

PCA can be used as a linear anomaly detector.

Assume most training transactions are legitimate and lie near a low-dimensional subspace.

Train PCA on normal transactions:

$$
X_{\text{normal}}.
$$

For a new transaction \(x\):

1. centre and optionally standardise it;
2. project it into the PCA subspace;
3. reconstruct it;
4. calculate reconstruction error.

The reconstruction is

$$
\hat{x}
=
V_kV_k^\top(x-\mu)+\mu.
$$

The anomaly score is

$$
s(x)
=
\|x-\hat{x}\|_2^2.
$$

A transaction unlike the normal training structure may have high reconstruction error.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)

pca = PCA(n_components=0.95)
pca.fit(X_train_scaled)

X_test_scaled = scaler.transform(X_test)

scores = pca.reconstruction_error(X_test_scaled)
```

Choose a threshold using validation data:

```python
threshold = np.quantile(validation_normal_scores, 0.99)

predicted_fraud = test_scores > threshold
```

This says:

> Mark a transaction anomalous when its reconstruction error exceeds 99% of the errors observed among normal validation transactions.

The quantile is not universally optimal. In a real fraud system, choose the threshold using business costs, precision, recall and investigation capacity.

---

# 21. PCA versus an autoencoder

PCA and autoencoders solve closely related problems.

Both learn:

$$
x
\rightarrow
z
\rightarrow
\hat{x}.
$$

For PCA:

$$
z=(x-\mu)V_k,
$$

$$
\hat{x}=zV_k^\top+\mu.
$$

For an autoencoder:

$$
z=f_\theta(x),
$$

$$
\hat{x}=g_\phi(z).
$$

The essential difference is the family of functions they can represent.

## PCA

PCA learns a linear subspace:

$$
z=W^\top x.
$$

Its assumptions are approximately:

- meaningful structure is linear;
- variance is a useful measure of information;
- squared reconstruction error is appropriate;
- global directions describe the dataset adequately.

## Autoencoder

An autoencoder may learn nonlinear mappings:

$$
z=f_\theta(x).
$$

It can potentially model curved structures, interactions and more complicated distributions.

But it also introduces:

- optimisation difficulty;
- more hyperparameters;
- risk of overfitting;
- less interpretability;
- greater computational cost.

A linear autoencoder with:

- one linear hidden layer;
- mean squared reconstruction loss;
- no nonlinear activation;
- a bottleneck of size \(k\);

learns essentially the same principal subspace as PCA, although its coordinates may be rotated within that subspace.

PCA is therefore an excellent baseline for your fraud autoencoder project.

A strong portfolio comparison is:

1. PCA reconstruction anomaly detector;
2. linear autoencoder;
3. nonlinear autoencoder;
4. supervised fraud classifier, where labels exist.

That lets you demonstrate exactly what nonlinear modelling adds beyond the best linear reconstruction model.

---

# 22. Important assumptions and limitations

## PCA captures linear structure

If normal transactions lie near a curved manifold, PCA approximates that curve using a flat subspace.

A nonlinear autoencoder may represent the structure better.

## High variance is not always useful

A high-variance feature may be noise.

A low-variance feature may be highly predictive of fraud.

PCA is unsupervised: it does not inspect fraud labels.

## PCA is sensitive to outliers

Fraudulent observations may strongly rotate the principal directions because they can contribute large variance.

For anomaly detection, PCA is usually trained primarily or exclusively on legitimate observations.

## Components are global

PCA finds one linear coordinate system for the entire dataset. Different transaction populations may follow different patterns.

For example:

- retail customers;
- corporate customers;
- international transfers;
- cash withdrawals.

One global PCA model may poorly represent all groups.

## PCA does not model probability directly

A reconstruction error is an anomaly score, not a calibrated fraud probability.

---

# 23. Computational complexity

For

$$
X\in\mathbb{R}^{n\times d},
$$

the reduced SVD typically costs roughly

$$
O\bigl(\min(nd^2,n^2d)\bigr).
$$

When

$$
n \gg d,
$$

the cost is approximately

$$
O(nd^2).
$$

Transforming \(n\) observations into \(k\) components costs

$$
O(ndk).
$$

Storing the principal directions costs

$$
O(dk).
$$

For very large transaction datasets, full SVD may become expensive. Production systems may use randomised SVD, incremental PCA or distributed matrix methods. But full NumPy SVD is the correct place to learn the underlying algorithm.

---

# 24. The complete mental model

PCA can be understood as one connected chain:

### 1. Data contains redundancy

Several observed features may reflect fewer hidden factors.

### 2. Centre the data

$$
X_c=X-\mu.
$$

Now the origin represents the centre of the dataset.

### 3. Search for directions of variation

For a unit vector \(v\), projected variance is

$$
v^\top Cv.
$$

### 4. Maximising variance produces eigenvectors

$$
Cv=\lambda v.
$$

### 5. SVD extracts those directions directly

$$
X_c=U\Sigma V^\top.
$$

The columns of \(V\) are principal directions.

### 6. Singular values determine explained variance

$$
\lambda_j=\frac{\sigma_j^2}{n-1}.
$$

### 7. Keep the strongest \(k\) directions

$$
Z=X_cV_k.
$$

### 8. Reconstruct from the compressed representation

$$
\hat{X}=ZV_k^\top+\mu.
$$

### 9. Discarded directions become reconstruction error

$$
e_i=\|x_i-\hat{x}_i\|_2^2.
$$

### 10. Large reconstruction error can indicate fraud

A transaction that does not fit normal linear structure is considered suspicious.

The core idea is:

$$
\boxed{
\text{PCA finds the low-dimensional linear subspace that preserves the most variance}
}
$$

or, equivalently,

$$
\boxed{
\text{PCA finds the rank-}k\text{ linear reconstruction with minimum squared error}
}
$$

These are not two separate interpretations. They are two views of the same optimisation problem.
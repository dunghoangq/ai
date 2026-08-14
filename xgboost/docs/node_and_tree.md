# Building `Node` and `DecisionTree` for XGBoost from Scratch

## 1. What Are We Actually Building?

You already know that XGBoost builds an additive model:

$$
\hat y_i^{(t)}
=
\hat y_i^{(t-1)} + \eta f_t(x_i)
$$

where \(f_t\) is a regression tree.

So the immediate implementation problem is not yet:

> How do I build XGBoost?

It is:

> Given gradients and Hessians for every training observation, how do I construct **one tree** that minimises the second-order approximation of the loss?

That tree needs only two fundamental objects:

1. `Node` — represents one region of feature space.
2. `DecisionTree` — recursively builds and traverses those nodes.

Everything else follows from these two.

---

# 2. First Principle: What Does a Node Represent?

Consider one node containing observations indexed by

$$
I = \{i_1,i_2,\dots,i_m\}
$$

The node represents a subset of the training space.

For example:

```text
root:
    all observations

left child:
    Age < 35

right child:
    Age >= 35
```

A node therefore needs to answer two possible questions:

### If it is an internal node

It must know:

$$
\text{feature index}
$$

and

$$
\text{split threshold}
$$

so that prediction can route an observation:

```python
if x[feature] < threshold:
    go left
else:
    go right
```

### If it is a leaf

It must know the prediction value:

$$
w^*
$$

In XGBoost,

$$
w^*
=
-\frac{G}{H+\lambda}
$$

where

$$
G=\sum_{i\in I}g_i
$$

and

$$
H=\sum_{i\in I}h_i
$$

Thus a leaf does not predict a class directly.

It predicts a **correction to the current model prediction**.

That distinction matters enormously.

For logistic classification, the leaf output modifies the **logit**, not directly the probability.

---

# 3. Minimal `Node` Design

Start with the smallest useful structure.

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None

    left: Optional["Node"] = None
    right: Optional["Node"] = None

    value: Optional[float] = None
```

Why these fields?

| Field | Meaning |
|---|---|
| `feature_idx` | feature used for splitting |
| `threshold` | split value |
| `left` | left subtree |
| `right` | right subtree |
| `value` | leaf prediction |

A leaf might look like:

```python
Node(value=-0.37)
```

An internal node might look like:

```python
Node(
    feature_idx=4,
    threshold=2.7,
    left=...,
    right=...
)
```

Notice something important:

An internal node does **not** need a prediction value.

A leaf does **not** need a feature or threshold.

This naturally suggests a helper property.

```python
@dataclass
class Node:
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[float] = None

    @property
    def is_leaf(self) -> bool:
        return self.value is not None
```

Now tree traversal becomes clean:

```python
if node.is_leaf:
    return node.value
```

---

# 4. Should the Node Store Training Samples?

You might be tempted to write:

```python
Node(
    X=X_subset,
    gradients=g_subset,
    hessians=h_subset
)
```

Avoid this in the final tree.

Why?

Once training finishes, the tree only needs enough information for prediction:

```text
feature
threshold
children
leaf value
```

Training data are temporary.

Keeping them inside every node causes unnecessary memory consumption.

For learning purposes, you might temporarily store:

```python
n_samples
gain
depth
```

for debugging.

A useful version is:

```python
@dataclass
class Node:
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None

    left: Optional["Node"] = None
    right: Optional["Node"] = None

    value: Optional[float] = None

    gain: float = 0.0
    n_samples: int = 0

    @property
    def is_leaf(self):
        return self.value is not None
```

`gain` is not needed for prediction, but it is extremely useful when debugging your implementation.

---

# 5. What Does `DecisionTree` Need?

The tree controls the building process.

Conceptually:

```text
DecisionTree.fit()
        |
        v
build root
        |
        v
find best split
        |
        +--------+
        |        |
        v        v
    build left  build right
        |        |
        +--------+
```

So the tree needs methods corresponding almost exactly to this logic:

```python
fit()
_build_tree()
_find_best_split()
_split()
_leaf_value()
_gain()
predict()
_predict_one()
```

That is already almost the complete implementation.

---

# 6. Constructor

Let us begin with the hyperparameters that affect tree growth.

```python
class DecisionTree:

    def __init__(
        self,
        max_depth=3,
        min_samples_split=2,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight

        self.root = None
```

Each parameter has a mathematical meaning.

---

# 7. `max_depth`

Without a stopping condition, recursive splitting could continue until every sample receives its own leaf.

So:

```python
max_depth
```

controls structural complexity.

At depth \(d\):

```text
depth 0: root
depth 1: children
depth 2: grandchildren
...
```

For your first implementation:

```python
max_depth = 3
```

is enough.

Do not begin with unlimited-depth trees.

---

# 8. `reg_lambda`

XGBoost penalises large leaf weights:

$$
\Omega(f)
=
\gamma T
+
\frac{1}{2}\lambda\sum_jw_j^2
$$

The \(\lambda\) term appears directly in:

$$
w_j^*
=
-\frac{G_j}{H_j+\lambda}
$$

Therefore your tree object needs `reg_lambda`.

---

# 9. `gamma`

A split creates another leaf.

XGBoost charges a structural complexity penalty:

$$
\gamma
$$

for creating leaves.

Therefore a split is accepted only if its improvement is sufficiently large.

Informally:

```text
small improvement
    ↓
not worth additional complexity
    ↓
do not split
```

---

# 10. `min_child_weight`

In XGBoost, this is related to the total Hessian in a child node.

For a candidate split:

$$
H_L \geq \text{min\_child\_weight}
$$

and

$$
H_R \geq \text{min\_child\_weight}
$$

must hold.

This is different from ordinary decision trees.

A standard tree often asks:

```text
Does this child contain enough observations?
```

XGBoost effectively asks:

```text
Does this child contain enough optimisation information?
```

The Hessian measures curvature of the loss.

---

# 11. `fit`: What Does the Tree Receive?

A normal decision tree receives:

```python
X, y
```

But your XGBoost tree should receive:

```python
X, gradients, hessians
```

because the boosting algorithm has already differentiated the loss.

```python
def fit(self, X, gradients, hessians):
    self.root = self._build_tree(
        X,
        gradients,
        hessians,
        depth=0
    )
    return self
```

This separation is excellent software design.

The tree does not need to know whether you are solving:

- squared-error regression,
- logistic classification,
- something else.

The loss function computes:

$$
g_i,\ h_i
$$

and the tree only consumes them.

Therefore:

```text
Loss function
    ↓
gradients + Hessians
    ↓
DecisionTree
```

This decoupling is one of the cleanest ways to design your implementation.

---

# 12. Leaf Value

For observations inside one node:

$$
G=\sum_i g_i
$$

$$
H=\sum_i h_i
$$

The optimal leaf value is

$$
w^*
=
-\frac{G}{H+\lambda}
$$

so:

```python
def _leaf_value(self, gradients, hessians):
    G = gradients.sum()
    H = hessians.sum()

    return -G / (H + self.reg_lambda)
```

This is the first core mathematical method.

---

# 13. Why This Formula Exists

Suppose a node predicts the same value \(w\) for every sample inside it.

The approximate objective is:

$$
\sum_i
\left[
g_iw+\frac12h_iw^2
\right]
+
\frac12\lambda w^2
$$

Collecting terms:

$$
Gw+\frac12(H+\lambda)w^2
$$

Differentiate:

$$
\frac{dL}{dw}
=
G+(H+\lambda)w
$$

Set to zero:

$$
G+(H+\lambda)w=0
$$

Therefore:

$$
w^*
=
-\frac{G}{H+\lambda}
$$

So `_leaf_value()` is not an arbitrary prediction heuristic.

It is the exact minimiser of the second-order local objective.

---

# 14. Node Score

Before understanding split gain, define the quality of one leaf.

Plugging the optimal weight back into the objective gives:

$$
-\frac12
\frac{G^2}{H+\lambda}
$$

Because we care about improvement, we commonly work with the positive score:

$$
S(G,H)
=
\frac{G^2}{H+\lambda}
$$

Implement:

```python
def _score(self, gradients, hessians):
    G = gradients.sum()
    H = hessians.sum()

    return (G * G) / (H + self.reg_lambda)
```

This gives us the building block for split gain.

---

# 15. Split Gain

Suppose a node is divided into:

$$
I_L,\ I_R
$$

Let:

$$
G_L,H_L
$$

be the gradient/Hessian sums on the left, and

$$
G_R,H_R
$$

on the right.

The split gain is

$$
\text{Gain}
=
\frac12
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}
\right]
-
\gamma
$$

Implement directly:

```python
def _gain(self, g_left, h_left, g_right, h_right):

    G_L = g_left.sum()
    H_L = h_left.sum()

    G_R = g_right.sum()
    H_R = h_right.sum()

    G = G_L + G_R
    H = H_L + H_R

    gain = 0.5 * (
        G_L**2 / (H_L + self.reg_lambda)
        + G_R**2 / (H_R + self.reg_lambda)
        - G**2 / (H + self.reg_lambda)
    ) - self.gamma

    return gain
```

---

# 16. What Gain Really Means

Do not memorise the formula as "some XGBoost equation".

Conceptually:

$$
\text{Gain}
=
\text{quality after split}
-
\text{quality before split}
-
\text{complexity cost}
$$

Specifically:

```text
left leaf score
+
right leaf score
-
original leaf score
-
gamma
```

So your algorithm asks:

> If I replace this one leaf with two specialised leaves, how much does the approximate objective improve?

That is all tree construction is doing.

---

# 17. Candidate Splits

Suppose one feature contains:

```python
[1.2, 2.7, 0.8, 4.1, 3.0]
```

A simple implementation can consider every unique value:

```python
thresholds = np.unique(feature_values)
```

Then test:

```python
left_mask = feature_values < threshold
right_mask = ~left_mask
```

This is perfectly acceptable for your first from-scratch version.

It is not computationally optimal.

That comes later.

---

# 18. Why `< threshold`?

You need one deterministic split convention.

For example:

```python
X[:, feature_idx] < threshold
```

goes left.

Everything else goes right.

You could use:

```python
<=
```

instead.

Neither is fundamentally superior.

The important thing is consistency between:

```python
training
```

and

```python
prediction
```

If training uses `<` but prediction uses `<=`, you have created a subtle bug.

---

# 19. `_split`

Make splitting its own method.

```python
def _split(self, X, feature_idx, threshold):
    left_mask = X[:, feature_idx] < threshold
    right_mask = ~left_mask

    return left_mask, right_mask
```

Why make such a tiny method?

Because it centralises your split convention.

That gives you exactly one place to modify later when handling:

- missing values,
- categorical features,
- sparse matrices.

---

# 20. Finding the Best Split

This is the computational heart of the tree.

The brute-force algorithm is:

```text
best_gain = -infinity

for every feature:
    for every candidate threshold:
        partition samples
        compute left G, H
        compute right G, H
        calculate gain
        keep best split
```

Implementation:

```python
import numpy as np


def _find_best_split(self, X, gradients, hessians):

    n_samples, n_features = X.shape

    best_gain = -np.inf
    best_feature = None
    best_threshold = None
    best_left_mask = None
    best_right_mask = None

    for feature_idx in range(n_features):

        feature_values = X[:, feature_idx]
        thresholds = np.unique(feature_values)

        for threshold in thresholds:

            left_mask, right_mask = self._split(
                X,
                feature_idx,
                threshold
            )

            if not left_mask.any() or not right_mask.any():
                continue

            g_left = gradients[left_mask]
            h_left = hessians[left_mask]

            g_right = gradients[right_mask]
            h_right = hessians[right_mask]

            if h_left.sum() < self.min_child_weight:
                continue

            if h_right.sum() < self.min_child_weight:
                continue

            gain = self._gain(
                g_left,
                h_left,
                g_right,
                h_right
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
                best_left_mask = left_mask
                best_right_mask = right_mask

    return (
        best_gain,
        best_feature,
        best_threshold,
        best_left_mask,
        best_right_mask,
    )
```

This is not fast.

That is fine.

Your first goal should be:

> correctness before optimisation.

---

# 21. Why Store the Masks?

You could return only:

```python
best_feature
best_threshold
```

and recompute the masks afterwards.

But returning the masks avoids repeating work.

So:

```python
_find_best_split()
```

returns everything `_build_tree()` needs to continue recursion.

Later, once you optimise the implementation, you might instead pass index arrays rather than repeatedly copying `X`.

For now, readability wins.

---

# 22. Threshold Choice: Unique Values vs Midpoints

Suppose sorted values are:

```text
1, 2, 5, 7
```

You could test thresholds:

```text
1, 2, 5, 7
```

But conceptually, decision boundaries lie **between observations**:

```text
1.5, 3.5, 6
```

These are computed as:

$$
t_i = \frac{x_i+x_{i+1}}{2}
$$

Implementation:

```python
values = np.unique(feature_values)

thresholds = (
    values[:-1] + values[1:]
) / 2
```

This is slightly cleaner mathematically.

Why?

Because if your split is:

```python
x < 3.5
```

then all values up to `2` go left and all values from `5` onward go right.

No training observation sits exactly on the boundary.

For your implementation I would recommend **midpoint thresholds**.

```python
def _candidate_thresholds(self, feature_values):

    values = np.unique(feature_values)

    if len(values) <= 1:
        return np.array([])

    return (values[:-1] + values[1:]) / 2
```

Then:

```python
thresholds = self._candidate_thresholds(feature_values)
```

---

# 23. Recursive Tree Construction

Now all pieces exist.

The recursive logic is:

```text
build(node):

    should we stop?
        yes → leaf

    find best split

    no useful split?
        yes → leaf

    recursively construct left child
    recursively construct right child

    return internal node
```

Implementation:

```python
def _build_tree(
    self,
    X,
    gradients,
    hessians,
    depth,
):

    n_samples = X.shape[0]

    if depth >= self.max_depth:
        return Node(
            value=self._leaf_value(gradients, hessians),
            n_samples=n_samples,
        )

    if n_samples < self.min_samples_split:
        return Node(
            value=self._leaf_value(gradients, hessians),
            n_samples=n_samples,
        )

    (
        best_gain,
        best_feature,
        best_threshold,
        left_mask,
        right_mask,
    ) = self._find_best_split(
        X,
        gradients,
        hessians
    )

    if best_feature is None or best_gain <= 0:
        return Node(
            value=self._leaf_value(gradients, hessians),
            n_samples=n_samples,
        )

    left_child = self._build_tree(
        X[left_mask],
        gradients[left_mask],
        hessians[left_mask],
        depth + 1
    )

    right_child = self._build_tree(
        X[right_mask],
        gradients[right_mask],
        hessians[right_mask],
        depth + 1
    )

    return Node(
        feature_idx=best_feature,
        threshold=best_threshold,
        left=left_child,
        right=right_child,
        gain=best_gain,
        n_samples=n_samples,
    )
```

This method now expresses almost the entire XGBoost tree algorithm.

---

# 24. Understand the Recursion Deeply

Suppose the root finds:

```text
feature 2 < 4.5
```

Then:

```text
                 root
             X[:,2] < 4.5
              /        \
             /          \
      subset A          subset B
```

Then `_build_tree()` is independently called on both subsets.

That means the exact same optimisation problem happens again:

```text
Within subset A:
    which split gives maximum gain?

Within subset B:
    which split gives maximum gain?
```

This recursive structure is what turns a single split into a tree.

There is no separate algorithm for the second level.

It is the same algorithm applied to a smaller subset.

---

# 25. Prediction

Prediction is easier than training.

For one observation:

```python
def _predict_one(self, x, node):

    if node.is_leaf:
        return node.value

    if x[node.feature_idx] < node.threshold:
        return self._predict_one(x, node.left)

    return self._predict_one(x, node.right)
```

For the whole dataset:

```python
def predict(self, X):

    return np.array([
        self._predict_one(x, self.root)
        for x in X
    ])
```

Again, remember:

`DecisionTree.predict()` returns:

$$
f_t(x)
$$

the correction produced by this particular boosting tree.

It does **not** necessarily return:

- probability,
- class,
- final XGBoost prediction.

---

# 26. Complete First Version

You can now combine everything.

```python
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Node:
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None

    left: Optional["Node"] = None
    right: Optional["Node"] = None

    value: Optional[float] = None

    gain: float = 0.0
    n_samples: int = 0

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTree:

    def __init__(
        self,
        max_depth=3,
        min_samples_split=2,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight

        self.root = None

    def fit(self, X, gradients, hessians):

        X = np.asarray(X, dtype=float)
        gradients = np.asarray(gradients, dtype=float)
        hessians = np.asarray(hessians, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if len(gradients) != len(X):
            raise ValueError(
                "gradients must have one value per sample."
            )

        if len(hessians) != len(X):
            raise ValueError(
                "hessians must have one value per sample."
            )

        self.root = self._build_tree(
            X,
            gradients,
            hessians,
            depth=0
        )

        return self

    def _leaf_value(self, gradients, hessians):

        G = gradients.sum()
        H = hessians.sum()

        return -G / (H + self.reg_lambda)

    def _gain(
        self,
        g_left,
        h_left,
        g_right,
        h_right
    ):

        G_L = g_left.sum()
        H_L = h_left.sum()

        G_R = g_right.sum()
        H_R = h_right.sum()

        G = G_L + G_R
        H = H_L + H_R

        gain = 0.5 * (
            G_L**2 / (H_L + self.reg_lambda)
            + G_R**2 / (H_R + self.reg_lambda)
            - G**2 / (H + self.reg_lambda)
        ) - self.gamma

        return gain

    def _candidate_thresholds(self, feature_values):

        values = np.unique(feature_values)

        if len(values) <= 1:
            return np.array([])

        return (
            values[:-1] + values[1:]
        ) / 2

    def _split(
        self,
        X,
        feature_idx,
        threshold
    ):

        left_mask = X[:, feature_idx] < threshold
        right_mask = ~left_mask

        return left_mask, right_mask

    def _find_best_split(
        self,
        X,
        gradients,
        hessians
    ):

        _, n_features = X.shape

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        best_left_mask = None
        best_right_mask = None

        for feature_idx in range(n_features):

            feature_values = X[:, feature_idx]

            thresholds = self._candidate_thresholds(
                feature_values
            )

            for threshold in thresholds:

                left_mask, right_mask = self._split(
                    X,
                    feature_idx,
                    threshold
                )

                if not left_mask.any():
                    continue

                if not right_mask.any():
                    continue

                g_left = gradients[left_mask]
                h_left = hessians[left_mask]

                g_right = gradients[right_mask]
                h_right = hessians[right_mask]

                if (
                    h_left.sum()
                    < self.min_child_weight
                ):
                    continue

                if (
                    h_right.sum()
                    < self.min_child_weight
                ):
                    continue

                gain = self._gain(
                    g_left,
                    h_left,
                    g_right,
                    h_right
                )

                if gain > best_gain:

                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

                    best_left_mask = left_mask
                    best_right_mask = right_mask

        return (
            best_gain,
            best_feature,
            best_threshold,
            best_left_mask,
            best_right_mask,
        )

    def _build_tree(
        self,
        X,
        gradients,
        hessians,
        depth,
    ):

        n_samples = X.shape[0]

        if depth >= self.max_depth:

            return Node(
                value=self._leaf_value(
                    gradients,
                    hessians
                ),
                n_samples=n_samples,
            )

        if n_samples < self.min_samples_split:

            return Node(
                value=self._leaf_value(
                    gradients,
                    hessians
                ),
                n_samples=n_samples,
            )

        (
            best_gain,
            best_feature,
            best_threshold,
            left_mask,
            right_mask,
        ) = self._find_best_split(
            X,
            gradients,
            hessians
        )

        if (
            best_feature is None
            or best_gain <= 0
        ):

            return Node(
                value=self._leaf_value(
                    gradients,
                    hessians
                ),
                n_samples=n_samples,
            )

        left_child = self._build_tree(
            X[left_mask],
            gradients[left_mask],
            hessians[left_mask],
            depth + 1
        )

        right_child = self._build_tree(
            X[right_mask],
            gradients[right_mask],
            hessians[right_mask],
            depth + 1
        )

        return Node(
            feature_idx=best_feature,
            threshold=best_threshold,

            left=left_child,
            right=right_child,

            gain=best_gain,
            n_samples=n_samples,
        )

    def _predict_one(self, x, node):

        if node.is_leaf:
            return node.value

        if x[node.feature_idx] < node.threshold:
            return self._predict_one(
                x,
                node.left
            )

        return self._predict_one(
            x,
            node.right
        )

    def predict(self, X):

        X = np.asarray(X, dtype=float)

        return np.array([
            self._predict_one(x, self.root)
            for x in X
        ])
```

That is already a legitimate simplified XGBoost tree.

---

# 27. Test the Tree Before Building XGBoost

Do not immediately wrap this in boosting.

First test the tree in isolation.

Suppose:

```python
X = np.array([
    [1.0],
    [2.0],
    [3.0],
    [8.0],
    [9.0],
    [10.0],
])
```

Imagine some loss function gives:

```python
g = np.array([
     1.0,
     1.2,
     0.8,
    -1.0,
    -1.1,
    -0.9,
])

h = np.ones(6)
```

Notice the structure:

```text
left observations → positive gradients
right observations → negative gradients
```

The tree should discover that separating the two groups produces a useful split.

Run:

```python
tree = DecisionTree(
    max_depth=1,
    reg_lambda=1.0
)

tree.fit(X, g, h)
```

Inspect:

```python
tree.root.feature_idx
tree.root.threshold
tree.root.gain
```

You should expect the threshold somewhere between:

$$
3
$$

and

$$
8
$$

most naturally:

$$
5.5
$$

---

# 28. Manually Verify One Split

This is worth doing once.

For threshold

$$
x < 5.5
$$

the left gradients are:

$$
1.0,\ 1.2,\ 0.8
$$

so

$$
G_L=3
$$

and

$$
H_L=3
$$

The right gradients are:

$$
-1.0,\ -1.1,\ -0.9
$$

so

$$
G_R=-3
$$

and

$$
H_R=3
$$

Assume:

$$
\lambda=1
$$

Then:

$$
G=0,\qquad H=6
$$

Therefore:

$$
Gain
=
\frac12
\left[
\frac{3^2}{3+1}
+
\frac{(-3)^2}{3+1}
-
\frac{0^2}{6+1}
\right]
$$

$$
=
\frac12
\left[
\frac94+\frac94
\right]
$$

$$
=2.25
$$

The leaf outputs are:

$$
w_L
=
-\frac3{4}
=
-0.75
$$

and

$$
w_R
=
-\frac{-3}{4}
=
0.75
$$

Thus the tree learns:

```text
x < 5.5
    prediction correction = -0.75

x >= 5.5
    prediction correction = +0.75
```

That is the mechanics of XGBoost made concrete.

---

# 29. Add a Tree Printer

You will thank yourself later for having this.

```python
def print_tree(self, node=None, depth=0):

    if node is None:
        node = self.root

    indent = "    " * depth

    if node.is_leaf:

        print(
            f"{indent}Leaf("
            f"value={node.value:.4f}, "
            f"n={node.n_samples})"
        )

        return

    print(
        f"{indent}"
        f"X[{node.feature_idx}] "
        f"< {node.threshold:.4f} "
        f"(gain={node.gain:.4f})"
    )

    self.print_tree(
        node.left,
        depth + 1
    )

    self.print_tree(
        node.right,
        depth + 1
    )
```

Then:

```python
tree.print_tree()
```

might produce:

```text
X[0] < 5.5000 (gain=2.2500)
    Leaf(value=-0.7500, n=3)
    Leaf(value=0.7500, n=3)
```

This is invaluable when debugging.

---

# 30. Validation Methods You Should Add

Your implementation should fail loudly if its mathematical assumptions are violated.

For example:

```python
if np.any(hessians < 0):
    raise ValueError(
        "Hessians must be non-negative "
        "for this tree implementation."
    )
```

For logistic loss and squared-error loss, Hessians are non-negative.

You should also check:

```python
if not np.isfinite(X).all():
    raise ValueError(
        "NaN/inf values are not supported yet."
    )
```

Why explicitly reject missing values?

Because real XGBoost has sophisticated default-direction handling for missing values.

Your simple implementation does not.

Silently treating `NaN` as ordinary numbers would make your split behaviour difficult to reason about.

---

# 31. Do Not Add Missing-Value Logic Yet

Production XGBoost learns a default branch for missing values.

Conceptually, for every split it can test:

```text
missing → left
```

versus

```text
missing → right
```

and choose whichever produces higher gain.

That is worth implementing eventually.

But not yet.

Your progression should be:

```text
dense numeric data
        ↓
correct tree
        ↓
correct boosting
        ↓
efficient split search
        ↓
missing values
        ↓
sampling / column subsampling
```

Do not reproduce the entire XGBoost library before verifying the essential algorithm.

---

# 32. One Important Architectural Improvement: Pass Indices

Your current recursion does this:

```python
X[left_mask]
```

That creates new arrays.

For a small educational model, this is excellent because it is transparent.

But the tree structure itself does not require copying the data.

An improved design keeps the original arrays and passes only sample indices:

```python
indices = np.arange(len(X))
```

Then each node receives:

```python
indices[left_mask]
```

instead of creating:

```python
X[left_mask]
```

Conceptually:

```text
Original X remains fixed
        ↓
Node stores/receives row indices
        ↓
Children receive subsets of indices
```

This is closer to how serious tree implementations are organised.

But I recommend implementing the simple copied-array version first.

---

# 33. Why This Brute-Force Algorithm Is Slow

Suppose:

- \(n\) samples,
- \(d\) features,
- up to \(n\) thresholds per feature.

Your current algorithm roughly performs:

$$
O(dn^2)
$$

work per node because for every threshold you scan the observations again to construct masks and compute sums.

This becomes expensive quickly.

A much better algorithm sorts each feature once.

---

# 34. The Key Optimisation: Sorted Prefix Sums

Suppose one feature is sorted:

```text
x:
1   2   3   8   9   10
```

with gradients:

```text
g:
1.0 1.2 0.8 -1.0 -1.1 -0.9
```

Instead of recomputing left sums for every threshold, maintain running sums:

$$
G_L
$$

and

$$
H_L
$$

As you move the threshold right:

```text
after sample 1:
G_L = 1.0

after sample 2:
G_L = 2.2

after sample 3:
G_L = 3.0
```

The right side follows immediately:

$$
G_R=G-G_L
$$

$$
H_R=H-H_L
$$

This reduces split evaluation dramatically.

---

# 35. Efficient Split Search Skeleton

Later, `_find_best_split()` should evolve towards:

```python
for feature_idx in range(n_features):

    order = np.argsort(X[:, feature_idx])

    x_sorted = X[order, feature_idx]
    g_sorted = gradients[order]
    h_sorted = hessians[order]

    G = g_sorted.sum()
    H = h_sorted.sum()

    G_L = 0.0
    H_L = 0.0

    for i in range(n_samples - 1):

        G_L += g_sorted[i]
        H_L += h_sorted[i]

        if x_sorted[i] == x_sorted[i + 1]:
            continue

        G_R = G - G_L
        H_R = H - H_L

        threshold = (
            x_sorted[i]
            + x_sorted[i + 1]
        ) / 2

        ...
```

This is the next major implementation milestone after correctness.

Do not optimise further until this version works.

---

# 36. Why Binary Search Trees Do Not Solve This

A conceptual trap is:

> Decision trees are trees, so perhaps a binary search tree can speed up training.

Not really.

A binary search tree optimises:

$$
\text{search for a key}
$$

based on key ordering.

But XGBoost training needs to solve:

$$
\arg\max_{j,t}
\operatorname{Gain}(j,t)
$$

over:

- every feature \(j\),
- every candidate threshold \(t\).

The expensive operation is not tree traversal.

It is **searching for the best split**.

So the relevant optimisation structures are:

```text
sorting
prefix sums
histograms
quantile sketches
feature bins
```

not ordinary binary search trees.

During prediction, your learned decision tree already behaves like a binary tree:

$$
O(\text{depth})
$$

per observation.

---

# 37. How This Connects to the XGBoost Class

Once your tree works, the outer booster becomes surprisingly small.

For binary classification:

```text
initial logits
      ↓
for each boosting round:
      ↓
compute probabilities
      ↓
compute gradients + Hessians
      ↓
fit DecisionTree(X, g, h)
      ↓
predict correction
      ↓
add learning_rate × correction
```

For logistic loss:

$$
p_i=\sigma(\hat y_i)
$$

$$
g_i=p_i-y_i
$$

$$
h_i=p_i(1-p_i)
$$

Then:

```python
tree.fit(X, gradients, hessians)
```

and:

```python
raw_predictions += (
    learning_rate
    * tree.predict(X)
)
```

The tree you built here is therefore exactly the engine used by the boosting loop.

---

# 38. Your Fraud-Detection Context

For fraud detection:

$$
y_i\in\{0,1\}
$$

and severe class imbalance means the model may initially see overwhelmingly legitimate observations.

For logistic loss:

$$
g_i=p_i-y_i
$$

Consider an initial prediction near:

$$
p=0.5
$$

For legitimate transactions:

$$
y=0
\Rightarrow
g=0.5
$$

For fraud:

$$
y=1
\Rightarrow
g=-0.5
$$

Therefore fraud observations exert gradient pressure in the opposite direction.

A useful split will often isolate regions where the aggregated gradient behaviour differs strongly.

That is precisely what your gain formula detects.

Later you can introduce sample weighting or `scale_pos_weight`, but your base tree should work correctly without it first.

---

# 39. Recommended Class Architecture

By the time your implementation matures, I would organise it roughly as:

```text
Node
│
├── is_leaf

DecisionTree
│
├── fit
├── predict
│
├── _build_tree
├── _find_best_split
├── _candidate_thresholds
├── _split
├── _gain
├── _leaf_value
├── _predict_one
└── print_tree

BinaryLogLoss
│
├── gradient
├── hessian
└── probability

XGBoostClassifier
│
├── fit
├── predict_proba
└── predict
```

Notice the separation of responsibilities:

```text
Loss
    knows calculus

DecisionTree
    knows optimisation over splits

Node
    stores tree structure

XGBoostClassifier
    knows boosting
```

This separation will make your implementation much easier to reason about.

---

# 40. Recommended Implementation Order

Build it in this exact order.

### Stage 1 — Node

Implement:

```python
Node
Node.is_leaf
```

Manually construct:

```text
root
├── leaf
└── leaf
```

and verify traversal.

---

### Stage 2 — Leaf optimisation

Implement:

```python
_leaf_value()
```

Verify manually:

$$
-\frac{G}{H+\lambda}
$$

---

### Stage 3 — Gain

Implement:

```python
_gain()
```

Use a tiny hand-calculated example and confirm the numerical result.

Do not proceed until it matches.

---

### Stage 4 — One split

Implement:

```python
_candidate_thresholds()
_split()
_find_best_split()
```

Test on one feature.

Inspect:

```python
feature
threshold
gain
```

---

### Stage 5 — Recursive construction

Implement:

```python
_build_tree()
```

Start with:

```python
max_depth=1
```

Then try:

```python
max_depth=2
```

Print the structure after every test.

---

### Stage 6 — Prediction

Implement:

```python
_predict_one()
predict()
```

Manually verify which leaf several observations reach.

---

### Stage 7 — Logistic loss

Implement separately:

```python
sigmoid()
gradient()
hessian()
```

Do not put this logic inside `DecisionTree`.

---

### Stage 8 — Boosting

Finally build:

```python
XGBoostClassifier
```

as a list:

```python
self.trees = []
```

Each round creates another `DecisionTree`.

---

# 41. The Mental Model to Keep

Strip away the library terminology and your implementation is doing this:

$$
\boxed{
\text{Current model makes errors}
}
$$

The loss converts those errors into:

$$
\boxed{
g_i,\ h_i
}
$$

The tree asks:

$$
\boxed{
\text{Which partition lets me reduce that loss most?}
}
$$

The split gain answers that question.

The leaves then compute:

$$
\boxed{
w_j=-\frac{G_j}{H_j+\lambda}
}
$$

which tells each region how much to correct the model.

Then the booster adds:

$$
\boxed{
\eta f_t(x)
}
$$

to the current prediction.

So the full chain is:

$$
\text{prediction}
\rightarrow
\text{loss}
\rightarrow
(g,h)
\rightarrow
\text{split gain}
\rightarrow
\text{tree}
\rightarrow
\text{leaf corrections}
\rightarrow
\text{new prediction}
$$

That is XGBoost.

Your `Node` is simply the stored representation of the chosen partition.

Your `DecisionTree` is the optimisation algorithm that discovers that partition.

And your future `XGBoostClassifier` will repeatedly ask these trees to correct the mistakes of everything that came before.
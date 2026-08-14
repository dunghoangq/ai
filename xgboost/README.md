# XGBoost from scratch

## Reference

- XGBoost: A Scalable Tree Boosting System (https://arxiv.org/pdf/1603.02754)


## Build XGBoost from Scratch

### Sequence

```text
NumPy
│
├── Node
├── QuantileBinner
├── Histogram
├── SplitFinder
├── HistTree
└── XGBoostClassifier
```

We want best split:

$$
\arg\max_{i,s}\text{Gain}(j,s)
$$

Searching every sample is expensive -> So: $\text{sort} \rightarrow \text{scan}$.

Too many unique values remain -> $\text{quantiles} \rightarrow \text{bins}$.

Once have bins, individual values aren't needed for split evaluation -> $\text{samples} \rightarrow \text{gradient histograms}$

Candidate boundaries require cumulative left statistics. -> $\text{histograms} \rightarrow \text{prefix sums}$

Sparse data wastes memory and computation. -> $\text{dense matrix} \rightarrow \text{sparse storage}$

Growing several candidate leaves requires choosing what to expand. -> $\text{candidate leaves} \rightarrow \text{heap}$

Large memory access becomes expensive. -> $\text{objects} \rightarrow \text{contiguous compressed arrays}$

Large datasets exceed one CPU or machine. Since gradient statistics are additive -> $\text{local histogram} \rightarrow \text{parallel reduction}$

### Stage 1 — Naïve tree

NumPy only.
Implement:

```python
class Node
class DecisionTree
```

At every node:

```text
for feature:
    sort feature
    for threshold:
        calculate gain
```

### Stage 2 — Pre-sorted columns

Create:

```python
sorted_indices[j]
```

### Stage 3 — Sparse matrix

Implement a tiny CSR class yourself:

```python
class CSRMatrix:
    data
    indices
    indptr
```

Implement:

```python
get_row(i)
```

Then implement CSC.

### Stage 4 — Quantisation

Implement:

```python
class QuantileBinner:
    cuts_
    bin_ids_
```

Start with `np.quantile`.
No sketch yet.

### Stage 5 — Gradient histogram

Implement:

```python
class GradientHistogram:
    G
    H
```

and

```python
build_histogram(bin_ids, gradients, hessians)
```

This is the milestone where your implementation begins to resemble modern boosted-tree systems.

### Stage 6 — Histogram split finder

Implement:

```python
find_best_split(histogram)
```

using running prefix sums.

You should never rescan individual training samples while evaluating each bin boundary.

### Stage 7 — Node-position array

Maintain:

```python
position[i]
```

to record where every training row currently lives.

Use it to build child histograms.

### Stage 8 — Histogram subtraction

Implement

$$
H_R = H_P - H_L.
$$

Verify numerically that explicitly building the right histogram produces exactly the same result.

This small exercise teaches you more about production optimisation than another hundred lines of boosting maths.

### Stage 9 — Heap-based tree growth

Implement: `heapq` around candidate leaves.

Store conceptually: `(-gain, node_id)` because Python's `heapq` is a min-heap.

Now implement best-first tree growth.

### Stage 10 — Missing values
Skip missing entries during histogram construction.
For each split, evaluate:

$$
\text{missing} \rightarrow L
$$

and

$$
\text{missing} \rightarrow R
$$

Store the winning default direction inside the node.

### Stage 11 — Full boosting

Only now connect your optimised tree builder to:

$$
g_i = \frac{\partial L}{\partial \hat{y}_i}, \qquad
h_i = \frac{\partial^2 L}{\partial \hat{y}_i^2}
$$

build:

```python
class XGBoost:
    trees_
    learning_rate
```

with:

```text
initial prediction
      ↓
gradients/Hessians
      ↓
histogram tree
      ↓
update predictions
      ↓
repeat
```

### Stage 12 — Quantile sketch
Finally remove your dependency on exact `np.quantile`.
Build a simple streaming quantile summary.
Only after you understand that should you study the weighted quantile sketch from the XGBoost paper.
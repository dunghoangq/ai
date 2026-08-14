# XGBoost - Data Structures

## What we need?

A data structure for XGBoost must have three things:

1. Find candidate thresholds
2. Move samples left and right
3. Maintain sums of gradients and hessians.

## Data structures

- Array to store gradients and hessians
- Tree node for the splits
- Numerical features are sorted and stored with row id - (value, row_id)
- **Column-major** instead of row-major.

### Sparse matrix representation

For One-hot encoding, storing alot of 0's is wasting.

$$
[\text{row_id}, \text{col_id}, \text{val}]
$$

e.g.,

$$
X = \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 1 & 0 & 0
\end{bmatrix}
$$

becomes

$$
\begin{bmatrix}
    0 & 0 & 1 \\
    1 & 2 & 1 \\
    2 & 1 & 1
\end{bmatrix}
$$

### Compressed Sparse Row (CSR)

Only store values != 0. Maintain three arrays

- `values`: stores non-zero values, left2right, top2bottom
- `indices`: column indices of each value.
- `indptr`: start indices of each row


### Compressed Sparse Column (CSC)

Tranposed version of CSR, stores:

- values
- row indices
- column pointers

## Summary

| Structure | Why it exists | Main operation |
|---|---|---|
| Array | gradients, Hessians, predictions | \(O(1)\) indexing |
| Node array/vector | store trained tree | fast traversal |
| Sorted feature column | exact split search | sequential scan |
| CSR | sparse row access | iterate row features |
| CSC / column structure | feature-oriented access | iterate feature values |
| Quantile sketch | approximate distribution | estimate cut points |
| Bin array | compressed feature values | lookup bin |
| Histogram | aggregate \(G,H\) | split evaluation |
| Prefix sums | cumulative \(G,H\) | evaluate boundaries |
| Node-position array | track sample membership | route rows |
| Bitset/mask | compact Boolean state | membership/filtering |
| Stack | depth-first growth | push/pop |
| Queue | breadth-first growth | enqueue/dequeue |
| Heap | best-first growth | extract max gain |
| Block/page | cache/disk efficiency | sequential processing |
| Shard | distributed dataset | parallel aggregation |
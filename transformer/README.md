# Understanding Transformer

## 1. Roadmap

### 1.0. Milestone 0: Prerequisite check

You can already do backprop and optimisers, so this is a test, not a lesson. You should be able to explain softmax and its gradient, cross-entropy as negative log-likelihood, why layer normalisation exists, and matrix shapes under batching (B, T, C).
Done when: you can derive the gradient of softmax followed by cross-entropy on paper.

### 1.1. Milestone 1: The task, not the architecture

Language modelling is just predicting the next token from the previous ones. Build a character-level bigram model in PyTorch with an embedding table, a loss and a sampling loop.
Done when: it trains, generates gibberish, and you can explain why its loss has a floor (it only sees one character of context).

### 1.2. Milestone 2: Attention in isolation (the core)

Build it in stages, as a ladder:

1. Each token averages all previous tokens, using a lower-triangular matrix.
2. Replace the uniform weights with learned, data-dependent weights, which gives you Q, K and V.
3. Add the scaling by √d_k. Find out why it’s there: look at the variance of a dot product of random vectors.
4. Add the causal mask and explain why it is needed during training.

Implement a single head in NumPy with explicit shapes, then derive and code its backward pass and check it numerically.
Done when: your gradient check passes, and you can explain what happens to the softmax if you omit the scaling at large d_k.

### 1.3. Milestone 3: The full block

Add multi-head attention (heads as parallel subspaces, concatenated and projected), the position-wise MLP, residual connections and LayerNorm. Then add positional information. Attention alone is permutation-equivariant, so without it the model sees a bag of tokens. Stack blocks into a decoder-only GPT on tiny Shakespeare.
Done when: it trains to a sensible loss (roughly 1.5 nats per character is a reasonable target for a small model), and you can state every tensor’s shape from memory.

### 1.4. Milestone 4: Ablations (where understanding becomes durable)

Remove one thing at a time and predict the outcome before running: the scaling, the residuals, LayerNorm, positional encoding, the causal mask, and one head versus many. A wrong prediction is the most useful thing that can happen here.
Done when: you can explain each failure mechanistically.

### 1.5. Milestone 5: Training dynamics

Cover AdamW, learning-rate warmup and decay, initialisation, gradient clipping, and mixed precision. Your optimiser curriculum pays off here.
Done when: you can explain why warmup helps and show a run that diverges without it.

### 1.6. Milestone 6: The original architecture

Milestone 6: The original architecture

Now read Attention Is All You Need. It will feel like a different, easier paper. Build the encoder-decoder variant, which adds cross-attention (queries from one sequence, keys and values from another). Test it on a toy task such as sequence reversal.
Done when: you can say precisely how encoder-only, decoder-only and encoder-decoder differ, and what each is suited to.

### 1.7. Milestone 7: Modern upgrades

Convert your model to a Llama-style one: RMSNorm, pre-norm, RoPE (rotary position embeddings), SwiGLU, and a KV cache for generation. Understand the idea behind FlashAttention and grouped-query attention, and why attention costs O(T²) in sequence length. That cost matters a great deal for long inputs.
Done when: your upgraded model matches or beats the baseline and generates faster with the cache.

### 1.8. Milestone 8: Beyond discrete tokens

Build a small Vision Transformer: patches become tokens via a linear projection. This teaches you that a “token” is just a vector, which is the key to using transformers on non-text data.
Done when: you can explain what changes and what doesn’t when inputs are continuous.

### 1.9. Milestone 9: Return to the paper

Reread it. You should be able to identify which parts are standard transformer machinery and which are genuinely novel. Only then does the novelty become visible.

## 2. Resources

- 3Blue1Brown’s transformer chapters for geometric intuition.
- Karpathy’s “Let’s build GPT” and nanoGPT, as a guide to follow, not a thing to copy.
- The Annotated Transformer (Harvard NLP) for the original architecture.
- The original paper, but only at Milestone 6.

## 3. Exercise

Without looking at anything, write single-head causal self-attention in NumPy for X of shape (T=4, d_model=8) with d_k = 4. Write the shape of every intermediate (Q, K, V, scores, weights, output).

Then answer, in two sentences: if you shuffle the order of the four input tokens and have no positional encoding, what happens to the output rows? Does your answer change once the causal mask is on? (That second part is trickier than it looks.)
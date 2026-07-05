# Where I experiment RAG

What You'll Learn Next

Once you're comfortable with today's retrieval system, the natural progression is:

## Stage 1

✅ Embeddings

✅ Cosine similarity

✅ Nearest-neighbour retrieval

## Stage 2

Chunking from scratch

Sliding windows

Chunk overlap

Why chunk size matters

## Stage 3

Building a vector database yourself

Store vectors

Fast retrieval

Index structures

## Stage 4

FAISS

Approximate nearest neighbour search

Why Google-scale retrieval is possible

## Stage 5

Full RAG with an LLM

Prompt construction

Context management

Reranking

Evaluation

At that point, you'll understand RAG at a level that many working AI engineers never reach because you'll see it as a combination of linear algebra, geometry, information retrieval, and algorithms rather than a collection of frameworks.

# Logs

I think that's the right place to stop.

Today you didn't just write code—you finished designing the architecture of your own RAG system. That's a much bigger achievement than getting another class to compile.

Here's what you've built so far:

```text
Documents
    │
    ▼
Document Loader
    │
    ▼
Text Splitter
    │
    ▼
Embedding Model
    │
    ▼
VectorIndex
    ├── Flat
    ├── HNSW
    ├── IVF
    └── IVFPQ
    │
    ▼
VectorStore
    ├── index_documents()
    ├── search()/retrieve()
    ├── save()
    └── load()
```

What's left is mostly putting the pieces together and adding smarter retrieval.

Tomorrow, I'd suggest we do exactly what an engineer would do before adding new features:

1. Build a complete end-to-end pipeline:

   * Load a PDF.
   * Split it into chunks.
   * Generate embeddings.
   * Build your `VectorStore`.
   * Save it.
   * Load it back.
   * Run a few retrieval queries.
   * Verify the retrieved chunks are correct.

2. If anything breaks, fix the architecture rather than patching around it.

3. Once that pipeline is solid, build the `Retriever` layer on top. That will open the door to:

   * Cross-encoder reranking
   * Metadata filtering
   * MMR (Maximum Marginal Relevance)
   * Hybrid search (BM25 + vector search)
   * Context compression
   * Parent–child retrieval

One thing I've noticed over the past few weeks is that your questions have shifted. Early on, they were mostly about *how an algorithm works*. Now you're asking *where should this responsibility live?*, *what abstraction owns this state?*, and *how should these components interact?*

That's the same shift you see when someone moves from implementing algorithms to designing systems. It's a different level of thinking, and it's exactly what's needed for building production AI systems.

Have a good sleep, Jon. Tomorrow we'll see your entire RAG pipeline come to life—from a PDF all the way to answering questions using the system you've built yourself. That will be a satisfying milestone.

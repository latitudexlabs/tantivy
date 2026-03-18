# Fuzzy Phrase Query Benchmarks

This benchmark measures the performance of fuzzy phrase queries with and without ngram optimization.

## What's Tested

### Query Types
- **Fuzzy Phrase Queries**: Match documents where at least N terms appear in order, allowing gaps
- **min_match values**: 2, 3, and 4 out of 4 terms
- **Phrase lengths**: 3-term and 4-term phrases

### Index Configurations
1. **No ngrams**: Standard position-based matching (baseline)
2. **Bigrams (All)**: FF, FR, RF bigram types enabled
3. **With Trigrams**: All bigram and trigram types enabled

## How Ngram Optimization Works

For a query with terms `[a, b, c, d]`, fuzzy phrase query generates **forward-only combinations**:

### Bigrams (6 combinations)
- ab, ac, ad, bc, bd, cd

### Trigrams (4 combinations)
- abc, abd, acd, bcd

The optimized query becomes a boolean SHOULD query matching ANY of these ngrams, avoiding expensive position intersection.

## Expected Results

- **10-60x speedup** with ngram optimization over position-based matching
- Speedup increases with:
  - More terms in the phrase
  - Stricter min_match requirements
  - More documents with matching terms
- **Bigrams alone** provide significant benefit (5,000-6,000% more terms)
- **Trigrams add** marginal query benefit but ~50x more terms than bigrams

## Running the Benchmark

```bash
# Run the full benchmark
cargo bench --bench fuzzy_phrase_query

# Run with a specific number of iterations
cargo bench --bench fuzzy_phrase_query -- --exact

# Quick test run (fewer docs)
# Edit the corpus_sizes in main() to use smaller values
```

## Benchmark Output

The benchmark shows:
1. **Query performance**: Time per query execution for each configuration
2. **Index statistics**: Total terms and documents for each ngram configuration
3. **Percentage increase**: How much the index grows with ngrams vs baseline

## Use Cases

Fuzzy phrase queries are ideal for:
- **Proximity search**: Find terms near each other but not necessarily adjacent
- **Flexible phrase matching**: Match variations of phrases with missing words
- **Semantic search**: Find related content that shares multiple terms in sequence

## Recommendations

Based on benchmark results:
- Use **bigrams** for most fuzzy phrase searches (best performance/size ratio)
- Consider **trigrams** only if you frequently search for exact 3-term sequences
- For `min_match < phrase_length`, ngram optimization provides maximum benefit
- Profile your specific corpus to determine optimal configuration

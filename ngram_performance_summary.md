# Ngram vs Simple Phrase Query Performance Summary

Results from `cargo bench --bench ngram_phrase_query -- "Small"` (10K documents)

## Performance Comparison

### 2-Word Phrases: **13-67x faster** with bigrams

| Query | Simple (no ngrams) | Bigrams | Speedup |
|-------|-------------------|---------|---------|
| "the quick" | 0.186ms | 0.015ms | **13x** |
| "in the" | 0.824ms | 0.012ms | **67x** |
| "telescope and" | 0.421ms | 0.010ms | **42x** |
| "world of" | 0.659ms | 0.011ms | **58x** |

### 3-Word Phrases: **11-39x faster** with bigrams

| Query | Simple (no ngrams) | Bigrams | Speedup |
|-------|-------------------|---------|---------|
| "the quick brown" | 0.288ms | 0.027ms | **11x** |
| "in the world" | 0.870ms | 0.022ms | **39x** |
| "telescope and nebula" | 0.357ms | 0.018ms | **20x** |

### 4-Word Phrases: **24-27x faster** with bigrams

| Query | Simple (no ngrams) | Bigrams | Speedup |
|-------|-------------------|---------|---------|
| "in the world of" | 0.924ms | 0.034ms | **27x** |

## Index Size Impact

| Configuration | Terms | Size Increase |
|--------------|-------|---------------|
| No ngrams (baseline) | 60 | — |
| Bigrams | 3,490 | +5,717% |
| With trigrams | 188,301 | +313,735% (54x larger than bigrams) |

## Key Findings

### ✅ Bigrams: Excellent Performance/Size Trade-off

- **Consistent speedups**: 10-67x faster across all phrase lengths
- **Best for frequent bigrams**: "in the" shows 67x improvement
- **Even rare combinations benefit**: 20-40x speedups
- **Index cost is reasonable**: ~5,700% term increase is worth the performance gain

### ❌ Trigrams: Worse Than Bigrams

Performance comparison (bigrams vs trigrams):

| Phrase Length | Bigrams | Trigrams | Result |
|--------------|---------|----------|--------|
| 3-word | 0.027ms | 0.046ms | **1.7x slower** |
| 4-word | 0.034ms | 0.058ms | **1.7x slower** |

- Trigrams add query planning overhead
- No measurable benefit over bigrams + position matching
- Massive index explosion (54x) is not justified

## Recommendation

**For phrase query workloads, always use bigrams:**

```rust
WordNgramSet::new()
    .with_ngram_ff()  // Frequent-Frequent
    .with_ngram_fr()  // Frequent-Rare
    .with_ngram_rf()  // Rare-Frequent
```

- **10-67x performance improvement** over simple phrase queries
- **Moderate index cost** (~5,700% term growth)
- **Avoid trigrams** - they provide negative value

## Detailed Results

```
2-word phrases — Small (10K docs)
the_quick_no_ngrams                Avg: 0.1863ms    Output: 10    
the_quick_bigrams_ff               Avg: 0.0145ms    Output: 10    
the_quick_bigrams_all              Avg: 0.0146ms    Output: 10    
the_quick_with_trigrams            Avg: 0.0143ms    Output: 10    

in_the_no_ngrams                   Avg: 0.8240ms    Output: 10    
in_the_bigrams_ff                  Avg: 0.0122ms    Output: 10    
in_the_bigrams_all                 Avg: 0.0123ms    Output: 10    
in_the_with_trigrams               Avg: 0.0116ms    Output: 10    

telescope_and_no_ngrams            Avg: 0.4214ms    Output: 10    
telescope_and_bigrams_ff           Avg: 0.0104ms    Output: 10    
telescope_and_bigrams_all          Avg: 0.0100ms    Output: 10    
telescope_and_with_trigrams        Avg: 0.0089ms    Output: 10    

world_of_no_ngrams                 Avg: 0.6594ms    Output: 10    
world_of_bigrams_ff                Avg: 0.0113ms    Output: 10    
world_of_bigrams_all               Avg: 0.0113ms    Output: 10    
world_of_with_trigrams             Avg: 0.0110ms    Output: 10    

3-word phrases — Small (10K docs)
the_quick_brown_no_ngrams          Avg: 0.2880ms    Output: 10    
the_quick_brown_bigrams_ff         Avg: 0.0301ms    Output: 10    
the_quick_brown_bigrams_all        Avg: 0.0269ms    Output: 10    
the_quick_brown_with_trigrams      Avg: 0.0460ms    Output: 10    

in_the_world_no_ngrams             Avg: 0.8699ms    Output: 10    
in_the_world_bigrams_ff            Avg: 0.0235ms    Output: 10    
in_the_world_bigrams_all           Avg: 0.0223ms    Output: 10    
in_the_world_with_trigrams         Avg: 0.0373ms    Output: 10    

telescope_and_nebula_no_ngrams     Avg: 0.3571ms    Output: 10    
telescope_and_nebula_bigrams_ff    Avg: 0.0181ms    Output: 10    
telescope_and_nebula_bigrams_all   Avg: 0.0175ms    Output: 10    
telescope_and_nebula_with_trigrams Avg: 0.0248ms    Output: 10    

4-word phrases — Small (10K docs)
in_the_world_of_no_ngrams          Avg: 0.9239ms    Output: 10    
in_the_world_of_bigrams_ff         Avg: 0.0337ms    Output: 10    
in_the_world_of_bigrams_all        Avg: 0.0387ms    Output: 10    
in_the_world_of_with_trigrams      Avg: 0.0580ms    Output: 10    
```

## Index Statistics

```
Small (10K docs) - Index Statistics:
  No ngrams:      60 terms, 10000 docs
  Bigrams (FF):   3490 terms, 10000 docs (+5716.7% terms)
  Bigrams (All):  3490 terms, 10000 docs (+5716.7% terms)
  With trigrams:  188301 terms, 10000 docs (+313735.0% terms)

Medium (50K docs) - Index Statistics:
  No ngrams:      60 terms, 50000 docs
  Bigrams (FF):   3490 terms, 50000 docs (+5716.7% terms)
  Bigrams (All):  3490 terms, 50000 docs (+5716.7% terms)
  With trigrams:  201467 terms, 50000 docs (+335678.3% terms)

Large (100K docs) - Index Statistics:
  No ngrams:      60 terms, 100000 docs
  Bigrams (FF):   3490 terms, 100000 docs (+5716.7% terms)
  Bigrams (All):  3490 terms, 100000 docs (+5716.7% terms)
  With trigrams:  202049 terms, 100000 docs (+336648.3% terms)
```

---

*Generated: February 5, 2026*  
*See [ngram_phrase_query_README.md](benches/ngram_phrase_query_README.md) for more details*

// Benchmarks fuzzy phrase queries with and without word ngram indexing.
//
// What's measured:
// - Fuzzy phrase query performance with standard position-based matching
// - Fuzzy phrase query performance with ngram term optimization (forward-only combinations)
// - Different min_match values (2, 3, 4 out of 4 terms)
// - Different ngram configurations (bigrams, trigrams, combinations, edge ngrams)
// - Impact on query performance vs index size
//
// Key Differences from Regular Phrase Query:
// - Fuzzy phrase matches documents with terms in order but allows gaps
// - Generates all forward-only ngram combinations (6 bigrams + 4 trigrams for 4 terms)
// - Uses SHOULD (OR) boolean query instead of MUST (AND)
// - More expensive without ngrams due to position intersection checking
//
// Expected Results:
// - Ngram optimization provides 10-60x speedup by avoiding position matching
// - Speedup increases with more terms and stricter min_match requirements
// - Bigrams alone provide significant benefit for most use cases
// - Trigrams add marginal benefit but increase index size substantially
// - Edge ngrams enable prefix matching but significantly increase index size

use binggan::{black_box, BenchGroup, BenchRunner, PeakMemAlloc, INSTRUMENTED_SYSTEM};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tantivy::collector::TopDocs;
use tantivy::query::{FuzzyPhraseQuery, Query};
use tantivy::schema::{IndexRecordOption, Schema, TextFieldIndexing, TEXT};
use tantivy::{doc, Index, ReloadPolicy, Searcher, Term, WordNgramConfig, WordNgramSet};

#[global_allocator]
pub static GLOBAL: &PeakMemAlloc<std::alloc::System> = &INSTRUMENTED_SYSTEM;

#[derive(Clone)]
struct BenchIndex {
    #[allow(dead_code)]
    index: Index,
    searcher: Searcher,
    field: tantivy::schema::Field,
}

const FREQUENT_WORDS: &[&str] = &[
    "the", "a", "an", "in", "of", "to", "and", "for", "on", "with", "as", "at", "by", "from",
    "is", "was", "are", "were", "be", "been", "have", "has", "had", "do", "does", "did",
];

const RARE_WORDS: &[&str] = &[
    "elephant", "telescope", "algorithm", "symphony", "volcano", "cathedral", "molecule",
    "glacier", "orchestra", "nebula", "quantum", "crystal", "dinosaur", "emerald",
];

const MEDIUM_WORDS: &[&str] = &[
    "world", "people", "time", "day", "year", "way", "man", "work", "life", "hand",
    "part", "place", "case", "week", "company", "system", "program", "question",
];

/// Build an index with or without ngram indexing
fn build_index(num_docs: usize, ngram_config: Option<WordNgramSet>) -> BenchIndex {
    build_index_with_mode(num_docs, ngram_config, false)
}

/// Build an index with optional all_combinations mode
fn build_index_with_mode(num_docs: usize, ngram_config: Option<WordNgramSet>, all_combinations: bool) -> BenchIndex {
    let mut schema_builder = Schema::builder();

    let text_indexing = if let Some(ngram_set) = ngram_config {
        let config = if all_combinations {
            WordNgramConfig::builder()
                .ngram_types(ngram_set)
                .all_combinations(true)
                .build()
        } else {
            WordNgramConfig::new(ngram_set.bits())
        };
        TextFieldIndexing::default()
            .set_index_option(IndexRecordOption::WithFreqsAndPositions)
            .set_word_ngrams(config)
    } else {
        TextFieldIndexing::default().set_index_option(IndexRecordOption::WithFreqsAndPositions)
    };

    let text_field = schema_builder.add_text_field("text", TEXT.clone().set_indexing_options(text_indexing));
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);

    // Populate index with realistic documents containing phrases with gaps
    let mut rng = StdRng::from_seed([42u8; 32]);
    {
        let mut writer = index.writer_with_num_threads(1, 500_000_000).unwrap();
        
        for _ in 0..num_docs {
            let mut tokens = Vec::new();
            
            // Generate a document with 50-200 words
            let doc_length = rng.random_range(50..200);
            
            for _ in 0..doc_length {
                // 60% frequent, 30% medium, 10% rare
                let word_choice: f64 = rng.random();
                let word = if word_choice < 0.6 {
                    FREQUENT_WORDS[rng.random_range(0..FREQUENT_WORDS.len())]
                } else if word_choice < 0.9 {
                    MEDIUM_WORDS[rng.random_range(0..MEDIUM_WORDS.len())]
                } else {
                    RARE_WORDS[rng.random_range(0..RARE_WORDS.len())]
                };
                tokens.push(word);
            }
            
            // Inject specific phrases with gaps for fuzzy matching
            if rng.random_bool(0.3) {
                let pos = rng.random_range(0..tokens.len().saturating_sub(5));
                tokens[pos] = "the";
                tokens[pos + 2] = "quick"; // Gap!
                tokens[pos + 3] = "brown";
                tokens[pos + 5] = "fox";   // Gap!
            }
            if rng.random_bool(0.2) {
                let pos = rng.random_range(0..tokens.len().saturating_sub(6));
                tokens[pos] = "in";
                tokens[pos + 1] = "the";
                tokens[pos + 4] = "world"; // Gap!
                tokens[pos + 5] = "of";
            }
            if rng.random_bool(0.15) {
                let pos = rng.random_range(0..tokens.len().saturating_sub(4));
                tokens[pos] = "telescope";
                tokens[pos + 2] = "and";   // Gap!
                tokens[pos + 3] = "nebula";
            }
            
            writer.add_document(doc!(text_field => tokens.join(" "))).unwrap();
        }
        writer.commit().unwrap();
    }

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    let searcher = reader.searcher();

    BenchIndex {
        index,
        searcher,
        field: text_field,
    }
}

fn main() {
    let mut runner = BenchRunner::new();

    // Test different corpus sizes
    let corpus_sizes = vec![
        ("Small (10K docs)", 10_000),
        ("Medium (50K docs)", 50_000),
    ];

    for (size_label, num_docs) in corpus_sizes {
        // Build indices with different configurations
        let no_ngrams = build_index(num_docs, None);
        let bigrams_all = build_index(
            num_docs,
            Some(
                WordNgramSet::new()
                    .with_ngram_ff()
                    .with_ngram_fr()
                    .with_ngram_rf(),
            ),
        );
        let with_trigrams = build_index(
            num_docs,
            Some(
                WordNgramSet::new()
                    .with_ngram_ff()
                    .with_ngram_fr()
                    .with_ngram_rf()
                    .with_ngram_fff()
                    .with_ngram_ffr()
                    .with_ngram_frf()
                    .with_ngram_rff(),
            ),
        );
        let all_combinations = build_index_with_mode(
            num_docs,
            Some(
                WordNgramSet::new()
                    .with_ngram_ff()
                    .with_ngram_fr()
                    .with_ngram_rf(),
            ),
            true,
        );
        
        // Build index with edge ngrams enabled
        let edge_ngrams = {
            let mut schema_builder = Schema::builder();
            let config = WordNgramConfig::builder()
                .ngram_types(
                    WordNgramSet::new()
                        .with_ngram_ff()
                        .with_ngram_fr()
                        .with_ngram_rf(),
                )
                .edge_ngram(true)
                .min_edge_ngram(3)
                .build();
            let text_indexing = TextFieldIndexing::default()
                .set_index_option(IndexRecordOption::WithFreqsAndPositions)
                .set_word_ngrams(config);
            let text_field = schema_builder.add_text_field("text", TEXT.clone().set_indexing_options(text_indexing));
            let schema = schema_builder.build();
            let index = Index::create_in_ram(schema);

            let mut rng = StdRng::from_seed([42u8; 32]);
            {
                let mut writer = index.writer_with_num_threads(1, 500_000_000).unwrap();
                
                for _ in 0..num_docs {
                    let mut tokens = Vec::new();
                    let doc_length = rng.random_range(50..200);
                    
                    for _ in 0..doc_length {
                        let word_choice: f64 = rng.random();
                        let word = if word_choice < 0.6 {
                            FREQUENT_WORDS[rng.random_range(0..FREQUENT_WORDS.len())]
                        } else if word_choice < 0.9 {
                            MEDIUM_WORDS[rng.random_range(0..MEDIUM_WORDS.len())]
                        } else {
                            RARE_WORDS[rng.random_range(0..RARE_WORDS.len())]
                        };
                        tokens.push(word);
                    }
                    
                    if rng.random_bool(0.3) {
                        let pos = rng.random_range(0..tokens.len().saturating_sub(5));
                        tokens[pos] = "the";
                        tokens[pos + 2] = "quick";
                        tokens[pos + 3] = "brown";
                        tokens[pos + 5] = "fox";
                    }
                    if rng.random_bool(0.2) {
                        let pos = rng.random_range(0..tokens.len().saturating_sub(6));
                        tokens[pos] = "in";
                        tokens[pos + 1] = "the";
                        tokens[pos + 4] = "world";
                        tokens[pos + 5] = "of";
                    }
                    if rng.random_bool(0.15) {
                        let pos = rng.random_range(0..tokens.len().saturating_sub(4));
                        tokens[pos] = "telescope";
                        tokens[pos + 2] = "and";
                        tokens[pos + 3] = "nebula";
                    }
                    
                    writer.add_document(doc!(text_field => tokens.join(" "))).unwrap();
                }
                writer.commit().unwrap();
            }

            let reader = index
                .reader_builder()
                .reload_policy(ReloadPolicy::Manual)
                .try_into()
                .unwrap();
            let searcher = reader.searcher();

            BenchIndex {
                index,
                searcher,
                field: text_field,
            }
        };

        // Benchmark 4-term fuzzy phrases with different min_match values
        {
            let mut group = runner.new_group();
            group.set_name(format!("4-term fuzzy phrase (min_match=2) — {}", size_label));
            
            let phrases = vec![
                ("the quick brown fox", vec!["the", "quick", "brown", "fox"], 2),
                ("in the world of", vec!["in", "the", "world", "of"], 2),
            ];

            for (phrase_name, terms, min_match) in &phrases {
                add_fuzzy_phrase_bench(
                    &mut group,
                    &no_ngrams,
                    phrase_name,
                    terms,
                    *min_match,
                    "no_ngrams",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &bigrams_all,
                    phrase_name,
                    terms,
                    *min_match,
                    "bigrams_all",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &with_trigrams,
                    phrase_name,
                    terms,
                    *min_match,
                    "with_trigrams",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &all_combinations,
                    phrase_name,
                    terms,
                    *min_match,
                    "all_combinations",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &edge_ngrams,
                    phrase_name,
                    terms,
                    *min_match,
                    "edge_ngrams",
                );
            }
            group.run();
        }

        {
            let mut group = runner.new_group();
            group.set_name(format!("4-term fuzzy phrase (min_match=3) — {}", size_label));
            
            let phrases = vec![
                ("the quick brown fox", vec!["the", "quick", "brown", "fox"], 3),
                ("in the world of", vec!["in", "the", "world", "of"], 3),
            ];

            for (phrase_name, terms, min_match) in &phrases {
                add_fuzzy_phrase_bench(
                    &mut group,
                    &no_ngrams,
                    phrase_name,
                    terms,
                    *min_match,
                    "no_ngrams",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &bigrams_all,
                    phrase_name,
                    terms,
                    *min_match,
                    "bigrams_all",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &with_trigrams,
                    phrase_name,
                    terms,
                    *min_match,
                    "with_trigrams",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &all_combinations,
                    phrase_name,
                    terms,
                    *min_match,
                    "all_combinations",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &edge_ngrams,
                    phrase_name,
                    terms,
                    *min_match,
                    "edge_ngrams",
                );
            }
            group.run();
        }

        {
            let mut group = runner.new_group();
            group.set_name(format!("4-term fuzzy phrase (min_match=4) — {}", size_label));
            
            let phrases = vec![
                ("the quick brown fox", vec!["the", "quick", "brown", "fox"], 4),
                ("in the world of", vec!["in", "the", "world", "of"], 4),
            ];

            for (phrase_name, terms, min_match) in &phrases {
                add_fuzzy_phrase_bench(
                    &mut group,
                    &no_ngrams,
                    phrase_name,
                    terms,
                    *min_match,
                    "no_ngrams",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &bigrams_all,
                    phrase_name,
                    terms,
                    *min_match,
                    "bigrams_all",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &with_trigrams,
                    phrase_name,
                    terms,
                    *min_match,
                    "with_trigrams",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &all_combinations,
                    phrase_name,
                    terms,
                    *min_match,
                    "all_combinations",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &edge_ngrams,
                    phrase_name,
                    terms,
                    *min_match,
                    "edge_ngrams",
                );
            }
            group.run();
        }

        // Benchmark 3-term fuzzy phrases
        {
            let mut group = runner.new_group();
            group.set_name(format!("3-term fuzzy phrase (min_match=2) — {}", size_label));
            
            let phrases = vec![
                ("telescope and nebula", vec!["telescope", "and", "nebula"], 2),
                ("quick brown fox", vec!["quick", "brown", "fox"], 2),
            ];

            for (phrase_name, terms, min_match) in &phrases {
                add_fuzzy_phrase_bench(
                    &mut group,
                    &no_ngrams,
                    phrase_name,
                    terms,
                    *min_match,
                    "no_ngrams",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &bigrams_all,
                    phrase_name,
                    terms,
                    *min_match,
                    "bigrams_all",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &with_trigrams,
                    phrase_name,
                    terms,
                    *min_match,
                    "with_trigrams",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &all_combinations,
                    phrase_name,
                    terms,
                    *min_match,
                    "all_combinations",
                );
                add_fuzzy_phrase_bench(
                    &mut group,
                    &edge_ngrams,
                    phrase_name,
                    terms,
                    *min_match,
                    "edge_ngrams",
                );
            }
            group.run();
        }

        // Index statistics comparison
        println!("\n{} - Index Statistics:", size_label);
        let (terms_none, docs_none) = estimate_index_size(&no_ngrams);
        let (terms_all, docs_all) = estimate_index_size(&bigrams_all);
        let (terms_tri, docs_tri) = estimate_index_size(&with_trigrams);
        let (terms_combo, docs_combo) = estimate_index_size(&all_combinations);
        let (terms_edge, docs_edge) = estimate_index_size(&edge_ngrams);
        
        println!("  No ngrams:         {} terms, {} docs", terms_none, docs_none);
        println!("  Bigrams (All):     {} terms, {} docs (+{:.1}% terms)", 
                 terms_all, docs_all,
                 ((terms_all as f64 / terms_none as f64) - 1.0) * 100.0);
        println!("  With trigrams:     {} terms, {} docs (+{:.1}% terms)", 
                 terms_tri, docs_tri,
                 ((terms_tri as f64 / terms_none as f64) - 1.0) * 100.0);
        println!("  All combinations:  {} terms, {} docs (+{:.1}% terms)", 
                 terms_combo, docs_combo,
                 ((terms_combo as f64 / terms_none as f64) - 1.0) * 100.0);
        println!("  Edge ngrams:       {} terms, {} docs (+{:.1}% terms)", 
                 terms_edge, docs_edge,
                 ((terms_edge as f64 / terms_none as f64) - 1.0) * 100.0);
    }
}

fn add_fuzzy_phrase_bench(
    bench_group: &mut BenchGroup,
    bench_index: &BenchIndex,
    phrase_name: &str,
    terms: &[&str],
    min_match: usize,
    config_name: &str,
) {
    let task_name = format!("{}_{}", phrase_name.replace(" ", "_"), config_name);
    let phrase_terms: Vec<Term> = terms
        .iter()
        .map(|t| Term::from_field_text(bench_index.field, t))
        .collect();
    let query = FuzzyPhraseQuery::new(phrase_terms, min_match);
    let search_task = SearchTask {
        searcher: bench_index.searcher.clone(),
        query: Box::new(query),
    };
    bench_group.register(task_name, move |_| black_box(search_task.run()));
}

struct SearchTask {
    searcher: Searcher,
    query: Box<dyn Query>,
}

impl SearchTask {
    #[inline(never)]
    pub fn run(&self) -> usize {
        let result = self
            .searcher
            .search(&self.query, &TopDocs::with_limit(10).order_by_score())
            .unwrap();
        result.len()
    }
}

fn estimate_index_size(bench_index: &BenchIndex) -> (usize, usize) {
    // Get actual term count and document count
    let mut total_terms = 0;
    let mut total_docs = 0;
    
    for segment_reader in bench_index.searcher.segment_readers() {
        total_docs += segment_reader.num_docs() as usize;
        if let Ok(inverted_index) = segment_reader.inverted_index(bench_index.field) {
            let terms = inverted_index.terms();
            total_terms += terms.num_terms() as usize;
        }
    }
    (total_terms, total_docs)
}

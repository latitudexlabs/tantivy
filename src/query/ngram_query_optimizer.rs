/// Query optimization for leveraging word ngram indexes
///
/// This module provides query rewriting to use word ngram indexes when
/// available, speeding up phrase-like queries by reducing (phrase queries)
/// or avoiding (fuzzy phrase queries) position matching.
///
/// The indexer writes *every* bigram/trigram of the configured sizes (see
/// `PostingsWriter::index_text`): frequency-based ngram types only describe
/// which combinations a field intends to serve, they do not filter what gets
/// indexed. Query-side generation therefore only needs to mirror the
/// indexer's structural rules:
/// - consecutive mode indexes adjacent ngrams only; `all_combinations` mode
///   indexes every ordered combination within the configured window,
/// - bigrams of two identical words are never indexed,
/// - trigrams of three identical words are never indexed.
use crate::query::fuzzy_query::FuzzyTermQuery;
use crate::query::term_query::TermQuery;
use crate::query::{BooleanQuery, Occur, Query};
use crate::schema::{Field, FieldType, IndexRecordOption, Schema, Term};
use std::sync::Arc;

/// Maximum number of exact ngram clauses generated for a fuzzy phrase query.
const MAX_NGRAM_CLAUSES: usize = 128;
/// Maximum number of ngram clauses when fuzzy matching is involved: each
/// fuzzy clause performs an automaton intersection with the term dictionary
/// of every segment, which is far more expensive than an exact term lookup.
const MAX_FUZZY_NGRAM_CLAUSES: usize = 32;

/// Optimizes phrase queries by rewriting them to use ngram terms when available
pub struct NgramQueryOptimizer {
    schema: Arc<Schema>,
}

impl NgramQueryOptimizer {
    /// Create a new optimizer for the given schema
    pub fn new(schema: Arc<Schema>) -> Self {
        Self { schema }
    }

    /// Rewrites a fuzzy phrase query into a boolean SHOULD query over
    /// combination ngrams, avoiding position matching entirely.
    ///
    /// The presence of a bigram proves that 2 query terms appear in order,
    /// and a trigram proves 3, so the rewrite can only honor
    /// `min_match == 2` (bigrams) or `min_match == 3` (trigrams). Any other
    /// `min_match` returns `None` and the caller falls back to position
    /// matching.
    ///
    /// Combination ngrams only exist in the index when `all_combinations`
    /// was enabled at indexing time; without it the rewrite would silently
    /// drop every gapped match, so `None` is returned as well.
    ///
    /// # Approximation
    ///
    /// Even when the rewrite applies, ordered term combinations are only
    /// indexed within the `all_combinations_window_size` window: matches
    /// spanning a wider gap are not found by the rewritten query. Callers
    /// needing exact semantics can disable the rewrite (see
    /// [`FuzzyPhraseQuery::set_ngram_optimization_enabled`](crate::query::FuzzyPhraseQuery::set_ngram_optimization_enabled)).
    ///
    /// When fuzzy matching is enabled (`min_term_length_for_fuzzy > 0`), the
    /// edit distance applies to the ngram string as a whole (e.g.
    /// `"quick brown"`), not to each word independently.
    pub fn optimize_fuzzy_phrase_query(
        &self,
        field: Field,
        terms: &[Term],
        min_match: usize,
        min_term_length_for_fuzzy: usize,
        fuzzy_distance: u8,
        fuzzy_transposition_cost_one: bool,
    ) -> Option<Box<dyn Query>> {
        let ngram_config = self.ngram_config(field)?;

        // Combination ngrams are only indexed in all_combinations mode.
        if !ngram_config.all_combinations {
            return None;
        }

        if terms.len() < 2 {
            return None;
        }

        // A bigram match proves exactly 2 ordered terms, a trigram match 3.
        let use_bigrams = min_match == 2 && ngram_config.contains_bigrams();
        let use_trigrams = min_match == 3 && ngram_config.contains_trigrams();
        if !use_bigrams && !use_trigrams {
            return None;
        }

        // Ngram strings can only be built from valid UTF-8 term values.
        let term_texts: Vec<&str> = terms
            .iter()
            .map(|term| std::str::from_utf8(term.serialized_value_bytes()).ok())
            .collect::<Option<Vec<&str>>>()?;

        // Determine which terms should use fuzzy matching based on length
        let use_fuzzy_per_term: Vec<bool> = if min_term_length_for_fuzzy > 0 {
            term_texts
                .iter()
                .map(|text| text.chars().count() >= min_term_length_for_fuzzy)
                .collect()
        } else {
            vec![false; term_texts.len()]
        };

        let n = term_texts.len();
        let projected_clauses = if use_bigrams {
            n * (n - 1) / 2
        } else {
            n * (n - 1) * (n - 2) / 6
        };
        let max_clauses = if use_fuzzy_per_term.iter().any(|&fuzzy| fuzzy) {
            MAX_FUZZY_NGRAM_CLAUSES
        } else {
            MAX_NGRAM_CLAUSES
        };
        if projected_clauses > max_clauses {
            return None;
        }

        let mut ngram_queries: Vec<(Occur, Box<dyn Query>)> =
            Vec::with_capacity(projected_clauses);
        let mut ngram_buffer = String::with_capacity(64);

        let mut push_ngram_clause = |ngram_text: &str, use_fuzzy: bool| {
            let ngram_term = Term::from_field_text(field, ngram_text);
            let query: Box<dyn Query> = if use_fuzzy {
                Box::new(FuzzyTermQuery::new(
                    ngram_term,
                    fuzzy_distance,
                    fuzzy_transposition_cost_one,
                ))
            } else {
                Box::new(TermQuery::new(ngram_term, IndexRecordOption::Basic))
            };
            ngram_queries.push((Occur::Should, query));
        };

        if use_bigrams {
            // All ordered pairs (i, j) with i < j.
            for i in 0..n {
                for j in (i + 1)..n {
                    // The indexer never writes a bigram of two identical
                    // words; the clause would match nothing.
                    if term_texts[i] == term_texts[j] {
                        continue;
                    }
                    ngram_buffer.clear();
                    ngram_buffer.push_str(term_texts[i]);
                    ngram_buffer.push(' ');
                    ngram_buffer.push_str(term_texts[j]);
                    push_ngram_clause(
                        &ngram_buffer,
                        use_fuzzy_per_term[i] || use_fuzzy_per_term[j],
                    );
                }
            }
        } else {
            // All ordered triplets (i, j, k) with i < j < k.
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        // The indexer skips trigrams of three identical words.
                        if term_texts[i] == term_texts[j] && term_texts[j] == term_texts[k] {
                            continue;
                        }
                        ngram_buffer.clear();
                        ngram_buffer.push_str(term_texts[i]);
                        ngram_buffer.push(' ');
                        ngram_buffer.push_str(term_texts[j]);
                        ngram_buffer.push(' ');
                        ngram_buffer.push_str(term_texts[k]);
                        push_ngram_clause(
                            &ngram_buffer,
                            use_fuzzy_per_term[i]
                                || use_fuzzy_per_term[j]
                                || use_fuzzy_per_term[k],
                        );
                    }
                }
            }
        }

        if ngram_queries.is_empty() {
            return None;
        }
        // Document matches if ANY ngram is present.
        Some(Box::new(BooleanQuery::with_minimum_required_clauses(
            ngram_queries,
            1,
        )))
    }

    /// Builds an ngram-based prefilter for an exact phrase query.
    ///
    /// The returned query matches a *superset* of the documents containing
    /// the phrase: any document with the phrase necessarily contains every
    /// consecutive bigram/trigram of it. The prefilter is NOT sufficient on
    /// its own (e.g. `"a b x b c"` contains the bigrams `"a b"` and `"b c"`
    /// without containing the phrase `"a b c"`), so callers must intersect
    /// it with an exact position-verified phrase query. On selective
    /// phrases this prunes most candidates before any positions are decoded.
    ///
    /// Returns `None` when the field has no usable ngram index or no clause
    /// can be generated.
    pub fn phrase_ngram_prefilter(
        &self,
        field: Field,
        terms: &[(usize, Term)],
        slop: u32,
    ) -> Option<Box<dyn Query>> {
        // With slop, terms need not be adjacent, so consecutive ngrams are
        // not a necessary condition anymore.
        if slop > 0 {
            return None;
        }

        let ngram_config = self.ngram_config(field)?;

        // Check if terms are consecutive (offsets differ by 1); phrases with
        // holes have no adjacency guarantee between the remaining terms.
        let is_consecutive = terms.windows(2).all(|w| w[1].0 == w[0].0 + 1);
        if !is_consecutive {
            return None;
        }

        let term_texts: Vec<&str> = terms
            .iter()
            .map(|(_, term)| std::str::from_utf8(term.serialized_value_bytes()).ok())
            .collect::<Option<Vec<&str>>>()?;
        let n = term_texts.len();

        let mut ngram_queries: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        let mut ngram_buffer = String::with_capacity(64);

        if ngram_config.contains_bigrams() && n >= 2 {
            for i in 0..n - 1 {
                // The indexer never writes a bigram of two identical words,
                // so requiring it would match nothing.
                if term_texts[i] == term_texts[i + 1] {
                    continue;
                }
                ngram_buffer.clear();
                ngram_buffer.push_str(term_texts[i]);
                ngram_buffer.push(' ');
                ngram_buffer.push_str(term_texts[i + 1]);
                let term_query =
                    TermQuery::new(Term::from_field_text(field, &ngram_buffer), IndexRecordOption::Basic);
                ngram_queries.push((Occur::Must, Box::new(term_query)));
            }
        }

        if ngram_config.contains_trigrams() && n >= 3 {
            for i in 0..n - 2 {
                // The indexer skips trigrams of three identical words.
                if term_texts[i] == term_texts[i + 1] && term_texts[i + 1] == term_texts[i + 2] {
                    continue;
                }
                ngram_buffer.clear();
                ngram_buffer.push_str(term_texts[i]);
                ngram_buffer.push(' ');
                ngram_buffer.push_str(term_texts[i + 1]);
                ngram_buffer.push(' ');
                ngram_buffer.push_str(term_texts[i + 2]);
                let term_query =
                    TermQuery::new(Term::from_field_text(field, &ngram_buffer), IndexRecordOption::Basic);
                ngram_queries.push((Occur::Must, Box::new(term_query)));
            }
        }

        if ngram_queries.is_empty() {
            None
        } else {
            Some(Box::new(BooleanQuery::new(ngram_queries)))
        }
    }

    /// Returns the word ngram config of `field` if it is an ngram-enabled
    /// text field.
    fn ngram_config(&self, field: Field) -> Option<&crate::indexer::WordNgramConfig> {
        let field_entry = self.schema.get_field_entry(field);
        let text_options = match field_entry.field_type() {
            FieldType::Str(options) => options,
            _ => return None,
        };
        let indexing_options = text_options.get_indexing_options()?;
        let ngram_config = indexing_options.word_ngrams()?;
        if !ngram_config.is_enabled() {
            return None;
        }
        Some(ngram_config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::WordNgramSet;
    use crate::WordNgramConfig;

    fn ngram_schema(
        ngram_set: WordNgramSet,
        all_combinations: bool,
    ) -> (Arc<Schema>, Field, Field) {
        let mut schema_builder = Schema::builder();
        let ngram_config = WordNgramConfig::builder()
            .ngram_types(ngram_set)
            .all_combinations(all_combinations)
            .build();
        let text_field = schema_builder.add_text_field(
            "text",
            crate::schema::TextOptions::default().set_indexing_options(
                crate::schema::TextFieldIndexing::default()
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions)
                    .set_word_ngrams(ngram_config),
            ),
        );
        let text_field_no_ngrams = schema_builder.add_text_field(
            "text_no_ngrams",
            crate::schema::TextOptions::default().set_indexing_options(
                crate::schema::TextFieldIndexing::default()
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            ),
        );
        (Arc::new(schema_builder.build()), text_field, text_field_no_ngrams)
    }

    fn bigram_set() -> WordNgramSet {
        WordNgramSet::new()
            .with_ngram_ff()
            .with_ngram_fr()
            .with_ngram_rf()
    }

    fn bigram_and_trigram_set() -> WordNgramSet {
        bigram_set()
            .with_ngram_fff()
            .with_ngram_ffr()
            .with_ngram_frf()
            .with_ngram_rff()
    }

    #[test]
    fn test_phrase_prefilter_gates() {
        let (schema, text_field, text_field_no_ngrams) = ngram_schema(bigram_set(), false);
        let optimizer = NgramQueryOptimizer::new(schema);

        let terms = vec![
            (0, Term::from_field_text(text_field, "hello")),
            (1, Term::from_field_text(text_field, "world")),
        ];

        // No prefilter with slop: adjacency is not required anymore.
        assert!(optimizer
            .phrase_ngram_prefilter(text_field, &terms, 1)
            .is_none());
        // Prefilter generated for an exact phrase.
        assert!(optimizer
            .phrase_ngram_prefilter(text_field, &terms, 0)
            .is_some());

        // No prefilter without ngram indexing.
        let terms_no_ngrams = vec![
            (0, Term::from_field_text(text_field_no_ngrams, "hello")),
            (1, Term::from_field_text(text_field_no_ngrams, "world")),
        ];
        assert!(optimizer
            .phrase_ngram_prefilter(text_field_no_ngrams, &terms_no_ngrams, 0)
            .is_none());

        // Phrases with position holes have no adjacency guarantee.
        let terms_with_hole = vec![
            (0, Term::from_field_text(text_field, "hello")),
            (2, Term::from_field_text(text_field, "world")),
        ];
        assert!(optimizer
            .phrase_ngram_prefilter(text_field, &terms_with_hole, 0)
            .is_none());
    }

    #[test]
    fn test_phrase_prefilter_skips_identical_word_ngrams() {
        let (schema, text_field, _) = ngram_schema(bigram_set(), false);
        let optimizer = NgramQueryOptimizer::new(schema);

        // "no no no": every bigram would pair identical words, which the
        // indexer never writes; no clause remains.
        let terms_repeated = vec![
            (0, Term::from_field_text(text_field, "no")),
            (1, Term::from_field_text(text_field, "no")),
            (2, Term::from_field_text(text_field, "no")),
        ];
        assert!(optimizer
            .phrase_ngram_prefilter(text_field, &terms_repeated, 0)
            .is_none());

        // "no x no": the bigrams "no x" and "x no" are both indexable.
        let terms_alternating = vec![
            (0, Term::from_field_text(text_field, "no")),
            (1, Term::from_field_text(text_field, "x")),
            (2, Term::from_field_text(text_field, "no")),
        ];
        assert!(optimizer
            .phrase_ngram_prefilter(text_field, &terms_alternating, 0)
            .is_some());
    }

    #[test]
    fn test_optimize_fuzzy_phrase_query_gates() {
        let (schema, text_field, text_field_no_ngrams) =
            ngram_schema(bigram_and_trigram_set(), true);
        let optimizer = NgramQueryOptimizer::new(schema);

        let terms = vec![
            Term::from_field_text(text_field, "the"),
            Term::from_field_text(text_field, "quick"),
            Term::from_field_text(text_field, "brown"),
            Term::from_field_text(text_field, "fox"),
        ];

        // Single term cannot form an ngram.
        let single_term = vec![Term::from_field_text(text_field, "hello")];
        assert!(optimizer
            .optimize_fuzzy_phrase_query(text_field, &single_term, 2, 0, 1, true)
            .is_none());

        // min_match of 2 (bigrams) and 3 (trigrams) are supported.
        assert!(optimizer
            .optimize_fuzzy_phrase_query(text_field, &terms, 2, 0, 1, true)
            .is_some());
        assert!(optimizer
            .optimize_fuzzy_phrase_query(text_field, &terms, 3, 0, 1, true)
            .is_some());

        // min_match of 1 or >= 4 cannot be proven with bigrams/trigrams.
        assert!(optimizer
            .optimize_fuzzy_phrase_query(text_field, &terms, 1, 0, 1, true)
            .is_none());
        assert!(optimizer
            .optimize_fuzzy_phrase_query(text_field, &terms, 4, 0, 1, true)
            .is_none());

        // No rewrite without ngram indexing.
        let terms_no_ngrams = vec![
            Term::from_field_text(text_field_no_ngrams, "hello"),
            Term::from_field_text(text_field_no_ngrams, "world"),
        ];
        assert!(optimizer
            .optimize_fuzzy_phrase_query(text_field_no_ngrams, &terms_no_ngrams, 2, 0, 1, true)
            .is_none());
    }

    #[test]
    fn test_optimize_fuzzy_phrase_query_requires_all_combinations() {
        // Without all_combinations, only adjacent ngrams are indexed and the
        // rewrite would drop every gapped match.
        let (schema, text_field, _) = ngram_schema(bigram_set(), false);
        let optimizer = NgramQueryOptimizer::new(schema);
        let terms = vec![
            Term::from_field_text(text_field, "hello"),
            Term::from_field_text(text_field, "world"),
        ];
        assert!(optimizer
            .optimize_fuzzy_phrase_query(text_field, &terms, 2, 0, 1, true)
            .is_none());
    }

    #[test]
    fn test_optimize_fuzzy_phrase_query_min_match_needs_matching_ngram_size() {
        // min_match == 3 requires trigrams; a bigram-only config must fall
        // back rather than accept 2-term matches.
        let (schema, text_field, _) = ngram_schema(bigram_set(), true);
        let optimizer = NgramQueryOptimizer::new(schema);
        let terms = vec![
            Term::from_field_text(text_field, "the"),
            Term::from_field_text(text_field, "quick"),
            Term::from_field_text(text_field, "brown"),
        ];
        assert!(optimizer
            .optimize_fuzzy_phrase_query(text_field, &terms, 3, 0, 1, true)
            .is_none());
        assert!(optimizer
            .optimize_fuzzy_phrase_query(text_field, &terms, 2, 0, 1, true)
            .is_some());
    }

    #[test]
    fn test_optimize_fuzzy_phrase_query_duplicate_terms() {
        let (schema, text_field, _) = ngram_schema(bigram_set(), true);
        let optimizer = NgramQueryOptimizer::new(schema);

        // Both bigrams would pair identical words: no clause remains, so
        // the caller must fall back to position matching.
        let terms = vec![
            Term::from_field_text(text_field, "test"),
            Term::from_field_text(text_field, "test"),
        ];
        assert!(optimizer
            .optimize_fuzzy_phrase_query(text_field, &terms, 2, 0, 1, true)
            .is_none());

        // With a distinct term in between, some bigrams survive.
        let terms_mixed = vec![
            Term::from_field_text(text_field, "test"),
            Term::from_field_text(text_field, "gap"),
            Term::from_field_text(text_field, "test"),
        ];
        assert!(optimizer
            .optimize_fuzzy_phrase_query(text_field, &terms_mixed, 2, 0, 1, true)
            .is_some());
    }
}

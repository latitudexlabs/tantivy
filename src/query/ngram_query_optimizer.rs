/// Query optimization for leveraging word ngram indexes
///
/// This module provides query rewriting to use word ngram indexes when available,
/// significantly speeding up phrase queries by avoiding position matching.
use crate::core::searcher::Searcher;
use crate::indexer::{FrequentTermTracker, NgramType};
use crate::query::fuzzy_query::FuzzyTermQuery;
use crate::query::term_query::TermQuery;
use crate::query::{BooleanQuery, Occur, Query};
use crate::schema::{Field, FieldType, IndexRecordOption, Schema, Term};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Optimizes phrase queries by rewriting them to use ngram terms when available
pub struct NgramQueryOptimizer {
    schema: Arc<Schema>,
}

impl NgramQueryOptimizer {
    /// Create a new optimizer for the given schema
    pub fn new(schema: Arc<Schema>) -> Self {
        Self { schema }
    }

    /// Optimize fuzzy phrase query by generating ngrams for all ordered combinations
    ///
    /// For a query with terms [a, b, c, d], this generates ngrams for:
    /// - All 2-term combinations: ab, ac, ad, bc, bd, cd
    /// - All 3-term combinations: abc, abd, acd, bcd
    ///
    /// When fuzzy matching is enabled (min_term_length_for_fuzzy > 0), ngrams containing
    /// fuzzy-eligible terms will use FuzzyTermQuery instead of exact TermQuery.
    ///
    /// Returns a boolean SHOULD query matching any of these ngrams.
    /// This allows efficient matching with gaps without position checking.
    pub fn optimize_fuzzy_phrase_query(
        &self,
        field: Field,
        terms: &[Term],
        searcher_opt: Option<&Searcher>,
        min_term_length_for_fuzzy: usize,
        fuzzy_distance: u8,
        fuzzy_transposition_cost_one: bool,
    ) -> Option<Box<dyn Query>> {
        // Get field configuration
        let field_entry = self.schema.get_field_entry(field);
        let text_options = match field_entry.field_type() {
            FieldType::Str(options) => options,
            _ => return None,
        };

        let indexing_options = text_options.get_indexing_options()?;
        let ngram_config = indexing_options.word_ngrams()?;

        // Early return if ngrams are not enabled
        if !ngram_config.is_enabled() {
            return None;
        }

        if terms.len() < 2 {
            return None;
        }

        // Extract term strings from existing terms - avoid .to_vec() allocation
        let term_texts: Vec<&str> = terms
            .iter()
            .map(|term| std::str::from_utf8(term.serialized_value_bytes()).unwrap_or(""))
            .collect();

        if term_texts.len() < 2 {
            return None;
        }

        // Determine which terms should use fuzzy matching based on length
        let use_fuzzy_per_term: Vec<bool> = if min_term_length_for_fuzzy > 0 {
            term_texts
                .iter()
                .map(|text| text.chars().count() >= min_term_length_for_fuzzy)
                .collect()
        } else {
            vec![false; term_texts.len()]
        };

        // Check if we need frequency tracking
        // If all ngram types are configured, we can skip frequency tracking entirely
        let needs_frequency_tracking = if ngram_config.contains_bigrams() {
            !ngram_config.has_all_bigram_types()
        } else if ngram_config.contains_trigrams() {
            !ngram_config.has_all_trigram_types()
        } else {
            false
        };

        // Get frequent terms tracker (only if needed)
        let frequent_tracker_opt = if needs_frequency_tracking {
            searcher_opt.and_then(|searcher| {
                searcher
                    .segment_readers()
                    .first()
                    .and_then(|reader| reader.get_frequent_terms(field.field_id()))
            })
        } else {
            None
        };

        // Only require frequency data if we need it for classification
        if needs_frequency_tracking && frequent_tracker_opt.is_none() {
            return None;
        }

        let frequent_tracker = frequent_tracker_opt.as_ref().map(|arc| arc.as_ref());

        // Pre-compute term hashes only if needed for frequency tracking
        let term_hashes: Vec<u64> = if needs_frequency_tracking {
            term_texts.iter().map(|t| Self::hash_term(t)).collect()
        } else {
            Vec::new()
        };

        // Pre-allocate with estimated capacity for forward-only combinations
        // Bigrams: n*(n-1)/2, Trigrams: n*(n-1)*(n-2)/6
        let n = term_texts.len();
        let bigram_capacity = if ngram_config.contains_bigrams() {
            n * (n - 1) / 2
        } else {
            0
        };
        let trigram_capacity = if ngram_config.contains_trigrams() {
            n * (n - 1) * (n - 2) / 6
        } else {
            0
        };
        let mut ngram_queries = Vec::with_capacity(bigram_capacity + trigram_capacity);

        // Reusable buffer for ngram construction
        let mut ngram_buffer = String::with_capacity(64);

        // Generate all ordered combinations for bigrams and trigrams
        // For bigrams: all pairs (i, j) where i < j
        if ngram_config.contains_bigrams() {
            for i in 0..term_texts.len() {
                for j in (i + 1)..term_texts.len() {
                    // Check if this ngram type should be indexed
                    let ngram_type =
                        self.classify_combination_bigram(i, j, &term_hashes, frequent_tracker);

                    if ngram_config.has_ngram_type(&ngram_type) {
                        // Build ngram string efficiently
                        ngram_buffer.clear();
                        let needed_len = term_texts[i].len() + 1 + term_texts[j].len();
                        if ngram_buffer.capacity() < needed_len {
                            ngram_buffer.reserve(needed_len - ngram_buffer.capacity());
                        }
                        ngram_buffer.push_str(term_texts[i]);
                        ngram_buffer.push(' ');
                        ngram_buffer.push_str(term_texts[j]);

                        let ngram_term = Term::from_field_text(field, &ngram_buffer);
                        
                        // Use fuzzy matching if any component term should use fuzzy
                        let use_fuzzy = use_fuzzy_per_term[i] || use_fuzzy_per_term[j];
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
                    }
                }
            }
        }

        // Generate all ordered combinations for trigrams
        // For trigrams: all triplets (i, j, k) where i < j < k
        if ngram_config.contains_trigrams() {
            for i in 0..term_texts.len() {
                for j in (i + 1)..term_texts.len() {
                    for k in (j + 1)..term_texts.len() {
                        let ngram_type = self.classify_combination_trigram(
                            i,
                            j,
                            k,
                            &term_hashes,
                            frequent_tracker,
                        );

                        if ngram_config.has_ngram_type(&ngram_type) {
                            // Build trigram string efficiently
                            ngram_buffer.clear();
                            let needed_len = term_texts[i].len()
                                + 1
                                + term_texts[j].len()
                                + 1
                                + term_texts[k].len();
                            if ngram_buffer.capacity() < needed_len {
                                ngram_buffer.reserve(needed_len - ngram_buffer.capacity());
                            }
                            ngram_buffer.push_str(term_texts[i]);
                            ngram_buffer.push(' ');
                            ngram_buffer.push_str(term_texts[j]);
                            ngram_buffer.push(' ');
                            ngram_buffer.push_str(term_texts[k]);

                            let ngram_term = Term::from_field_text(field, &ngram_buffer);
                            
                            // Use fuzzy matching if any component term should use fuzzy
                            let use_fuzzy = use_fuzzy_per_term[i] || use_fuzzy_per_term[j] || use_fuzzy_per_term[k];
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
                        }
                    }
                }
            }
        }

        if ngram_queries.is_empty() {
            return None;
        }
        let minimum_should_match = if ngram_queries.len() > 1 { 1 } else { 0 };
        // Create boolean query with SHOULD (OR) - document matches if ANY ngram is present
        Some(Box::new(BooleanQuery::with_minimum_required_clauses(
            ngram_queries,
            minimum_should_match,
        )))
    }

    /// Classify a bigram combination based on term frequencies
    fn classify_combination_bigram(
        &self,
        i: usize,
        j: usize,
        term_hashes: &[u64],
        tracker: Option<&FrequentTermTracker>,
    ) -> NgramType {
        if let Some(tracker) = tracker {
            let first_frequent = tracker.is_frequent(term_hashes[i]);
            let second_frequent = tracker.is_frequent(term_hashes[j]);

            match (first_frequent, second_frequent) {
                (true, true) => NgramType::NgramFF,
                (true, false) => NgramType::NgramFR,
                (false, true) => NgramType::NgramRF,
                (false, false) => {
                    // Rare-Rare - pick one based on position
                    if i == 0 {
                        NgramType::NgramRF
                    } else {
                        NgramType::NgramFR
                    }
                }
            }
        } else {
            // No frequency data, use position-based heuristic
            if i == 0 {
                NgramType::NgramFF
            } else {
                NgramType::NgramFR
            }
        }
    }

    /// Classify a trigram combination based on term frequencies
    fn classify_combination_trigram(
        &self,
        i: usize,
        j: usize,
        k: usize,
        term_hashes: &[u64],
        tracker: Option<&FrequentTermTracker>,
    ) -> NgramType {
        if let Some(tracker) = tracker {
            let first_frequent = tracker.is_frequent(term_hashes[i]);
            let second_frequent = tracker.is_frequent(term_hashes[j]);
            let third_frequent = tracker.is_frequent(term_hashes[k]);

            match (first_frequent, second_frequent, third_frequent) {
                (true, true, true) => NgramType::NgramFFF,
                (false, true, true) => NgramType::NgramRFF,
                (true, true, false) => NgramType::NgramFFR,
                (true, false, true) => NgramType::NgramFRF,
                _ => {
                    // Other combinations - pick based on position
                    if i == 0 {
                        NgramType::NgramFFF
                    } else {
                        NgramType::NgramFFR
                    }
                }
            }
        } else {
            // No frequency data, use position-based heuristic
            if i == 0 {
                NgramType::NgramFFF
            } else {
                NgramType::NgramFFR
            }
        }
    }

    /// Try to optimize a phrase query using ngrams
    ///
    /// Returns Some(optimized_query) if optimization is possible, None otherwise
    pub fn optimize_phrase_query(
        &self,
        field: Field,
        terms: &[(usize, Term)],
        slop: u32,
        searcher_opt: Option<&Searcher>,
    ) -> Option<Box<dyn Query>> {
        // Don't optimize if slop is used - ngrams only work for exact phrases
        if slop > 0 {
            return None;
        }

        // Get field configuration
        let field_entry = self.schema.get_field_entry(field);
        let text_options = match field_entry.field_type() {
            FieldType::Str(options) => options,
            _ => return None,
        };

        let indexing_options = text_options.get_indexing_options()?;
        let ngram_config = indexing_options.word_ngrams()?;

        // Early return if ngrams are not enabled
        if !ngram_config.is_enabled() {
            return None;
        }

        // Check if we have positions indexed (fallback to regular phrase query if not)
        let has_positions = indexing_options.index_option().has_positions();
        if !has_positions {
            return None;
        }

        // Check if terms are consecutive (offsets differ by 1)
        let is_consecutive = terms.windows(2).all(|w| w[1].0 == w[0].0 + 1);
        if !is_consecutive {
            return None; // Skip phrases with gaps
        }

        // Check for repeated terms - ngram optimization doesn't work correctly for these
        // because duplicate ngrams collapse into a single term check
        // Example: "no no no" generates ["no no", "no no"] which both check for the same term
        // This would incorrectly match a document containing only "no no"
        let term_bytes: Vec<_> = terms
            .iter()
            .map(|(_, t)| t.serialized_value_bytes())
            .collect();
        let has_repeated_terms = term_bytes.windows(2).any(|w| w[0] == w[1]);
        if has_repeated_terms {
            return None; // Fall back to position-based matching for repeated terms
        }

        // Check if we need frequency tracking
        // If all ngram types are configured, we can skip frequency tracking entirely
        let needs_frequency_tracking = if ngram_config.contains_bigrams() {
            !ngram_config.has_all_bigram_types()
        } else if ngram_config.contains_trigrams() {
            !ngram_config.has_all_trigram_types()
        } else {
            false
        };

        // Try to get frequent terms info from searcher (only if needed)
        let frequent_tracker_opt = if needs_frequency_tracking {
            searcher_opt.and_then(|searcher| {
                searcher
                    .segment_readers()
                    .first()
                    .and_then(|reader| reader.get_frequent_terms(field.field_id()))
            })
        } else {
            None
        };

        // Only require frequency data if we need it for classification
        if needs_frequency_tracking && frequent_tracker_opt.is_none() {
            return None;
        }

        // Extract just the term strings - avoid .to_vec() allocation
        let term_texts: Vec<&str> = terms
            .iter()
            .map(|(_, term)| std::str::from_utf8(term.serialized_value_bytes()).unwrap_or(""))
            .collect();

        // Pre-compute term hashes to avoid repeated hashing (only if needed)
        let term_hashes: Vec<u64> = if needs_frequency_tracking {
            term_texts.iter().map(|t| Self::hash_term(t)).collect()
        } else {
            Vec::new()
        };

        // Generate ngram queries based on configured ngram types and term frequencies
        // Pre-allocate with estimated capacity
        let bigram_cap = if ngram_config.contains_bigrams() {
            term_texts.len().saturating_sub(1)
        } else {
            0
        };
        let trigram_cap = if ngram_config.contains_trigrams() {
            term_texts.len().saturating_sub(2)
        } else {
            0
        };
        let mut ngram_queries = Vec::with_capacity(bigram_cap + trigram_cap);

        // Reusable buffer for ngram text construction
        let mut ngram_buffer = String::with_capacity(64);

        // Try to generate bigrams
        if ngram_config.contains_bigrams() && term_texts.len() >= 2 {
            for i in 0..term_texts.len() - 1 {
                let ngram_type = self.classify_bigram_with_hashes(
                    i,
                    &term_hashes,
                    term_texts.len(),
                    frequent_tracker_opt.as_ref().map(|t| t.as_ref()),
                );

                // Check if this ngram type should be indexed
                if ngram_config.has_ngram_type(&ngram_type) {
                    // Build ngram string efficiently
                    ngram_buffer.clear();
                    let needed_len = term_texts[i].len() + 1 + term_texts[i + 1].len();
                    if ngram_buffer.capacity() < needed_len {
                        ngram_buffer.reserve(needed_len - ngram_buffer.capacity());
                    }
                    ngram_buffer.push_str(term_texts[i]);
                    ngram_buffer.push(' ');
                    ngram_buffer.push_str(term_texts[i + 1]);

                    let ngram_term = Term::from_field_text(field, &ngram_buffer);
                    let term_query = TermQuery::new(ngram_term, IndexRecordOption::Basic);
                    ngram_queries.push((Occur::Must, Box::new(term_query) as Box<dyn Query>));
                }
            }
        }

        // Try to generate trigrams
        if ngram_config.contains_trigrams() && term_texts.len() >= 3 {
            for i in 0..term_texts.len() - 2 {
                let ngram_type = self.classify_trigram_with_hashes(
                    i,
                    &term_hashes,
                    term_texts.len(),
                    frequent_tracker_opt.as_ref().map(|t| t.as_ref()),
                );

                if ngram_config.has_ngram_type(&ngram_type) {
                    // Build trigram string efficiently
                    ngram_buffer.clear();
                    let needed_len = term_texts[i].len()
                        + 1
                        + term_texts[i + 1].len()
                        + 1
                        + term_texts[i + 2].len();
                    if ngram_buffer.capacity() < needed_len {
                        ngram_buffer.reserve(needed_len - ngram_buffer.capacity());
                    }
                    ngram_buffer.push_str(term_texts[i]);
                    ngram_buffer.push(' ');
                    ngram_buffer.push_str(term_texts[i + 1]);
                    ngram_buffer.push(' ');
                    ngram_buffer.push_str(term_texts[i + 2]);

                    let ngram_term = Term::from_field_text(field, &ngram_buffer);
                    let term_query = TermQuery::new(ngram_term, IndexRecordOption::Basic);
                    ngram_queries.push((Occur::Must, Box::new(term_query) as Box<dyn Query>));
                }
            }
        }

        // If we generated any ngram queries, return a boolean query combining them
        if !ngram_queries.is_empty() {
            Some(Box::new(BooleanQuery::new(ngram_queries)))
        } else {
            None
        }
    }

    /// Compute hash for a term string
    fn hash_term(term: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        term.hash(&mut hasher);
        hasher.finish()
    }

    /// Classify bigram based on actual term frequencies (optimized with pre-computed hashes)
    fn classify_bigram_with_hashes(
        &self,
        position: usize,
        term_hashes: &[u64],
        total_terms: usize,
        tracker: Option<&FrequentTermTracker>,
    ) -> NgramType {
        if let Some(tracker) = tracker {
            let first_frequent = tracker.is_frequent(term_hashes[position]);
            let second_frequent = tracker.is_frequent(term_hashes[position + 1]);

            match (first_frequent, second_frequent) {
                (true, true) => NgramType::NgramFF,
                (true, false) => NgramType::NgramFR,
                (false, true) => NgramType::NgramRF,
                (false, false) => return self.classify_bigram_position(position, total_terms),
            }
        } else {
            self.classify_bigram_position(position, total_terms)
        }
    }

    /// Classify trigram based on actual term frequencies (optimized with pre-computed hashes)
    fn classify_trigram_with_hashes(
        &self,
        position: usize,
        term_hashes: &[u64],
        total_terms: usize,
        tracker: Option<&FrequentTermTracker>,
    ) -> NgramType {
        if let Some(tracker) = tracker {
            let first_frequent = tracker.is_frequent(term_hashes[position]);
            let second_frequent = tracker.is_frequent(term_hashes[position + 1]);
            let third_frequent = tracker.is_frequent(term_hashes[position + 2]);

            match (first_frequent, second_frequent, third_frequent) {
                (true, true, true) => NgramType::NgramFFF,
                (false, true, true) => NgramType::NgramRFF,
                (true, true, false) => NgramType::NgramFFR,
                (true, false, true) => NgramType::NgramFRF,
                _ => return self.classify_trigram_position(position, total_terms),
            }
        } else {
            self.classify_trigram_position(position, total_terms)
        }
    }

    /// Classify bigram position to determine its type
    fn classify_bigram_position(&self, position: usize, total_terms: usize) -> NgramType {
        let is_first = position == 0;
        let is_last = position == total_terms - 2; // Last possible bigram

        match (is_first, is_last) {
            (true, true) => NgramType::NgramFF,   // Only one bigram
            (true, false) => NgramType::NgramFF,  // First bigram
            (false, true) => NgramType::NgramFR,  // Last bigram
            (false, false) => NgramType::NgramRF, // Middle bigram
        }
    }

    /// Classify trigram position to determine its type
    fn classify_trigram_position(&self, position: usize, total_terms: usize) -> NgramType {
        let is_first = position == 0;
        let is_last = position == total_terms - 3; // Last possible trigram

        match (is_first, is_last) {
            (true, true) => NgramType::NgramFFF,   // Only one trigram
            (true, false) => NgramType::NgramFFF,  // First trigram
            (false, true) => NgramType::NgramFFR,  // Last trigram
            (false, false) => NgramType::NgramFRF, // Middle trigram
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::WordNgramSet;
    use crate::WordNgramConfig;

    #[test]
    fn test_bigram_classification() {
        let schema = Arc::new(Schema::builder().build());
        let optimizer = NgramQueryOptimizer::new(schema);

        // Single bigram in 2-term phrase
        assert_eq!(optimizer.classify_bigram_position(0, 2), NgramType::NgramFF);

        // Three-term phrase
        assert_eq!(optimizer.classify_bigram_position(0, 3), NgramType::NgramFF);
        assert_eq!(optimizer.classify_bigram_position(1, 3), NgramType::NgramFR);

        // Four-term phrase
        assert_eq!(optimizer.classify_bigram_position(0, 4), NgramType::NgramFF);
        assert_eq!(optimizer.classify_bigram_position(1, 4), NgramType::NgramRF);
        assert_eq!(optimizer.classify_bigram_position(2, 4), NgramType::NgramFR);
    }

    #[test]
    fn test_trigram_classification() {
        let schema = Arc::new(Schema::builder().build());
        let optimizer = NgramQueryOptimizer::new(schema);

        // Single trigram in 3-term phrase
        assert_eq!(
            optimizer.classify_trigram_position(0, 3),
            NgramType::NgramFFF
        );

        // Four-term phrase
        assert_eq!(
            optimizer.classify_trigram_position(0, 4),
            NgramType::NgramFFF
        );
        assert_eq!(
            optimizer.classify_trigram_position(1, 4),
            NgramType::NgramFFR
        );

        // Five-term phrase
        assert_eq!(
            optimizer.classify_trigram_position(0, 5),
            NgramType::NgramFFF
        );
        assert_eq!(
            optimizer.classify_trigram_position(1, 5),
            NgramType::NgramFRF
        );
        assert_eq!(
            optimizer.classify_trigram_position(2, 5),
            NgramType::NgramFFR
        );
    }

    #[test]
    fn test_optimize_phrase_with_slop() {
        let mut schema_builder = Schema::builder();
        let ngram_config = WordNgramConfig::builder()
            .ngram_types(
                WordNgramSet::new()
                    .with_ngram_ff()
                    .with_ngram_fr()
                    .with_ngram_rf(),
            )
            .build();

        let text_field = schema_builder.add_text_field(
            "text",
            crate::schema::TextOptions::default().set_indexing_options(
                crate::schema::TextFieldIndexing::default()
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions)
                    .set_word_ngrams(ngram_config),
            ),
        );
        let schema = Arc::new(schema_builder.build());
        let optimizer = NgramQueryOptimizer::new(schema);

        let terms = vec![
            (0, Term::from_field_text(text_field, "hello")),
            (1, Term::from_field_text(text_field, "world")),
        ];

        // Should not optimize with slop > 0
        let result = optimizer.optimize_phrase_query(text_field, &terms, 1, None);
        assert!(result.is_none());

        // Should now optimize without searcher when all bigram types are configured
        // (no frequency tracking needed)
        let result = optimizer.optimize_phrase_query(text_field, &terms, 0, None);
        assert!(
            result.is_some(),
            "Should optimize when all bigram types are configured"
        );
    }

    #[test]
    fn test_optimize_fuzzy_phrase_query() {
        let mut schema_builder = Schema::builder();

        // Create field with ngrams enabled
        let ngram_config = WordNgramConfig::builder()
            .ngram_types(
                WordNgramSet::new()
                    .with_ngram_ff()
                    .with_ngram_fr()
                    .with_ngram_rf()
                    .with_ngram_fff()
                    .with_ngram_ffr()
                    .with_ngram_frf(),
            )
            .build();

        let text_field = schema_builder.add_text_field(
            "text",
            crate::schema::TextOptions::default().set_indexing_options(
                crate::schema::TextFieldIndexing::default()
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions)
                    .set_word_ngrams(ngram_config),
            ),
        );

        // Create field without ngrams
        let text_field_no_ngrams = schema_builder.add_text_field(
            "text_no_ngrams",
            crate::schema::TextOptions::default().set_indexing_options(
                crate::schema::TextFieldIndexing::default()
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            ),
        );

        let schema = Arc::new(schema_builder.build());
        let optimizer = NgramQueryOptimizer::new(schema);

        // Test: Should return None with single term
        let single_term = vec![Term::from_field_text(text_field, "hello")];
        let result = optimizer.optimize_fuzzy_phrase_query(text_field, &single_term, None, 0, 1, true);
        assert!(result.is_none());

        // Test: Can now optimize without searcher when all bigram and trigram types configured
        // (no frequency tracking needed)
        let terms = vec![
            Term::from_field_text(text_field, "the"),
            Term::from_field_text(text_field, "quick"),
            Term::from_field_text(text_field, "brown"),
            Term::from_field_text(text_field, "fox"),
        ];
        let result = optimizer.optimize_fuzzy_phrase_query(text_field, &terms, None, 0, 1, true);
        // Note: This config has all bigram types but is missing NgramRFF for trigrams,
        // so it can still optimize bigrams without a searcher
        assert!(
            result.is_some(),
            "Should optimize bigrams when all bigram types configured"
        );

        // Test: Should return None when ngrams not enabled
        let terms_no_ngrams = vec![
            Term::from_field_text(text_field_no_ngrams, "hello"),
            Term::from_field_text(text_field_no_ngrams, "world"),
        ];
        let result =
            optimizer.optimize_fuzzy_phrase_query(text_field_no_ngrams, &terms_no_ngrams, None, 0, 1, true);
        assert!(result.is_none());

        // Test: Verify forward-only combination generation with 4 terms
        // For [a, b, c, d] with bigrams and trigrams:
        // - Bigrams: ab, ac, ad, bc, bd, cd (6 combinations)
        // - Trigrams: abc, abd, acd, bcd (4 combinations)
        // Total: 10 combinations
        // Without a real searcher/index, we can't fully test the query generation,
        // but the early return behavior is tested above.
    }

    #[test]
    fn test_partial_ngram_types_require_frequency_tracking() {
        let mut schema_builder = Schema::builder();

        // Create field with only partial bigram types (missing RF)
        let ngram_config = WordNgramConfig::builder()
            .ngram_types(WordNgramSet::new().with_ngram_ff().with_ngram_fr())
            .build();

        let text_field = schema_builder.add_text_field(
            "text",
            crate::schema::TextOptions::default().set_indexing_options(
                crate::schema::TextFieldIndexing::default()
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions)
                    .set_word_ngrams(ngram_config),
            ),
        );

        let schema = Arc::new(schema_builder.build());
        let optimizer = NgramQueryOptimizer::new(schema);

        let terms = vec![
            (0, Term::from_field_text(text_field, "hello")),
            (1, Term::from_field_text(text_field, "world")),
        ];

        // Should NOT optimize without searcher when partial types configured
        // (frequency tracking is needed to classify ngrams)
        let result = optimizer.optimize_phrase_query(text_field, &terms, 0, None);
        assert!(
            result.is_none(),
            "Should not optimize with partial types and no searcher"
        );
    }

    #[test]
    fn test_repeated_terms_fallback() -> crate::Result<()> {
        use crate::schema::Schema;
        use crate::{Index, IndexWriter};

        // Create schema with ngram indexing
        let mut schema_builder = Schema::builder();
        let ngram_config = crate::indexer::WordNgramConfig::with_set(
            WordNgramSet::new()
                .with_ngram_ff()
                .with_ngram_fr()
                .with_ngram_rf(),
        )
        .with_frequent_threshold(0.01)
        .with_max_frequent_terms(10_000);

        let text_field = schema_builder.add_text_field(
            "text",
            crate::schema::TextOptions::default().set_indexing_options(
                crate::schema::TextFieldIndexing::default()
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions)
                    .set_word_ngrams(ngram_config),
            ),
        );

        let schema = Arc::new(schema_builder.build());
        let index = Index::create_in_ram((*schema).clone());

        // Index documents with repeated terms
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field => "no no"))?;
            index_writer.add_document(doc!(text_field => "no no no"))?;
            index_writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let optimizer = NgramQueryOptimizer::new(schema);

        // Test: Phrase with repeated terms should NOT be optimized
        // Query: "no no no" should not match document "no no"
        let terms_repeated = vec![
            (0, Term::from_field_text(text_field, "no")),
            (1, Term::from_field_text(text_field, "no")),
            (2, Term::from_field_text(text_field, "no")),
        ];

        let result =
            optimizer.optimize_phrase_query(text_field, &terms_repeated, 0, Some(&searcher));
        assert!(
            result.is_none(),
            "Should NOT optimize phrase queries with repeated terms to avoid false matches"
        );

        Ok(())
    }
}

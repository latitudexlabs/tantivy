use super::FuzzyPhraseWeight;
use crate::query::bm25::Bm25Weight;
use crate::query::ngram_query_optimizer::NgramQueryOptimizer;
use crate::query::{EnableScoring, Query, Weight};
use crate::schema::{Field, IndexRecordOption, Term};
use std::sync::Arc;

/// `FuzzyPhraseQuery` matches documents where at least a minimum number of phrase terms
/// appear in order, allowing for missing terms between them.
///
/// For instance, a fuzzy phrase query for `["test1", "test2"]` with min_match=2 will match
/// documents containing "test1 ... test2" in that order, where "..." can be any content.
///
/// Unlike a strict phrase query, this allows gaps and missing terms from the original query.
///
/// This query uses intersection sets for efficient matching.
///
/// # Ngram Optimization
///
/// When the field indexes word ngrams with `all_combinations` enabled, the
/// query can be rewritten to a boolean SHOULD query over ngram terms,
/// avoiding position matching entirely. Since a bigram proves 2 terms in
/// order and a trigram proves 3, the rewrite only applies when:
/// - `min_match == 2` and bigrams are indexed: for terms [a, b, c, d] the
///   rewrite matches any of the bigrams ab, ac, ad, bc, bd, cd;
/// - `min_match == 3` and trigrams are indexed: it matches any of the
///   trigrams abc, abd, acd, bcd.
///
/// Any other configuration falls back to position-based matching.
///
/// The rewrite is an approximation: ordered combinations are only indexed
/// within the `all_combinations_window_size` window, so matches whose terms
/// are further apart are not found. Use
/// [`set_ngram_optimization_enabled(false)`](Self::set_ngram_optimization_enabled)
/// to force exact position-based matching.
///
/// # Fuzzy Term Matching
///
/// When combined with ngram fields, you can enable fuzzy matching based on
/// term length. This allows matching documents with typos in the indexed
/// terms: searching for "programming language features" will also match
/// documents containing "programing language features".
///
/// Use [`set_min_term_length_for_fuzzy`](Self::set_min_term_length_for_fuzzy) to enable
/// fuzzy matching for terms above a minimum character length. Note that the
/// edit distance applies to the ngram string as a whole (e.g.
/// `"programming language"`), not to each word independently: one edit
/// budget is shared across the words of an ngram.
///
/// # Example with Fuzzy Ngrams
///
/// ```rust
/// use tantivy::schema::{Schema, TextOptions, TextFieldIndexing, IndexRecordOption};
/// use tantivy::{Index, Term, doc};
/// use tantivy::query::FuzzyPhraseQuery;
/// use tantivy::indexer::WordNgramSet;
/// use tantivy::WordNgramConfig;
///
/// # fn main() -> tantivy::Result<()> {
/// let mut schema_builder = Schema::builder();
///
/// // Configure field with ngram combinations for optimization
/// let ngram_config = WordNgramConfig::builder()
///     .ngram_types(WordNgramSet::new().with_ngram_ff().with_ngram_fr().with_ngram_rf())
///     .all_combinations(true)
///     .build();
///
/// let text_options = TextOptions::default()
///     .set_indexing_options(
///         TextFieldIndexing::default()
///             .set_index_option(IndexRecordOption::WithFreqsAndPositions)
///             .set_word_ngrams(ngram_config),
///     );
/// let text_field = schema_builder.add_text_field("text", text_options);
/// let schema = schema_builder.build();
/// let index = Index::create_in_ram(schema);
///
/// // Index a document with a typo
/// let mut index_writer = index.writer(50_000_000)?;
/// index_writer.add_document(doc!(text_field => "programing language features"))?;
/// index_writer.commit()?;
///
/// // Search with fuzzy matching for terms >= 5 characters
/// let terms = vec![
///     Term::from_field_text(text_field, "programming"),  // Will fuzzy match "programing"
///     Term::from_field_text(text_field, "language"),
///     Term::from_field_text(text_field, "features"),
/// ];
///
/// let query = FuzzyPhraseQuery::new(terms, 2)
///     .set_min_term_length_for_fuzzy(5)      // Enable fuzzy for terms >= 5 chars
///     .set_fuzzy_distance(1)                  // Allow 1 edit distance
///     .set_fuzzy_transposition_cost_one(true);
///
/// let reader = index.reader()?;
/// let searcher = reader.searcher();
/// let top_docs = searcher.search(&query, &tantivy::collector::TopDocs::with_limit(10).order_by_score())?;
///
/// // Will match the document despite the typo "programing" vs "programming"
/// assert_eq!(top_docs.len(), 1);
/// # Ok(())
/// # }
/// ```
///
/// Using a `FuzzyPhraseQuery` on a field requires positions
/// to be indexed for this field.
#[derive(Clone, Debug)]
pub struct FuzzyPhraseQuery {
    field: Field,
    phrase_terms: Arc<Vec<Term>>,
    min_match: usize,
    /// Minimum term length to apply fuzzy matching (0 = no fuzzy matching)
    min_term_length_for_fuzzy: usize,
    /// Maximum edit distance for fuzzy matching
    fuzzy_distance: u8,
    /// Whether transpositions cost 1 (true) or 2 (false)
    fuzzy_transposition_cost_one: bool,
    /// Whether the ngram-based rewrite may be applied (default: true)
    ngram_optimization_enabled: bool,
}

impl FuzzyPhraseQuery {
    /// Creates a new `FuzzyPhraseQuery` given a list of terms and minimum matches required.
    ///
    /// The `phrase_terms` are the terms that should appear in order in the document.
    /// The `min_match` parameter specifies the minimum number of these terms that must
    /// appear in the document (in the given order) for it to be considered a match.
    ///
    /// All terms must belong to the same field.
    ///
    /// # Panics
    /// - If there are fewer than 2 terms
    /// - If min_match is 0 or greater than the number of terms
    /// - If terms belong to different fields
    pub fn new(phrase_terms: Vec<Term>, min_match: usize) -> FuzzyPhraseQuery {
        assert!(
            phrase_terms.len() >= 2,
            "A fuzzy phrase query requires at least 2 terms"
        );
        assert!(
            min_match > 0 && min_match <= phrase_terms.len(),
            "min_match must be between 1 and the number of terms ({})",
            phrase_terms.len()
        );

        let field = phrase_terms[0].field();
        assert!(
            phrase_terms[1..].iter().all(|term| term.field() == field),
            "All terms from a fuzzy phrase query must belong to the same field"
        );

        FuzzyPhraseQuery {
            field,
            phrase_terms: Arc::new(phrase_terms),
            min_match,
            min_term_length_for_fuzzy: 0, // No fuzzy matching by default
            fuzzy_distance: 1,
            fuzzy_transposition_cost_one: true,
            ngram_optimization_enabled: true,
        }
    }

    /// The [`Field`] this `FuzzyPhraseQuery` is targeting.
    pub fn field(&self) -> Field {
        self.field
    }

    /// Terms in the phrase.
    pub fn phrase_terms(&self) -> &[Term] {
        &self.phrase_terms
    }

    /// Minimum number of terms that must match.
    pub fn min_match(&self) -> usize {
        self.min_match
    }

    /// Sets the minimum term length for fuzzy matching in ngram fields.
    /// Terms with length >= this value will use fuzzy matching.
    /// Set to 0 (default) to disable fuzzy matching.
    pub fn set_min_term_length_for_fuzzy(mut self, min_length: usize) -> Self {
        self.min_term_length_for_fuzzy = min_length;
        self
    }

    /// Sets the maximum edit distance for fuzzy matching (default: 1).
    pub fn set_fuzzy_distance(mut self, distance: u8) -> Self {
        self.fuzzy_distance = distance;
        self
    }

    /// Sets whether transpositions cost 1 (true, default) or 2 (false).
    pub fn set_fuzzy_transposition_cost_one(mut self, transposition_cost_one: bool) -> Self {
        self.fuzzy_transposition_cost_one = transposition_cost_one;
        self
    }

    /// Enables or disables the ngram-based rewrite (default: enabled).
    ///
    /// The rewrite only finds matches whose terms co-occur within the
    /// index-time `all_combinations` window; disable it to force exact,
    /// position-based matching regardless of the field's ngram config.
    pub fn set_ngram_optimization_enabled(mut self, enabled: bool) -> Self {
        self.ngram_optimization_enabled = enabled;
        self
    }

    /// Returns the [`FuzzyPhraseWeight`] for the given fuzzy phrase query.
    pub(crate) fn fuzzy_phrase_weight(
        &self,
        enable_scoring: EnableScoring<'_>,
    ) -> crate::Result<FuzzyPhraseWeight> {
        let schema = enable_scoring.schema();
        let field_entry = schema.get_field_entry(self.field);
        let has_positions = field_entry
            .field_type()
            .get_index_record_option()
            .map(IndexRecordOption::has_positions)
            .unwrap_or(false);
        if !has_positions {
            let field_name = field_entry.name();
            return Err(crate::TantivyError::SchemaError(format!(
                "Applied fuzzy phrase query on field {field_name:?}, which does not have positions indexed"
            )));
        }

        let bm25_weight_opt = match enable_scoring {
            EnableScoring::Enabled {
                statistics_provider,
                ..
            } => Some(Bm25Weight::for_terms(
                statistics_provider,
                &self.phrase_terms,
            )?),
            EnableScoring::Disabled { .. } => None,
        };

        Ok(FuzzyPhraseWeight::new(
            Arc::clone(&self.phrase_terms),
            self.min_match,
            bm25_weight_opt,
        ))
    }
}

impl Query for FuzzyPhraseQuery {
    /// Create the weight associated with a query.
    ///
    /// See [`Weight`].
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        // Try ngram optimization: generate all ordered combinations as ngram terms
        if self.ngram_optimization_enabled {
            let schema = enable_scoring.schema();
            let optimizer = NgramQueryOptimizer::new(Arc::new(schema.clone()));

            if let Some(optimized_query) = optimizer.optimize_fuzzy_phrase_query(
                self.field,
                &self.phrase_terms,
                self.min_match,
                self.min_term_length_for_fuzzy,
                self.fuzzy_distance,
                self.fuzzy_transposition_cost_one,
            ) {
                // Use the optimized query (ngram-based boolean OR)
                return optimized_query.weight(enable_scoring);
            }
        }

        // Fall back to regular fuzzy phrase query (position-based matching)
        let weight = self.fuzzy_phrase_weight(enable_scoring)?;
        Ok(Box::new(weight))
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        for term in self.phrase_terms.iter() {
            visitor(term, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Schema;
    use crate::{Index, Term};

    // Helper function to extract text from a document
    fn get_doc_text(
        searcher: &crate::Searcher,
        addr: crate::DocAddress,
        field: crate::schema::Field,
    ) -> String {
        let doc = searcher
            .doc::<crate::schema::TantivyDocument>(addr)
            .unwrap();
        let field_value = doc.get_first(field).unwrap();
        let value: crate::schema::OwnedValue = field_value.into();
        if let crate::schema::OwnedValue::Str(s) = value {
            s
        } else {
            panic!("Expected string value");
        }
    }

    fn create_test_index(docs: &[&str]) -> crate::Result<Index> {
        use crate::schema::{IndexRecordOption, TextFieldIndexing, TextOptions};

        let mut schema_builder = Schema::builder();
        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored();
        let text = schema_builder.add_text_field("text", text_options);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);

        let mut index_writer = index.writer(50_000_000)?;
        for doc_text in docs {
            let mut doc = crate::schema::TantivyDocument::new();
            doc.add_text(text, doc_text);
            index_writer.add_document(doc)?;
        }
        // Commit once to create a single segment
        index_writer.commit()?;

        Ok(index)
    }

    #[test]
    fn test_fuzzy_phrase_basic() -> crate::Result<()> {
        let docs = vec![
            "test1 test2 test3",
            "test2 test3",
            "test3 test2",
            "test1 test2",
        ];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "test1"),
            Term::from_field_text(text_field, "test2"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 2);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // Should match 2 docs (docs with test1 and test2 in order)
        assert_eq!(top_docs.len(), 2);

        // Verify the correct documents matched by checking their content
        let mut matched_texts: Vec<String> = top_docs
            .iter()
            .map(|(_, addr)| get_doc_text(&searcher, *addr, text_field))
            .collect();
        matched_texts.sort();

        assert!(matched_texts.contains(&"test1 test2 test3".to_string()));
        assert!(matched_texts.contains(&"test1 test2".to_string()));

        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_with_gaps() -> crate::Result<()> {
        let docs = vec![
            "test1 gap1 gap2 test2",
            "test1 test2",
            "test2 test1",
            "test1 gap1 test2 test3",
        ];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "test1"),
            Term::from_field_text(text_field, "test2"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 2);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // Should match 3 docs (all that have test1 before test2)
        // Should not match the doc with test2 before test1
        assert_eq!(top_docs.len(), 3);

        let mut matched_texts: Vec<String> = top_docs
            .iter()
            .map(|(_, addr)| get_doc_text(&searcher, *addr, text_field))
            .collect();
        matched_texts.sort();

        assert!(matched_texts.contains(&"test1 gap1 gap2 test2".to_string()));
        assert!(matched_texts.contains(&"test1 test2".to_string()));
        assert!(matched_texts.contains(&"test1 gap1 test2 test3".to_string()));
        assert!(!matched_texts.contains(&"test2 test1".to_string()));

        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_min_match() -> crate::Result<()> {
        let docs = vec![
            "test1 gap test2 gap test3",
            "test1 gap test2",
            "test2 gap test3",
            "test1",
        ];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "test1"),
            Term::from_field_text(text_field, "test2"),
            Term::from_field_text(text_field, "test3"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 2);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // The first three docs have at least 2 terms in order.
        assert_eq!(top_docs.len(), 3);

        let mut matched_texts: Vec<String> = top_docs
            .iter()
            .map(|(_, addr)| get_doc_text(&searcher, *addr, text_field))
            .collect();
        matched_texts.sort();

        // Docs with at least 2 terms in order: test1->test2, test1->test3, or test2->test3
        assert!(matched_texts.contains(&"test1 gap test2 gap test3".to_string()));
        assert!(matched_texts.contains(&"test1 gap test2".to_string()));
        assert!(matched_texts.contains(&"test2 gap test3".to_string()));
        assert!(!matched_texts.contains(&"test1".to_string())); // Only has 1 term

        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_no_matches() -> crate::Result<()> {
        let docs = vec!["alpha beta gamma", "delta epsilon", "zeta eta theta"];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        // Query for terms that don't exist in any document
        let terms = vec![
            Term::from_field_text(text_field, "test1"),
            Term::from_field_text(text_field, "test2"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 2);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // No documents should match
        assert_eq!(top_docs.len(), 0);

        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_single_term_match() -> crate::Result<()> {
        let docs = vec![
            "test1 gap test2 gap test3",
            "test1 only",
            "test2 only",
            "no matches here",
        ];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "test1"),
            Term::from_field_text(text_field, "test2"),
            Term::from_field_text(text_field, "test3"),
        ];
        // Require all 3 terms
        let query = FuzzyPhraseQuery::new(terms, 3);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // Only first document has all 3 terms in order
        assert_eq!(top_docs.len(), 1);

        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_repeated_terms() -> crate::Result<()> {
        let docs = vec![
            "apple orange apple banana",
            "apple banana orange apple",
            "banana apple orange",
            "apple apple apple",
        ];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "apple"),
            Term::from_field_text(text_field, "orange"),
            Term::from_field_text(text_field, "banana"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 3);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // Docs 0 and 1 have apple, orange, banana in that order
        // Doc 0: apple(0) orange(1) banana(3) - matches
        // Doc 1: apple(0) banana(1) orange(2) apple(3) - apple at 0, then need orange after that at 2, then banana but it's at 1 which is before orange - doesn't match
        // Actually doc 1: looking for apple->orange->banana, we have apple(0) and apple(3), orange(2), banana(1)
        // Starting from apple(0): need orange after 0, found at 2, need banana after 2, none found - depth 2
        // Starting from apple(3): no terms after - depth 1
        // So doc 1 doesn't match
        assert_eq!(top_docs.len(), 1);

        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_large_gaps() -> crate::Result<()> {
        let docs = vec![
            "start gap1 gap2 gap3 gap4 gap5 gap6 gap7 gap8 gap9 middle gap1 gap2 gap3 end",
            "start middle end",
            "start end middle",
        ];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "start"),
            Term::from_field_text(text_field, "middle"),
            Term::from_field_text(text_field, "end"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 3);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // Docs 0 and 1 have start->middle->end in order
        // Doc 2 has start->end->middle (wrong order)
        assert_eq!(top_docs.len(), 2);

        let mut matched_texts: Vec<String> = top_docs
            .iter()
            .map(|(_, addr)| get_doc_text(&searcher, *addr, text_field))
            .collect();
        matched_texts.sort();

        assert!(matched_texts.contains(
            &"start gap1 gap2 gap3 gap4 gap5 gap6 gap7 gap8 gap9 middle gap1 gap2 gap3 end"
                .to_string()
        ));
        assert!(matched_texts.contains(&"start middle end".to_string()));
        assert!(!matched_texts.contains(&"start end middle".to_string()));

        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_min_match_one() -> crate::Result<()> {
        let docs = vec![
            "test1 test2 test3",
            "test1 only",
            "test2 only",
            "no matches",
        ];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "test1"),
            Term::from_field_text(text_field, "test2"),
            Term::from_field_text(text_field, "test3"),
        ];
        // Only require 1 term to match
        let query = FuzzyPhraseQuery::new(terms, 1);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // First 3 documents have at least one of the terms
        assert_eq!(top_docs.len(), 3);

        let mut matched_texts: Vec<String> = top_docs
            .iter()
            .map(|(_, addr)| get_doc_text(&searcher, *addr, text_field))
            .collect();
        matched_texts.sort();

        assert!(matched_texts.contains(&"test1 test2 test3".to_string()));
        assert!(matched_texts.contains(&"test1 only".to_string()));
        assert!(matched_texts.contains(&"test2 only".to_string()));
        assert!(!matched_texts.contains(&"no matches".to_string()));

        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_partial_order() -> crate::Result<()> {
        let docs = vec!["a b c d e", "a c e", "e c a", "a d c b e"];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "a"),
            Term::from_field_text(text_field, "c"),
            Term::from_field_text(text_field, "e"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 3);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // Docs 0, 1, and 3 have a->c->e in order
        // Doc 2 has e->c->a (reversed order)
        assert_eq!(top_docs.len(), 3);

        let mut matched_texts: Vec<String> = top_docs
            .iter()
            .map(|(_, addr)| get_doc_text(&searcher, *addr, text_field))
            .collect();
        matched_texts.sort();

        assert!(matched_texts.contains(&"a b c d e".to_string()));
        assert!(matched_texts.contains(&"a c e".to_string()));
        assert!(matched_texts.contains(&"a d c b e".to_string()));
        assert!(!matched_texts.contains(&"e c a".to_string()));

        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_skip_middle_term() -> crate::Result<()> {
        let docs = vec![
            "word1 word2 word3 word4",
            "word1 word3 word4",
            "word1 word4",
            "word2 word3 word4",
        ];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "word1"),
            Term::from_field_text(text_field, "word2"),
            Term::from_field_text(text_field, "word3"),
            Term::from_field_text(text_field, "word4"),
        ];
        // Require at least 3 terms in order
        let query = FuzzyPhraseQuery::new(terms, 3);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // Doc 0: all 4 terms in order - matches
        // Doc 1: word1->word3->word4 (3 in order) - matches
        // Doc 2: word1->word4 (only 2 in order) - doesn't match
        // Doc 3: word2->word3->word4 (3 in order) - matches
        assert_eq!(top_docs.len(), 3);

        let mut matched_texts: Vec<String> = top_docs
            .iter()
            .map(|(_, addr)| get_doc_text(&searcher, *addr, text_field))
            .collect();
        matched_texts.sort();

        assert!(matched_texts.contains(&"word1 word2 word3 word4".to_string()));
        assert!(matched_texts.contains(&"word1 word3 word4".to_string()));
        assert!(matched_texts.contains(&"word2 word3 word4".to_string()));
        assert!(!matched_texts.contains(&"word1 word4".to_string()));

        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_empty_query() -> crate::Result<()> {
        let docs = vec!["test1 test2 test3"];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        // Create query with minimum 2 terms but only provide 2 terms
        let terms = vec![
            Term::from_field_text(text_field, "test1"),
            Term::from_field_text(text_field, "test2"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 2);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // Should find the document
        assert_eq!(top_docs.len(), 1);

        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_all_same_term() -> crate::Result<()> {
        let docs = vec!["test test test", "test gap test", "other words here"];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "test"),
            Term::from_field_text(text_field, "test"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 2);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // Both docs 0 and 1 have multiple occurrences of "test"
        assert_eq!(top_docs.len(), 2);

        let mut matched_texts: Vec<String> = top_docs
            .iter()
            .map(|(_, addr)| get_doc_text(&searcher, *addr, text_field))
            .collect();
        matched_texts.sort();

        assert!(matched_texts.contains(&"test test test".to_string()));
        assert!(matched_texts.contains(&"test gap test".to_string()));
        assert!(!matched_texts.contains(&"other words here".to_string()));

        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_with_ngram_optimization() -> crate::Result<()> {
        // Index documents that match different ngram combinations
        let docs = vec![
            "the quick brown fox jumps over lazy dog",
            "the xyz brown fox jumps", // missing 'quick' - matches via ngrams
            "quick brown fox",         // missing 'the' - matches via ngrams
            "the quick fox jumps",     // missing 'brown' - matches via ngrams
            "fox brown quick the",     // wrong order - shouldn't match
            "unrelated content here",  // no match
            "the fox",                 // only 2 terms - must not satisfy min_match=3
        ];
        let index = create_combinations_ngram_index(&docs, true)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Query for 4 terms, requiring at least 3 to match in order.
        // With min_match=3 the rewrite uses trigram combinations only:
        // the-quick-brown, the-quick-fox, the-brown-fox, quick-brown-fox
        let terms = vec![
            Term::from_field_text(text_field, "the"),
            Term::from_field_text(text_field, "quick"),
            Term::from_field_text(text_field, "brown"),
            Term::from_field_text(text_field, "fox"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 3);

        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // The first four documents have at least 3 terms in order.
        assert_eq!(top_docs.len(), 4, "Expected 4 documents to match");

        let mut matched_texts: Vec<String> = top_docs
            .iter()
            .map(|(_, addr)| get_doc_text(&searcher, *addr, text_field))
            .collect();
        matched_texts.sort();

        assert!(matched_texts.contains(&"the quick brown fox jumps over lazy dog".to_string()));
        assert!(matched_texts.contains(&"the xyz brown fox jumps".to_string()));
        assert!(matched_texts.contains(&"quick brown fox".to_string()));
        assert!(matched_texts.contains(&"the quick fox jumps".to_string()));
        assert!(!matched_texts.contains(&"fox brown quick the".to_string()));
        assert!(!matched_texts.contains(&"unrelated content here".to_string()));
        // Only 2 of the 4 query terms: a bigram match must not satisfy
        // min_match=3.
        assert!(!matched_texts.contains(&"the fox".to_string()));

        Ok(())
    }

    /// The ngram rewrite only finds matches within the all_combinations
    /// window; disabling it must restore exact position-based matching.
    #[test]
    fn test_fuzzy_phrase_ngram_optimization_opt_out() -> crate::Result<()> {
        // Gap wider than the default all_combinations window (100 tokens),
        // so the "quick brown" bigram is not indexed.
        let gap: String = (0..120).map(|i| format!("g{i} ")).collect();
        let doc_text = format!("quick {gap}brown");
        let docs = vec![doc_text.as_str()];
        let index = create_combinations_ngram_index(&docs, false)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "quick"),
            Term::from_field_text(text_field, "brown"),
        ];

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Documented approximation: the ngram rewrite misses the match.
        let optimized_query = FuzzyPhraseQuery::new(terms.clone(), 2);
        let top_docs = searcher.search(
            &optimized_query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;
        assert_eq!(top_docs.len(), 0);

        // Opting out restores exact position-based matching.
        let exact_query =
            FuzzyPhraseQuery::new(terms, 2).set_ngram_optimization_enabled(false);
        let top_docs = searcher.search(
            &exact_query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;
        assert_eq!(top_docs.len(), 1);

        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_with_edge_ngram_prefix_matching() -> crate::Result<()> {
        use crate::indexer::WordNgramSet;
        use crate::schema::{IndexRecordOption, TextFieldIndexing, TextOptions};
        use crate::WordNgramConfig;

        let mut schema_builder = Schema::builder();

        // Create field with bigram ngrams enabled
        let ngram_config = WordNgramConfig::builder()
            .ngram_types(
                WordNgramSet::new()
                    .with_ngram_ff()
                    .with_ngram_fr()
                    .with_ngram_rf(),
            )
            .build();

        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions)
                    .set_word_ngrams(ngram_config),
            )
            .set_stored();

        let text_field = schema_builder.add_text_field("text", text_options);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);

        // Simulate character edge ngrams by manually indexing edge terms
        // For "quick brown fox", we'd index: qu, qui, quic, quick, br, bro, brow, brown, fo, fox
        // This simulates what a character edge ngram tokenizer would produce
        let docs = vec![
            // Full words indexed with their edge ngrams
            "quick brown fox",
            "qui bro fo",        // Partial edge ngrams - should match via bigrams
            "quic brown fo",     // Mixed edges - should match
            "unrelated content", // No match
        ];

        let mut index_writer = index.writer(50_000_000)?;
        for doc_text in &docs {
            let mut doc = crate::schema::TantivyDocument::new();
            doc.add_text(text_field, doc_text);
            index_writer.add_document(doc)?;
        }
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Query with edge ngram prefixes - simulating prefix matching
        // In a real implementation, the query parser would generate these edge terms
        let terms = vec![
            Term::from_field_text(text_field, "qui"), // prefix of "quick"
            Term::from_field_text(text_field, "bro"), // prefix of "brown"
            Term::from_field_text(text_field, "fo"),  // prefix of "fox"
        ];
        let query = FuzzyPhraseQuery::new(terms, 2);

        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // Should match docs with at least 2 terms in order
        // Doc 0: no match (has "quick brown fox", not the edge terms)
        // Doc 1: has "qui bro fo" - all 3 edge terms match
        // Doc 2: has "quic brown fo" - 2 terms match (brown contains "bro", but we're looking for exact "bro" term)
        assert!(!top_docs.is_empty(), "Expected at least 1 document to match");

        let mut matched_texts: Vec<String> = top_docs
            .iter()
            .map(|(_, addr)| get_doc_text(&searcher, *addr, text_field))
            .collect();
        matched_texts.sort();

        // Verify the document with edge terms matched
        assert!(matched_texts.contains(&"qui bro fo".to_string()));

        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_edge_ngram_combinations() -> crate::Result<()> {
        use crate::indexer::WordNgramSet;
        use crate::schema::{IndexRecordOption, TextFieldIndexing, TextOptions};
        use crate::WordNgramConfig;

        let mut schema_builder = Schema::builder();

        // Enable bigrams WITH all_combinations mode to generate all ordered pairs
        let ngram_config = WordNgramConfig::builder()
            .ngram_types(
                WordNgramSet::new()
                    .with_ngram_ff()
                    .with_ngram_fr()
                    .with_ngram_rf(),
            )
            .all_combinations(true)  // KEY: Enable all combinations mode
            .build();

        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions)
                    .set_word_ngrams(ngram_config),
            )
            .set_stored();

        let text_field = schema_builder.add_text_field("text", text_options);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);

        // Test: Demonstrate edge ngram concept with simpler terms
        // In production, a custom tokenizer would generate: qu, qui, quic, quick, br, bro, brow, brown
        // Here we use simple edge-like terms to test the bigram generation concept
        let docs = vec![
            "qu qui quic quick br bro brow brown fo fox",
            // Document with edge terms in order
            "qu br fo",
            // Document with some edge terms
            "qu br other",
            // Document with all edge terms but spaced
            "qu gap br gap fo",
            // Wrong order
            "fo br qu",
        ];

        let mut index_writer = index.writer(50_000_000)?;
        for doc_text in &docs {
            let mut doc = crate::schema::TantivyDocument::new();
            doc.add_text(text_field, doc_text);
            index_writer.add_document(doc)?;
        }
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Query using edge-like terms with min_match=2
        // With all_combinations mode, the index will contain:
        // - Doc 0: qu-qui, qu-quic, qu-quick, qu-br, qu-bro, ..., qui-quic, qui-quick, qui-br, etc.
        // - Doc 1: qu-br, qu-fo, br-fo
        // - Doc 2: qu-br, qu-other, br-other
        // - Doc 3: qu-gap, qu-br, qu-fo, gap-br, gap-fo, br-gap, br-fo (all combinations)
        // - Doc 4: fo-br, fo-qu, br-qu
        let terms = vec![
            Term::from_field_text(text_field, "qu"),
            Term::from_field_text(text_field, "br"),
            Term::from_field_text(text_field, "fo"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 2);

        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // Should match documents with at least 2 terms in order
        // With all_combinations, all docs except the last should match:
        // - Doc 0: has qu-br (✓), qu-fo (✓), br-fo (✓) 
        // - Doc 1: has qu-br (✓), br-fo (✓), qu-fo (✓)
        // - Doc 2: has qu-br (✓), but no qu-fo or br-fo
        // - Doc 3: has qu-br (✓), qu-fo (✓), br-fo (✓) even with gaps
        // - Doc 4: wrong order - br before qu, fo before br
        assert!(
            top_docs.len() >= 4,
            "Expected at least 4 documents to match, got {}",
            top_docs.len()
        );

        let mut matched_texts: Vec<String> = top_docs
            .iter()
            .map(|(_, addr)| get_doc_text(&searcher, *addr, text_field))
            .collect();
        matched_texts.sort();

        // Verify correct matches - at minimum these should match
        assert!(matched_texts.contains(&"qu qui quic quick br bro brow brown fo fox".to_string()));
        assert!(matched_texts.contains(&"qu br fo".to_string()));
        assert!(matched_texts.contains(&"qu br other".to_string()));
        assert!(matched_texts.contains(&"qu gap br gap fo".to_string()));

        // Wrong order should never match
        assert!(!matched_texts.contains(&"fo br qu".to_string()));

        Ok(())
    }

    /// A chain of `min_match` ordered terms must be found even when a
    /// query term is present in the document at an out-of-order position.
    /// Here `word1(0) -> word3(1) -> word4(2)` is a valid chain of 3;
    /// `word2` appears after them and must be skipped, not consumed.
    #[test]
    fn test_fuzzy_phrase_chain_skips_out_of_order_term() -> crate::Result<()> {
        let docs = vec!["word1 word3 word4 word2"];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "word1"),
            Term::from_field_text(text_field, "word2"),
            Term::from_field_text(text_field, "word3"),
            Term::from_field_text(text_field, "word4"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 3);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        assert_eq!(
            top_docs.len(),
            1,
            "doc has 3 terms in order (word1 word3 word4) and should match"
        );
        Ok(())
    }

    /// A document matching more query terms in order must score higher than
    /// one matching fewer, all else (doc length) being equal.
    #[test]
    fn test_fuzzy_phrase_score_reflects_match_count() -> crate::Result<()> {
        // Same length, same first/last term; middle term differs.
        let docs = vec!["apple banana cherry", "apple pear cherry"];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "apple"),
            Term::from_field_text(text_field, "banana"),
            Term::from_field_text(text_field, "cherry"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 2);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;
        assert_eq!(top_docs.len(), 2);

        let mut scores_by_text: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();
        for (score, addr) in &top_docs {
            scores_by_text.insert(get_doc_text(&searcher, *addr, text_field), *score);
        }
        let full_match = scores_by_text["apple banana cherry"];
        let partial_match = scores_by_text["apple pear cherry"];
        assert!(
            full_match > partial_match,
            "3-term match ({full_match}) should outscore 2-term match ({partial_match})"
        );
        Ok(())
    }

    /// The scorer must report a non-zero size hint so query planners do not
    /// treat it as an empty/free docset.
    #[test]
    fn test_fuzzy_phrase_scorer_size_hint_nonzero() -> crate::Result<()> {
        use crate::docset::DocSet;
        use crate::query::EnableScoring;

        let docs = vec!["test1 test2", "test1 test2 test3", "test1 test2 extra"];
        let index = create_test_index(&docs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "test1"),
            Term::from_field_text(text_field, "test2"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 2);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let weight = query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        // The writer may spread the docs over several segments; every
        // segment holds at least one doc with both terms, so every scorer
        // must report a non-zero hint.
        for segment_reader in searcher.segment_readers() {
            let scorer = weight.scorer(segment_reader, 1.0)?;
            assert!(scorer.size_hint() > 0);
        }
        Ok(())
    }

    /// Helper: index with word-ngram combinations enabled (bigrams and
    /// optionally trigrams), using all_combinations mode.
    fn create_combinations_ngram_index(
        docs: &[&str],
        with_trigrams: bool,
    ) -> crate::Result<Index> {
        use crate::indexer::WordNgramSet;
        use crate::schema::{IndexRecordOption, TextFieldIndexing, TextOptions};
        use crate::WordNgramConfig;

        let mut ngram_set = WordNgramSet::new()
            .with_ngram_ff()
            .with_ngram_fr()
            .with_ngram_rf();
        if with_trigrams {
            ngram_set = ngram_set
                .with_ngram_fff()
                .with_ngram_ffr()
                .with_ngram_frf()
                .with_ngram_rff();
        }
        let ngram_config = WordNgramConfig::builder()
            .ngram_types(ngram_set)
            .all_combinations(true)
            .build();

        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions)
                    .set_word_ngrams(ngram_config),
            )
            .set_stored();

        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", text_options);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);

        let mut index_writer = index.writer(50_000_000)?;
        for doc_text in docs {
            let mut doc = crate::schema::TantivyDocument::new();
            doc.add_text(text_field, doc_text);
            index_writer.add_document(doc)?;
        }
        index_writer.commit()?;
        Ok(index)
    }

    /// min_match must be honored by the ngram-optimized path: a document
    /// containing only 2 of the query terms must not satisfy min_match=3
    /// just because one bigram matches.
    #[test]
    fn test_fuzzy_phrase_ngram_min_match_not_satisfied_by_bigram() -> crate::Result<()> {
        let docs = vec!["the fox", "the quick brown fox"];
        let index = create_combinations_ngram_index(&docs, true)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "the"),
            Term::from_field_text(text_field, "quick"),
            Term::from_field_text(text_field, "brown"),
            Term::from_field_text(text_field, "fox"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 3);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        assert_eq!(top_docs.len(), 1, "only the 4-term doc has 3 terms in order");
        let text = get_doc_text(&searcher, top_docs[0].1, text_field);
        assert_eq!(text, "the quick brown fox");
        Ok(())
    }

    /// min_match=1 cannot be expressed with bigrams; the query must fall
    /// back to position matching so single-term documents still match.
    #[test]
    fn test_fuzzy_phrase_ngram_min_match_one_falls_back() -> crate::Result<()> {
        let docs = vec!["programming only here", "nothing relevant"];
        let index = create_combinations_ngram_index(&docs, false)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "programming"),
            Term::from_field_text(text_field, "language"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 1);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        assert_eq!(top_docs.len(), 1, "doc containing 'programming' must match");
        Ok(())
    }

    /// The indexer never writes a bigram of two identical words, so a query
    /// of repeated terms must not be rewritten into a dead ngram clause.
    #[test]
    fn test_fuzzy_phrase_ngram_duplicate_terms_fall_back() -> crate::Result<()> {
        let docs = vec!["test gap test", "test only"];
        let index = create_combinations_ngram_index(&docs, false)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let terms = vec![
            Term::from_field_text(text_field, "test"),
            Term::from_field_text(text_field, "test"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 2);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        assert_eq!(top_docs.len(), 1, "doc with two 'test' occurrences must match");
        Ok(())
    }

    /// Without all_combinations, only adjacent ngrams are indexed, so the
    /// rewrite cannot honor the "gaps allowed" contract and must fall back.
    #[test]
    fn test_fuzzy_phrase_consecutive_ngrams_fall_back_for_gaps() -> crate::Result<()> {
        use crate::indexer::WordNgramSet;
        use crate::schema::{IndexRecordOption, TextFieldIndexing, TextOptions};
        use crate::WordNgramConfig;

        // Consecutive-only ngram config (all_combinations defaults to false).
        let ngram_config = WordNgramConfig::builder()
            .ngram_types(
                WordNgramSet::new()
                    .with_ngram_ff()
                    .with_ngram_fr()
                    .with_ngram_rf(),
            )
            .build();
        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions)
                    .set_word_ngrams(ngram_config),
            )
            .set_stored();
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", text_options);
        let index = Index::create_in_ram(schema_builder.build());
        let mut index_writer = index.writer(50_000_000)?;
        let mut doc = crate::schema::TantivyDocument::new();
        doc.add_text(text_field, "quick gap brown");
        index_writer.add_document(doc)?;
        index_writer.commit()?;

        let terms = vec![
            Term::from_field_text(text_field, "quick"),
            Term::from_field_text(text_field, "brown"),
        ];
        let query = FuzzyPhraseQuery::new(terms, 2);

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        assert_eq!(
            top_docs.len(),
            1,
            "'quick gap brown' has both terms in order and must match"
        );
        Ok(())
    }

    #[test]
    fn test_fuzzy_phrase_with_fuzzy_ngrams() -> crate::Result<()> {
        // Test documents with typos
        let docs = vec![
            "programming language features",
            "programing language features", // typo: missing 'm'
            "programming languag features", // typo: missing 'e'
            "programing languag features",  // typos in both words
            "coding syntax tools",          // completely different words
        ];
        let index = create_combinations_ngram_index(&docs, true)?;
        let schema = index.schema();
        let text_field = schema.get_field("text")?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Query with fuzzy matching for words >= 5 characters
        let terms = vec![
            Term::from_field_text(text_field, "programming"),
            Term::from_field_text(text_field, "language"),
            Term::from_field_text(text_field, "features"),
        ];

        let query = FuzzyPhraseQuery::new(terms, 3)
            .set_min_term_length_for_fuzzy(5) // Apply fuzzy to all three terms
            .set_fuzzy_distance(1) // Allow 1 edit
            .set_fuzzy_transposition_cost_one(true);

        let top_docs = searcher.search(
            &query,
            &crate::collector::TopDocs::with_limit(10).order_by_score(),
        )?;

        // Exact match plus the two single-typo docs. The edit distance
        // applies to the trigram string as a whole, so the doc with a typo
        // in two words (2 edits) must not match.
        assert_eq!(top_docs.len(), 3, "Expected 3 matches");

        let mut matched_texts: Vec<String> = top_docs
            .iter()
            .map(|(_, addr)| get_doc_text(&searcher, *addr, text_field))
            .collect();
        matched_texts.sort();

        // Exact match
        assert!(matched_texts.contains(&"programming language features".to_string()));

        // Single typo matches (within 1 edit distance)
        assert!(matched_texts.contains(&"programing language features".to_string()));
        assert!(matched_texts.contains(&"programming languag features".to_string()));

        // Two edits away from every query trigram
        assert!(!matched_texts.contains(&"programing languag features".to_string()));

        // Completely different words should NOT match
        assert!(!matched_texts.contains(&"coding syntax tools".to_string()));

        Ok(())
    }
}

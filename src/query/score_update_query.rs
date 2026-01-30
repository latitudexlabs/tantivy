use std::fmt;
use std::sync::Arc;

use crate::docset::{SeekDangerResult, COLLECT_BLOCK_BUFFER_LEN};
use crate::fastfield::AliveBitSet;
use crate::query::{EnableScoring, Explanation, Query, Scorer, Weight};
use crate::schema::Field;
use crate::{DocId, DocSet, Score, SegmentReader, Term};

/// Function type for updating scores (simple version)
pub type ScoreUpdateFn = Arc<dyn Fn(Score) -> Score + Send + Sync>;

/// Function type for context-aware scoring with field values and custom data
/// 
/// Parameters:
/// - `field_value`: The value from the specified field for the current document (None if not available)
/// - `base_scoring_value`: Optional reference value for comparison
/// - `original_score`: The original score from the underlying query
/// - `base_scoring_value_sort_order`: Optional sort order (0=None/based on score, 1=Asc, 2=Desc)
pub type ScoreUpdateWithContextFn =
    Arc<dyn Fn(Option<u64>, Option<u64>, Score, Option<u8>) -> Score + Send + Sync>;

/// Configuration for context-aware scoring
#[derive(Clone)]
pub struct ScoringContext {
    /// The field to read values from for scoring
    pub scoring_field: Field,
    /// Optional base value for comparison in scoring
    pub base_scoring_value: Option<u64>,
    /// Sort order: 0=None (based on score), 1=Asc, 2=Desc
    pub base_scoring_value_sort_order: Option<u8>,
}

impl ScoringContext {
    /// Create a new scoring context
    pub fn new(scoring_field: Field) -> Self {
        ScoringContext {
            scoring_field,
            base_scoring_value: None,
            base_scoring_value_sort_order: None,
        }
    }

    /// Set the base scoring value
    pub fn with_base_value(mut self, value: u64) -> Self {
        self.base_scoring_value = Some(value);
        self
    }

    /// Set the sort order (0=None, 1=Asc, 2=Desc)
    pub fn with_sort_order(mut self, order: u8) -> Self {
        self.base_scoring_value_sort_order = Some(order);
        self
    }
}

/// `ScoreUpdateQuery` is a wrapper over a query that allows updating the score
/// of matching documents using a custom function.
///
/// The document set matched by the `ScoreUpdateQuery` is strictly the same as the underlying query.
/// The score of each document is transformed by applying the score update function to the 
/// original score.
///
/// # Examples
///
/// ```rust
/// use tantivy::query::{ScoreUpdateQuery, AllQuery};
/// use std::sync::Arc;
///
/// // Create a query that squares all scores
/// let query = ScoreUpdateQuery::new(
///     Box::new(AllQuery),
///     Arc::new(|score| score * score)
/// );
/// ```
///
/// ```rust
/// use tantivy::query::{ScoreUpdateQuery, TermQuery};
/// use tantivy::{Term, Index};
/// use std::sync::Arc;
///
/// # fn example() -> tantivy::Result<()> {
/// # let index = Index::create_in_ram(tantivy::schema::Schema::builder().build());
/// # let term = Term::from_field_text(tantivy::schema::Field::from_field_id(0), "test");
/// // Apply a logarithmic transformation to scores
/// let query = ScoreUpdateQuery::new(
///     Box::new(TermQuery::new(term, Default::default())),
///     Arc::new(|score| (1.0 + score).ln())
/// );
/// # Ok(())
/// # }
/// ```
pub struct ScoreUpdateQuery {
    query: Box<dyn Query>,
    score_update_fn: ScoreUpdateFn,
}

/// `ScoreUpdateQueryWithContext` is an advanced wrapper that allows score updates
/// based on field values and custom context data.
///
/// This is useful when you want to adjust scores based on document field values,
/// such as boosting based on recency, popularity, or other custom metrics.
///
/// # Examples
///
/// ```rust,no_run
/// use tantivy::query::{ScoreUpdateQueryWithContext, ScoringContext, TermQuery};
/// use tantivy::schema::Field;
/// use tantivy::{Term, Index};
/// use std::sync::Arc;
///
/// # fn example() -> tantivy::Result<()> {
/// # let index = Index::create_in_ram(tantivy::schema::Schema::builder().build());
/// # let term = Term::from_field_text(Field::from_field_id(0), "test");
/// # let popularity_field = Field::from_field_id(1);
/// // Boost scores based on a popularity field
/// let context = ScoringContext::new(popularity_field)
///     .with_base_value(100);
///
/// let query = ScoreUpdateQueryWithContext::new(
///     Box::new(TermQuery::new(term, Default::default())),
///     context,
///     Arc::new(|field_val, base_val, score, _order| {
///         let boost = field_val.unwrap_or(0) as f32 / base_val.unwrap_or(100) as f32;
///         score * (1.0 + boost)
///     })
/// );
/// # Ok(())
/// # }
/// ```
pub struct ScoreUpdateQueryWithContext {
    query: Box<dyn Query>,
    scoring_context: ScoringContext,
    score_update_fn: ScoreUpdateWithContextFn,
}

impl ScoreUpdateQueryWithContext {
    /// Builds a context-aware score update query.
    ///
    /// # Arguments
    ///
    /// * `query` - The underlying query to wrap
    /// * `scoring_context` - Context containing field and parameters for scoring
    /// * `score_update_fn` - A function that takes field value, base value, score, and sort order
    pub fn new(
        query: Box<dyn Query>,
        scoring_context: ScoringContext,
        score_update_fn: ScoreUpdateWithContextFn,
    ) -> ScoreUpdateQueryWithContext {
        ScoreUpdateQueryWithContext {
            query,
            scoring_context,
            score_update_fn,
        }
    }

    /// Convenience method to create a context-aware score update query with a closure
    pub fn with_fn<F>(
        query: Box<dyn Query>,
        scoring_context: ScoringContext,
        f: F,
    ) -> ScoreUpdateQueryWithContext
    where
        F: Fn(Option<u64>, Option<u64>, Score, Option<u8>) -> Score + Send + Sync + 'static,
    {
        ScoreUpdateQueryWithContext::new(query, scoring_context, Arc::new(f))
    }
}

impl Clone for ScoreUpdateQueryWithContext {
    fn clone(&self) -> Self {
        ScoreUpdateQueryWithContext {
            query: self.query.box_clone(),
            scoring_context: self.scoring_context.clone(),
            score_update_fn: self.score_update_fn.clone(),
        }
    }
}

impl fmt::Debug for ScoreUpdateQueryWithContext {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ScoreUpdateWithContext(query={:?}, field={:?}, fn=<context-aware score function>)",
            self.query, self.scoring_context.scoring_field
        )
    }
}

impl Query for ScoreUpdateQueryWithContext {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        let weight_without_update = self.query.weight(enable_scoring)?;
        let updated_weight = if enable_scoring.is_scoring_enabled() {
            Box::new(ScoreUpdateWeightWithContext::new(
                weight_without_update,
                self.scoring_context.clone(),
                self.score_update_fn.clone(),
            ))
        } else {
            weight_without_update
        };
        Ok(updated_weight)
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        self.query.query_terms(visitor)
    }
}

impl ScoreUpdateQuery {
    /// Builds a score update query.
    ///
    /// # Arguments
    ///
    /// * `query` - The underlying query to wrap
    /// * `score_update_fn` - A function that takes the original score and returns the updated score
    pub fn new(query: Box<dyn Query>, score_update_fn: ScoreUpdateFn) -> ScoreUpdateQuery {
        ScoreUpdateQuery {
            query,
            score_update_fn,
        }
    }

    /// Convenience method to create a score update query with a simple closure
    pub fn with_fn<F>(query: Box<dyn Query>, f: F) -> ScoreUpdateQuery
    where
        F: Fn(Score) -> Score + Send + Sync + 'static,
    {
        ScoreUpdateQuery::new(query, Arc::new(f))
    }
}

impl Clone for ScoreUpdateQuery {
    fn clone(&self) -> Self {
        ScoreUpdateQuery {
            query: self.query.box_clone(),
            score_update_fn: self.score_update_fn.clone(),
        }
    }
}

impl fmt::Debug for ScoreUpdateQuery {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ScoreUpdate(query={:?}, fn=<score update function>)",
            self.query
        )
    }
}

impl Query for ScoreUpdateQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        let weight_without_update = self.query.weight(enable_scoring)?;
        let updated_weight = if enable_scoring.is_scoring_enabled() {
            Box::new(ScoreUpdateWeight::new(
                weight_without_update,
                self.score_update_fn.clone(),
            ))
        } else {
            weight_without_update
        };
        Ok(updated_weight)
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        self.query.query_terms(visitor)
    }
}

/// Weight associated to the ScoreUpdateQuery.
pub struct ScoreUpdateWeight {
    weight: Box<dyn Weight>,
    score_update_fn: ScoreUpdateFn,
}

impl ScoreUpdateWeight {
    /// Creates a new ScoreUpdateWeight.
    pub fn new(weight: Box<dyn Weight>, score_update_fn: ScoreUpdateFn) -> Self {
        ScoreUpdateWeight {
            weight,
            score_update_fn,
        }
    }
}

impl Weight for ScoreUpdateWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let inner_scorer = self.weight.scorer(reader, boost)?;
        Ok(Box::new(ScoreUpdateScorer::new(
            inner_scorer,
            self.score_update_fn.clone(),
        )))
    }

    fn explain(&self, reader: &SegmentReader, doc: u32) -> crate::Result<Explanation> {
        let underlying_explanation = self.weight.explain(reader, doc)?;
        let original_score = underlying_explanation.value();
        let updated_score = (self.score_update_fn)(original_score);
        let mut explanation = Explanation::new_with_string(
            format!(
                "ScoreUpdate (original={:.3} → updated={:.3})",
                original_score, updated_score
            ),
            updated_score,
        );
        explanation.add_detail(underlying_explanation);
        Ok(explanation)
    }

    fn count(&self, reader: &SegmentReader) -> crate::Result<u32> {
        self.weight.count(reader)
    }
}

/// Scorer that applies a score update function to the underlying scorer's scores.
pub struct ScoreUpdateScorer {
    underlying: Box<dyn Scorer>,
    score_update_fn: ScoreUpdateFn,
}

impl ScoreUpdateScorer {
    /// Creates a new ScoreUpdateScorer.
    pub fn new(underlying: Box<dyn Scorer>, score_update_fn: ScoreUpdateFn) -> Self {
        ScoreUpdateScorer {
            underlying,
            score_update_fn,
        }
    }
}

impl DocSet for ScoreUpdateScorer {
    fn advance(&mut self) -> DocId {
        self.underlying.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.underlying.seek(target)
    }

    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        self.underlying.seek_danger(target)
    }

    fn fill_buffer(&mut self, buffer: &mut [DocId; COLLECT_BLOCK_BUFFER_LEN]) -> usize {
        self.underlying.fill_buffer(buffer)
    }

    fn doc(&self) -> u32 {
        self.underlying.doc()
    }

    fn size_hint(&self) -> u32 {
        self.underlying.size_hint()
    }

    fn cost(&self) -> u64 {
        self.underlying.cost()
    }

    fn count(&mut self, alive_bitset: &AliveBitSet) -> u32 {
        self.underlying.count(alive_bitset)
    }

    fn count_including_deleted(&mut self) -> u32 {
        self.underlying.count_including_deleted()
    }
}

impl Scorer for ScoreUpdateScorer {
    #[inline]
    fn score(&mut self) -> Score {
        let original_score = self.underlying.score();
        (self.score_update_fn)(original_score)
    }
}

/// Weight associated to the ScoreUpdateQueryWithContext.
pub struct ScoreUpdateWeightWithContext {
    weight: Box<dyn Weight>,
    scoring_context: ScoringContext,
    score_update_fn: ScoreUpdateWithContextFn,
}

impl ScoreUpdateWeightWithContext {
    /// Creates a new ScoreUpdateWeightWithContext.
    pub fn new(
        weight: Box<dyn Weight>,
        scoring_context: ScoringContext,
        score_update_fn: ScoreUpdateWithContextFn,
    ) -> Self {
        ScoreUpdateWeightWithContext {
            weight,
            scoring_context,
            score_update_fn,
        }
    }
}

impl Weight for ScoreUpdateWeightWithContext {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let inner_scorer = self.weight.scorer(reader, boost)?;
        
        // Get the fast field reader for the scoring field
        let field_name = reader.schema().get_field_name(self.scoring_context.scoring_field);
        let fast_fields = reader.fast_fields();
        let field_reader = fast_fields.u64(field_name).ok();
        
        Ok(Box::new(ScoreUpdateScorerWithContext::new(
            inner_scorer,
            field_reader,
            self.scoring_context.clone(),
            self.score_update_fn.clone(),
        )))
    }

    fn explain(&self, reader: &SegmentReader, doc: u32) -> crate::Result<Explanation> {
        let underlying_explanation = self.weight.explain(reader, doc)?;
        let original_score = underlying_explanation.value();
        
        // Try to get field value for explanation
        let field_name = reader.schema().get_field_name(self.scoring_context.scoring_field);
        let fast_fields = reader.fast_fields();
        let field_value = fast_fields
            .u64(field_name)
            .ok()
            .and_then(|col| col.first(doc));
        
        let updated_score = (self.score_update_fn)(
            field_value,
            self.scoring_context.base_scoring_value,
            original_score,
            self.scoring_context.base_scoring_value_sort_order,
        );
        
        let mut explanation = Explanation::new_with_string(
            format!(
                "ScoreUpdateWithContext (field_val={:?}, base={:?}, original={:.3} → updated={:.3})",
                field_value,
                self.scoring_context.base_scoring_value,
                original_score,
                updated_score
            ),
            updated_score,
        );
        explanation.add_detail(underlying_explanation);
        Ok(explanation)
    }

    fn count(&self, reader: &SegmentReader) -> crate::Result<u32> {
        self.weight.count(reader)
    }
}

/// Scorer that applies a context-aware score update function based on field values.
pub struct ScoreUpdateScorerWithContext {
    underlying: Box<dyn Scorer>,
    field_reader: Option<crate::columnar::Column<u64>>,
    scoring_context: ScoringContext,
    score_update_fn: ScoreUpdateWithContextFn,
}

impl ScoreUpdateScorerWithContext {
    /// Creates a new ScoreUpdateScorerWithContext.
    pub fn new(
        underlying: Box<dyn Scorer>,
        field_reader: Option<crate::columnar::Column<u64>>,
        scoring_context: ScoringContext,
        score_update_fn: ScoreUpdateWithContextFn,
    ) -> Self {
        ScoreUpdateScorerWithContext {
            underlying,
            field_reader,
            scoring_context,
            score_update_fn,
        }
    }
}

impl DocSet for ScoreUpdateScorerWithContext {
    fn advance(&mut self) -> DocId {
        self.underlying.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.underlying.seek(target)
    }

    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        self.underlying.seek_danger(target)
    }

    fn fill_buffer(&mut self, buffer: &mut [DocId; COLLECT_BLOCK_BUFFER_LEN]) -> usize {
        self.underlying.fill_buffer(buffer)
    }

    fn doc(&self) -> u32 {
        self.underlying.doc()
    }

    fn size_hint(&self) -> u32 {
        self.underlying.size_hint()
    }

    fn cost(&self) -> u64 {
        self.underlying.cost()
    }

    fn count(&mut self, alive_bitset: &AliveBitSet) -> u32 {
        self.underlying.count(alive_bitset)
    }

    fn count_including_deleted(&mut self) -> u32 {
        self.underlying.count_including_deleted()
    }
}

impl Scorer for ScoreUpdateScorerWithContext {
    #[inline]
    fn score(&mut self) -> Score {
        let original_score = self.underlying.score();
        let doc_id = self.underlying.doc();
        
        // Read the field value for the current document
        let field_value = self.field_reader
            .as_ref()
            .and_then(|reader| reader.first(doc_id));
        
        (self.score_update_fn)(
            field_value,
            self.scoring_context.base_scoring_value,
            original_score,
            self.scoring_context.base_scoring_value_sort_order,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::ScoreUpdateQuery;
    use crate::query::{AllQuery, Query};
    use crate::schema::{Schema, STORED, TEXT};
    use crate::{DocAddress, Index, IndexWriter, TantivyDocument};
    use std::sync::Arc;

    #[test]
    fn test_score_update_query_basic() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT | STORED);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema.clone());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;

        let mut doc = TantivyDocument::new();
        doc.add_text(text_field, "hello world");
        index_writer.add_document(doc)?;
        
        let mut doc = TantivyDocument::new();
        doc.add_text(text_field, "hello");
        index_writer.add_document(doc)?;
        
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Create a query that doubles all scores
        let query = ScoreUpdateQuery::new(Box::new(AllQuery), Arc::new(|score| score * 2.0));

        let explanation = query.explain(&searcher, DocAddress::new(0, 0u32))?;
        // AllQuery returns score of 1.0, so after doubling it should be 2.0
        assert_eq!(explanation.value(), 2.0);

        Ok(())
    }

    #[test]
    fn test_score_update_query_with_fn() -> crate::Result<()> {
        let schema = Schema::builder().build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(TantivyDocument::new())?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Create a query using the convenience with_fn method
        let query = ScoreUpdateQuery::with_fn(Box::new(AllQuery), |score| score + 5.0);

        let explanation = query.explain(&searcher, DocAddress::new(0, 0u32))?;
        // AllQuery returns score of 1.0, so after adding 5.0 it should be 6.0
        assert_eq!(explanation.value(), 6.0);

        Ok(())
    }

    #[test]
    fn test_score_update_query_logarithmic() -> crate::Result<()> {
        let schema = Schema::builder().build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(TantivyDocument::new())?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Apply logarithmic transformation
        let query =
            ScoreUpdateQuery::with_fn(Box::new(AllQuery), |score| (1.0 + score).ln());

        let explanation = query.explain(&searcher, DocAddress::new(0, 0u32))?;
        // AllQuery returns score of 1.0, so ln(1 + 1) = ln(2) ≈ 0.693
        let expected = (1.0 + 1.0_f32).ln();
        assert!((explanation.value() - expected).abs() < 0.001);

        Ok(())
    }

    #[test]
    fn test_score_update_query_clamp() -> crate::Result<()> {
        let schema = Schema::builder().build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(TantivyDocument::new())?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Clamp scores to a maximum of 0.5
        let query = ScoreUpdateQuery::with_fn(Box::new(AllQuery), |score| score.min(0.5));

        let explanation = query.explain(&searcher, DocAddress::new(0, 0u32))?;
        // AllQuery returns score of 1.0, but we clamp it to 0.5
        assert_eq!(explanation.value(), 0.5);

        Ok(())
    }

    #[test]
    fn test_score_update_query_explain() -> crate::Result<()> {
        let schema = Schema::builder().build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(TantivyDocument::new())?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = ScoreUpdateQuery::with_fn(Box::new(AllQuery), |score| score * 3.0);

        let explanation = query.explain(&searcher, DocAddress::new(0, 0u32))?;
        let json = explanation.to_pretty_json();

        // The explanation should show the transformation
        assert!(json.contains("ScoreUpdate"));
        assert!(json.contains("original=1.000"));
        assert!(json.contains("updated=3.000"));
        assert_eq!(explanation.value(), 3.0);

        Ok(())
    }

    #[test]
    fn test_score_update_query_with_context() -> crate::Result<()> {
        use super::{ScoreUpdateQueryWithContext, ScoringContext};
        use crate::schema::FAST;

        let mut schema_builder = Schema::builder();
        let popularity_field = schema_builder.add_u64_field("popularity", FAST);
        let schema = schema_builder.build();
        
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        
        // Add documents with different popularity values
        let mut doc1 = TantivyDocument::new();
        doc1.add_u64(popularity_field, 100);
        index_writer.add_document(doc1)?;
        
        let mut doc2 = TantivyDocument::new();
        doc2.add_u64(popularity_field, 200);
        index_writer.add_document(doc2)?;
        
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Create a scoring context with base value
        let context = ScoringContext::new(popularity_field)
            .with_base_value(100);

        // Boost scores based on popularity
        let query = ScoreUpdateQueryWithContext::with_fn(
            Box::new(AllQuery),
            context,
            |field_val, base_val, score, _order| {
                let boost = field_val.unwrap_or(0) as f32 / base_val.unwrap_or(100) as f32;
                score * (1.0 + boost)
            }
        );

        // Test first document (popularity=100)
        let explanation = query.explain(&searcher, DocAddress::new(0, 0u32))?;
        // AllQuery score = 1.0, popularity = 100, base = 100
        // boost = 100/100 = 1.0, score = 1.0 * (1.0 + 1.0) = 2.0
        assert_eq!(explanation.value(), 2.0);

        // Test second document (popularity=200)
        let explanation = query.explain(&searcher, DocAddress::new(0, 1u32))?;
        // AllQuery score = 1.0, popularity = 200, base = 100
        // boost = 200/100 = 2.0, score = 1.0 * (1.0 + 2.0) = 3.0
        assert_eq!(explanation.value(), 3.0);

        Ok(())
    }

    #[test]
    fn test_score_update_query_with_context_and_sort_order() -> crate::Result<()> {
        use super::{ScoreUpdateQueryWithContext, ScoringContext};
        use crate::schema::FAST;

        let mut schema_builder = Schema::builder();
        let timestamp_field = schema_builder.add_u64_field("timestamp", FAST);
        let schema = schema_builder.build();
        
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        
        // Add documents with different timestamps
        let mut doc1 = TantivyDocument::new();
        doc1.add_u64(timestamp_field, 1000);
        index_writer.add_document(doc1)?;
        
        let mut doc2 = TantivyDocument::new();
        doc2.add_u64(timestamp_field, 2000);
        index_writer.add_document(doc2)?;
        
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Create a scoring context with sort order (1 = Asc, 2 = Desc)
        let context = ScoringContext::new(timestamp_field)
            .with_base_value(1500)
            .with_sort_order(2); // Descending - prefer newer timestamps

        // Adjust scores based on recency
        let query = ScoreUpdateQueryWithContext::with_fn(
            Box::new(AllQuery),
            context,
            |field_val, base_val, score, sort_order| {
                if let (Some(val), Some(base)) = (field_val, base_val) {
                    match sort_order {
                        Some(2) => {
                            // Descending - boost higher values
                            let recency_boost = (val as f32 / base as f32).max(0.5);
                            score * recency_boost
                        }
                        Some(1) => {
                            // Ascending - boost lower values
                            let boost = (base as f32 / val as f32).max(0.5);
                            score * boost
                        }
                        _ => score,
                    }
                } else {
                    score
                }
            }
        );

        // Test document with timestamp 2000 (newer than base 1500)
        let explanation = query.explain(&searcher, DocAddress::new(0, 1u32))?;
        let expected_boost = 2000.0 / 1500.0;
        assert!((explanation.value() - expected_boost).abs() < 0.001);

        Ok(())
    }
}

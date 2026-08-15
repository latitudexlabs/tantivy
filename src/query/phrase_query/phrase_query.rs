use super::PhraseWeight;
use crate::query::bm25::Bm25Weight;
use crate::query::ngram_query_optimizer::NgramQueryOptimizer;
use crate::query::{BooleanQuery, ConstScoreQuery, EnableScoring, Occur, Query, Weight};
use crate::schema::{Field, IndexRecordOption, Term};
use std::sync::Arc;

/// `PhraseQuery` matches a specific sequence of words.
///
/// For instance, the phrase query for `"part time"` will match
/// the sentence:
///
/// **Alan just got a part time job.**
///
/// On the other hand it will not match the sentence.
///
/// **This is my favorite part of the job.**
///
/// [Slop](PhraseQuery::set_slop) allows leniency in term proximity
/// for some performance trade-off.
///
/// Using a `PhraseQuery` on a field requires positions
/// to be indexed for this field.
#[derive(Clone, Debug)]
pub struct PhraseQuery {
    field: Field,
    phrase_terms: Vec<(usize, Term)>,
    slop: u32,
}

impl PhraseQuery {
    /// Creates a new `PhraseQuery` given a list of terms.
    ///
    /// There must be at least two terms, and all terms
    /// must belong to the same field.
    /// Offset for each term will be same as index in the Vector
    pub fn new(terms: Vec<Term>) -> PhraseQuery {
        let terms_with_offset = terms.into_iter().enumerate().collect();
        PhraseQuery::new_with_offset(terms_with_offset)
    }

    /// Creates a new `PhraseQuery` given a list of terms and their offsets.
    ///
    /// Can be used to provide custom offset for each term.
    pub fn new_with_offset(terms: Vec<(usize, Term)>) -> PhraseQuery {
        PhraseQuery::new_with_offset_and_slop(terms, 0)
    }

    /// Creates a new `PhraseQuery` given a list of terms, their offsets and a slop
    pub fn new_with_offset_and_slop(mut terms: Vec<(usize, Term)>, slop: u32) -> PhraseQuery {
        assert!(
            terms.len() > 1,
            "A phrase query is required to have strictly more than one term."
        );
        terms.sort_by_key(|&(offset, _)| offset);
        let field = terms[0].1.field();
        assert!(
            terms[1..].iter().all(|term| term.1.field() == field),
            "All terms from a phrase query must belong to the same field"
        );
        PhraseQuery {
            field,
            phrase_terms: terms,
            slop,
        }
    }

    /// Creates a new `PhraseQuery` given a list of terms, their offsets and a slop
    pub fn new_with_offset_and_slop_nosort(terms: Vec<(usize, Term)>, slop: u32) -> PhraseQuery {
        assert!(
            terms.len() > 1,
            "A phrase query is required to have strictly more than one term."
        );
        let field = terms[0].1.field();
        assert!(
            terms[1..].iter().all(|term| term.1.field() == field),
            "All terms from a phrase query must belong to the same field"
        );
        PhraseQuery {
            field,
            phrase_terms: terms,
            slop,
        }
    }

    /// Slop allowed for the phrase.
    ///
    /// The query will match if its terms are separated by `slop` terms at most.
    /// The slop can be considered a budget between all terms.
    /// E.g. "A B C" with slop 1 allows "A X B C", "A B X C", but not "A X B X C".
    ///
    /// Transposition costs 2, e.g. "A B" with slop 1 will not match "B A" but it would with slop 2
    /// Transposition is not a special case, in the example above A is moved 1 position and B is
    /// moved 1 position, so the slop is 2.
    ///
    /// As a result slop works in both directions, so the order of the terms may changed as long as
    /// they respect the slop.
    ///
    /// By default the slop is 0 meaning query terms need to be adjacent.
    pub fn set_slop(&mut self, value: u32) {
        self.slop = value;
    }

    /// The [`Field`] this `PhraseQuery` is targeting.
    pub fn field(&self) -> Field {
        self.field
    }

    /// `Term`s in the phrase without the associated offsets.
    pub fn phrase_terms(&self) -> Vec<Term> {
        self.phrase_terms
            .iter()
            .map(|(_, term)| term.clone())
            .collect::<Vec<Term>>()
    }

    /// The `Term`s in the phrase with their offsets.
    pub fn get_phrase_terms_with_offsets(&self) -> &[(usize, Term)] {
        &self.phrase_terms
    }

    /// Returns the [`PhraseWeight`] for the given phrase query given a specific `searcher`.
    ///
    /// This function is the same as [`Query::weight()`] except it returns
    /// a specialized type [`PhraseWeight`] instead of a Boxed trait.
    pub(crate) fn phrase_weight(
        &self,
        enable_scoring: EnableScoring<'_>,
    ) -> crate::Result<PhraseWeight> {
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
                "Applied phrase query on field {field_name:?}, which does not have positions \
                 indexed"
            )));
        }
        let terms = self.phrase_terms();
        let bm25_weight_opt = match enable_scoring {
            EnableScoring::Enabled {
                statistics_provider,
                ..
            } => Some(Bm25Weight::for_terms(statistics_provider, &terms)?),
            EnableScoring::Disabled { .. } => None,
        };
        let mut weight = PhraseWeight::new(self.phrase_terms.clone(), bm25_weight_opt);
        if self.slop > 0 {
            weight.slop(self.slop);
        }
        Ok(weight)
    }
}

impl Query for PhraseQuery {
    /// Create the weight associated with a query.
    ///
    /// See [`Weight`].
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        // When the field indexes word ngrams, intersect the phrase with a
        // cheap ngram prefilter: any document containing the phrase also
        // contains its consecutive bigrams/trigrams, so position
        // verification only runs on documents that pass the prefilter.
        let schema = enable_scoring.schema();
        let optimizer = NgramQueryOptimizer::new(Arc::new(schema.clone()));

        if let Some(ngram_prefilter) =
            optimizer.phrase_ngram_prefilter(self.field, &self.phrase_terms, self.slop)
        {
            // The constant 0.0 score keeps the composed score identical to
            // a plain phrase query; ExactPhraseWeightQuery bypasses this
            // method so the prefilter is not re-applied recursively.
            let filtered_query = BooleanQuery::new(vec![
                (
                    Occur::Must,
                    Box::new(ConstScoreQuery::new(ngram_prefilter, 0.0)) as Box<dyn Query>,
                ),
                (
                    Occur::Must,
                    Box::new(ExactPhraseWeightQuery(self.clone())) as Box<dyn Query>,
                ),
            ]);
            return filtered_query.weight(enable_scoring);
        }

        // No usable ngram index: plain position-based phrase query.
        let phrase_weight = self.phrase_weight(enable_scoring)?;
        Ok(Box::new(phrase_weight))
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        for (_, term) in &self.phrase_terms {
            visitor(term, true);
        }
    }
}

/// Wrapper building the exact position-based weight of a phrase query,
/// bypassing the ngram prefilter rewrite in [`PhraseQuery::weight`]. Used as
/// the verification leg of the prefiltered composition.
#[derive(Clone, Debug)]
struct ExactPhraseWeightQuery(PhraseQuery);

impl Query for ExactPhraseWeightQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(self.0.phrase_weight(enable_scoring)?))
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        self.0.query_terms(visitor);
    }
}

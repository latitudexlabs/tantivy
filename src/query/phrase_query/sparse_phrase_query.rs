use super::SparsePhraSeWeight;
use crate::query::bm25::Bm25Weight;
use crate::query::{EnableScoring, Query, Weight};
use crate::schema::{Field, IndexRecordOption, Term};

/// `SparsePhraSeQuery` matches documents containing query terms in order,
/// even if some terms are missing or not adjacent.
///
/// Unlike `PhraseQuery` which requires all terms to be present and adjacent (with optional slop),
/// `SparsePhraSeQuery` matches if the query terms appear in the correct order in a document,
/// regardless of whether all terms are present or how far apart they are.
///
/// **Scoring:**
/// - Documents are scored based on the number of consecutive terms that appear in order
/// - Higher scores for documents with more terms matching in order
/// - Score = (matched_terms / total_terms) * BM25_score
///
/// **Examples (for query "quick brown fox"):**
/// - "quick brown fox" → ✅ Match (all 3 terms in order)
/// - "quick brown" → ✅ Match (2 terms in order, score ≈ 0.67)
/// - "quick fox" → ✅ Match (2 terms in order, score ≈ 0.67)
/// - "the quick brown fox" → ✅ Match (all 3 terms in order)
/// - "fox brown quick" → ❌ No match (wrong order)
/// - "brown quick fox" → ❌ No match (wrong order)
#[derive(Clone, Debug)]
pub struct SparsePhraSeQuery {
    field: Field,
    phrase_terms: Vec<(usize, Term)>,
}

impl SparsePhraSeQuery {
    /// Creates a new `SparsePhraSeQuery` given a list of terms.
    ///
    /// There must be at least one term, and all terms
    /// must belong to the same field.
    /// All terms will be at offset 0 since we allow arbitrary skipping
    pub fn new(terms: Vec<Term>) -> SparsePhraSeQuery {
        let terms_with_offset = terms.into_iter().map(|t| (0, t)).collect();
        SparsePhraSeQuery::new_with_offset(terms_with_offset)
    }

    /// Creates a new `SparsePhraSeQuery` given a list of terms and their offsets.
    ///
    /// Can be used to provide custom offset for each term.
    /// Requires at least 2 terms to work properly.
    pub fn new_with_offset(mut terms: Vec<(usize, Term)>) -> SparsePhraSeQuery {
        assert!(
            terms.len() >= 2,
            "A sparse phrase query must have at least two terms."
        );
        terms.sort_by_key(|&(offset, _)| offset);
        let field = terms[0].1.field();
        assert!(
            terms[1..].iter().all(|term| term.1.field() == field),
            "All terms from a sparse phrase query must belong to the same field"
        );
        SparsePhraSeQuery {
            field,
            phrase_terms: terms,
        }
    }

    /// The [`Field`] this `SparsePhraSeQuery` is targeting.
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

    pub(crate) fn sparse_phrase_weight(
        &self,
        enable_scoring: EnableScoring<'_>,
    ) -> crate::Result<SparsePhraSeWeight> {
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
                "Applied sparse phrase query on field {field_name:?}, which does not have positions indexed"
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
        Ok(SparsePhraSeWeight::new(self.phrase_terms.clone(), bm25_weight_opt))
    }
}

impl Query for SparsePhraSeQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        let weight = self.sparse_phrase_weight(enable_scoring)?;
        Ok(Box::new(weight))
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        for (_, term) in &self.phrase_terms {
            visitor(term, true);
        }
    }
}
